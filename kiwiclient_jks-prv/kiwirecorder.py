#!/usr/bin/env python
## -*- python -*-

##
## FIXME:
## netcat should support the usual suspects:
##      IQ-swap, endian-reversal, option to include GPS data, ...
##

import array, logging, os, struct, sys, time, copy, threading, os
import gc
import math
import numpy as np
from copy import copy
from traceback import print_exc
import png
from kiwi import KiwiSDRStream, KiwiWorker
import optparse as optparse
from optparse import OptionParser
from optparse import OptionGroup

# 全局变量和辅助函数
HAS_PyYAML = True
try:
    ## needed for the --agc-yaml option
    import yaml
    if yaml.__version__.split('.')[0] < '5':
        print('wrong PyYAML version: %s < 5; PyYAML is only needed when using the --agc-yaml option' % yaml.__version__)
        raise ImportError
except ImportError:
    ## (only) when needed an exception is raised, see below
    HAS_PyYAML = False

HAS_RESAMPLER = True
try:
    ## if available use libsamplerate for resampling
    from samplerate import Resampler
except ImportError:
    ## otherwise linear interpolation is used
    HAS_RESAMPLER = False

try:
    if os.environ['USE_LIBSAMPLERATE'] == 'False':
        HAS_RESAMPLER = False
except KeyError:
    pass

def clamp(x, xmin, xmax):
    if x < xmin:
        x = xmin
    if x > xmax:
        x = xmax
    return x

def by_dBm(e):
    return e['dBm']

# WAV 文件头生成
def _write_wav_header(fp, filesize, samplerate, num_channels, is_kiwi_wav):
    samplerate = int(samplerate+0.5)
    fp.write(struct.pack('<4sI4s', b'RIFF', filesize - 8, b'WAVE'))
    bits_per_sample = 16
    byte_rate       = samplerate * num_channels * bits_per_sample // 8
    block_align     = num_channels * bits_per_sample // 8
    fp.write(struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, num_channels, samplerate, byte_rate, block_align, bits_per_sample))
    if not is_kiwi_wav:
        fp.write(struct.pack('<4sI', b'data', filesize - 12 - 8 - 16 - 8))

# 环形缓冲区类,实现了一个环形缓冲区，用于存储和处理数据
class RingBuffer(object):
    def __init__(self, len):
        self._array = np.zeros(len, dtype='float64')
        self._index = 0
        self._is_filled = False

    def insert(self, sample):
        self._array[self._index] = sample
        self._index += 1
        if self._index == len(self._array):
            self._is_filled = True
            self._index = 0

    def is_filled(self):
        return self._is_filled

    def applyFn(self, fn):
        return fn(self._array)

    def max_abs(self):
        return np.max(np.abs(self._array))

# GNSS 性能分析类,分析 GNSS 性能，记录 GPS 解决方案的变化和时间漂移。
class GNSSPerformance(object):
    def __init__(self):
        self._last_solution = -1
        self._last_ts = -1
        self._num_frames = 0
        self._buffer_dt_per_frame = RingBuffer(10)
        self._buffer_num_frames   = RingBuffer(10)

    def analyze(self, filename, gps):
        ## gps = {'last_gps_solution': 1, 'dummy': 0, 'gpsnsec': 886417795, 'gpssec': 466823}
        self._num_frames += 1
        if gps['last_gps_solution'] == 0 and self._last_solution != 0:
            ts = gps['gpssec'] + 1e-9 * gps['gpsnsec']
            msg_gnss_drift = ''
            dt = 0
            if self._last_ts != -1:
                dt = ts - self._last_ts
                if dt < -12*3600*7:
                    dt += 24*3600*7
                if abs(dt) < 10:
                    self._buffer_dt_per_frame.insert(dt / self._num_frames)
                    self._buffer_num_frames.insert(self._num_frames)
                if self._buffer_dt_per_frame.is_filled():
                    std_dt_per_frame  = self._buffer_dt_per_frame.applyFn(np.std)
                    mean_num_frames   = self._buffer_num_frames.applyFn(np.mean)
                    msg_gnss_drift = 'std(clk drift)= %5.1f m' % (3e8 * std_dt_per_frame * mean_num_frames)

            logging.info('%s: (%2d,%3d) t_gnss= %16.9f dt= %16.9f %s'
                         % (filename, self._last_solution, self._num_frames, ts, dt, msg_gnss_drift))
            self._num_frames = 0
            self._last_ts    = ts

        self._last_solution = gps['last_gps_solution']

#  静音检测类,实现静音检测功能，用于控制录音的开始和结束。
class Squelch(object):
    def __init__(self, options):
        self._status_msg  = not options.quiet
        self._threshold   = options.sq_thresh
        self._squelch_tail = options.squelch_tail ## in seconds
        self._ring_buffer = RingBuffer(65)
        self._squelch_on_seq = None
        self.set_sample_rate(12000.0) ## default setting

    def set_threshold(self, threshold):
        self._threshold = threshold
        return self

    def set_sample_rate(self, fs):
        self._tail_delay  = round(self._squelch_tail*fs/512) ## seconds to number of buffers

    def process(self, seq, rssi):
        if not self._ring_buffer.is_filled() or self._squelch_on_seq is None:
            self._ring_buffer.insert(rssi)
        if not self._ring_buffer.is_filled():
            return False
        median_nf   = self._ring_buffer.applyFn(np.median)
        rssi_thresh = median_nf + self._threshold
        is_open     = self._squelch_on_seq is not None
        if is_open:
            rssi_thresh -= 6
        rssi_green = rssi >= rssi_thresh
        if rssi_green:
            self._squelch_on_seq = seq
            is_open = True
        if self._status_msg:
            sys.stdout.write('\r Median: %6.1f Thr: %6.1f %s ' % (median_nf, rssi_thresh, ("s", "S")[is_open]))
            sys.stdout.flush()
            self._need_nl = True
        if not is_open:
            return False
        if seq > self._squelch_on_seq + self._tail_delay:
            logging.info("\nSquelch closed")
            self._squelch_on_seq = None
            return False
        return is_open

## -------------------------------------------------------------------------------------------------
# 音频录制类，继承自 KiwiSDRStream 类
class KiwiSoundRecorder(KiwiSDRStream):
    def __init__(self, options):
        super(KiwiSoundRecorder, self).__init__()  # 调用父类的初始化方法
        self._options = options  # 初始化选项参数
        self._type = 'SND'  # 设置类型为音频
        freq = options.frequency  # 获取频率选项
        # logging.info("%s:%s freq=%d" % (options.server_host, options.server_port, freq))  # 打印服务器主机、端口和频率
        self._freq = freq  # 设置频率
        self._freq_offset = options.freq_offset  # 设置频率偏移
        self._start_ts = None  # 初始化开始时间戳
        self._start_time = None  # 初始化开始时间
        self._squelch = Squelch(self._options) if options.sq_thresh is not None else None  # 初始化静噪器
        if options.scan_yaml is not None:  # 如果有扫描配置文件
            self._squelch = [Squelch(options).set_threshold(options.scan_yaml['threshold']) for _ in range(len(options.scan_yaml['frequencies']))]  # 初始化多个静噪器
        self._last_gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], [0,0,0,0]))  # 初始化 GPS 数据
        self._resampler = None  # 初始化重采样器
        self._kiwi_samplerate = False  # 标记是否使用 Kiwi 重采样器
        self._gnss_performance = GNSSPerformance()  # 初始化 GNSS 性能分析器

    def _setup_rx_params(self):
        if self._options.no_api:  # 如果禁用 API
            self._setup_no_api()  # 调用无 API 配置方法
            return
        self.set_name(self._options.user)  # 设置用户名

        self.set_freq(self._freq)  # 设置频率

        if self._options.agc_gain is not None:  # 如果设置了固定增益
            self.set_agc(on=False, gain=self._options.agc_gain)  # 关闭 AGC 并设置增益
        elif self._options.agc_yaml_file is not None:  # 如果有 AGC 配置文件
            self.set_agc(**self._options.agc_yaml)  # 使用配置文件中的 AGC 参数
        else:  # 默认启用 AGC
            self.set_agc(on=True)  # 启用 AGC

        if self._options.compression is False:  # 如果禁用压缩
            self._set_snd_comp(False)  # 禁用音频压缩

        if self._options.nb is True or self._options.nb_test is True:  # 如果启用了噪声抑制
            gate = self._options.nb_gate  # 获取噪声门限
            if gate < 100 or gate > 5000:  # 检查门限范围
                gate = 100  # 设置默认值
            nb_thresh = self._options.nb_thresh  # 获取噪声阈值
            if nb_thresh < 0 or nb_thresh > 100:  # 检查阈值范围
                nb_thresh = 50  # 设置默认值
            self.set_noise_blanker(gate, nb_thresh)  # 设置噪声抑制参数

        if self._options.de_emp is True:  # 如果启用了去加重
            self.set_de_emp(1)  # 启用去加重

        self._output_sample_rate = self._sample_rate  # 设置输出采样率

        if self._squelch:  # 如果有静噪器
            if type(self._squelch) == list:  # 如果是扫描模式
                for s in self._squelch:
                    s.set_sample_rate(self._sample_rate)  # 设置每个静噪器的采样率
            else:
                self._squelch.set_sample_rate(self._sample_rate)  # 设置单个静噪器的采样率

        if self._options.test_mode:  # 如果是测试模式
            self._set_stats()  # 设置统计信息

        if self._options.resample > 0 and not HAS_RESAMPLER:  # 如果需要重采样且没有重采样库
            self._setup_resampler()  # 设置重采样器

        if self._options.devel is not None:  # 如果有开发选项
            for pair in self._options.devel.split(','):  # 分割开发选项
                vals = pair.split(':')  # 分割键值对
                if len(vals) != 2:  # 检查格式
                    raise Exception("--devel arg \"%s\" needs to be format \"[0-7]:float_value\"" % pair)
                which = int(vals[0])  # 获取索引
                value = float(vals[1])  # 获取值
                if not (0 <= which <= 7):  # 检查索引范围
                    raise Exception("--devel first arg \"%d\" of \"[0-7]:float_value\" is out of range" % which)
                self._send_message('SET devl.p%d=%.9g' % (which, value))  # 发送开发命令

    def _setup_resampler(self):
        if self._options.resample > 0:  # 如果需要重采样
            if not HAS_RESAMPLER:  # 如果没有重采样库
                self._output_sample_rate = self._options.resample  # 设置输出采样率
                self._ratio = float(self._output_sample_rate) / self._sample_rate  # 计算重采样比例
                logging.warning("CAUTION: libsamplerate not available; low-quality linear interpolation will be used for resampling.")  # 警告低质量重采样
                logging.warning("See the README file instructions to build the Kiwi samplerate module.")  # 提示安装高质量重采样模块
                logging.warning('resampling from %g to %d Hz (ratio=%f)' % (self._sample_rate, self._options.resample, self._ratio))  # 打印重采样信息
            else:
                if hasattr(self._resampler, 'kiwi_samplerate'):  # 如果有 Kiwi 重采样器
                    self._kiwi_samplerate = True  # 标记使用 Kiwi 重采样器
                if self._kiwi_samplerate is True:  # 如果使用 Kiwi 重采样器
                    logging.warning("Using Kiwi high-quality samplerate module.")  # 警告使用高质量重采样模块
                    self._ratio = self._options.resample / self._sample_rate  # 计算重采样比例
                else:
                    fs = 10 * round(self._sample_rate / 10)  # 四舍五入采样率
                    ratio = self._options.resample / fs  # 计算重采样比例
                    n = 512  # Kiwi SDR 块长度
                    m = round(ratio * n)  # 计算整数倍
                    self._ratio = m / n  # 计算最终重采样比例
                    logging.warning('CAUTION: using python-samplerate instead of Kiwi samplerate module containing fixes.')  # 警告使用 Python 重采样模块
                self._output_sample_rate = self._ratio * self._sample_rate  # 计算输出采样率
                logging.warning('resampling from %g to %g Hz (ratio=%f)' % (self._sample_rate, self._output_sample_rate, self._ratio))  # 打印重采样信息

    def _squelch_status(self, seq, samples, rssi):
        if not self._options.quiet:  # 如果不静默
            sys.stdout.write('\rBlock: %08x, RSSI: %6.1f ' % (seq, rssi))  # 打印块号和 RSSI
            self._need_nl = True  # 标记需要换行
        if self._squelch and type(self._squelch) == list:  # 如果是扫描模式
            if self._options.quiet:
                sys.stdout.write('\r')  # 清除行
            sys.stdout.write(" scan: [%s] freq = %g kHz      " % (self._options.scan_state, self._freq))  # 打印扫描状态和频率
            self._need_nl = True  # 标记需要换行
        sys.stdout.flush()  # 刷新输出

        is_open = True  # 初始化静噪状态为打开
        if self._squelch:  # 如果有静噪器
            if type(self._squelch) == list:  # 如果是扫描模式
                if self._options.scan_state == "WAIT":  # 如果等待状态
                    is_open = False  # 关闭静噪
                    now = time.time()  # 获取当前时间
                    if now - self._options.scan_time > self._options.scan_yaml['wait']:  # 如果等待时间超过设定值
                        self._options.scan_time = now  # 更新扫描时间
                        self._options.scan_state = 'DWELL'  # 切换到驻留状态
                if self._options.scan_state == 'DWELL':  # 如果驻留状态
                    is_open = self._squelch[self._options.scan_index].process(seq, rssi)  # 处理静噪
                    now = time.time()  # 获取当前时间
                    if not is_open and now - self._options.scan_time > self._options.scan_yaml['dwell']:  # 如果静噪关闭且驻留时间超过设定值
                        self._options.scan_index = (self._options.scan_index + 1) % len(self._options.scan_yaml['frequencies'])  # 更新扫描索引
                        self.set_freq(self._options.scan_yaml['frequencies'][self._options.scan_index])  # 设置新频率
                        self._options.scan_time = now  # 更新扫描时间
                        self._options.scan_state = 'WAIT'  # 切换到等待状态
                        self._start_ts = None  # 重置开始时间戳
                        self._start_time = None  # 重置开始时间
            else:  # 单通道模式
                is_open = self._squelch.process(seq, rssi)  # 处理静噪
                if not is_open:  # 如果静噪关闭
                    self._start_ts = None  # 重置开始时间戳
                    self._start_time = None  # 重置开始时间
        return is_open  # 返回静噪状态

    def _process_audio_samples(self, seq, samples, rssi):
        is_open = self._squelch_status(seq, samples, rssi)  # 获取静噪状态
        if not is_open:  # 如果静噪关闭
            return  # 直接返回

        if self._options.resample > 0:  # 如果需要重采样
            if HAS_RESAMPLER:  # 如果有重采样库
                if self._resampler is None:  # 如果没有初始化重采样器
                    self._resampler = Resampler(converter_type='sinc_best')  # 初始化重采样器
                    self._setup_resampler()  # 设置重采样器
                samples = np.round(self._resampler.process(samples, ratio=self._ratio)).astype(np.int16)  # 重采样并转换数据类型
            else:  # 如果没有重采样库
                n = len(samples)  # 获取样本数量
                xa = np.arange(round(n * self._ratio)) / self._ratio  # 计算插值点
                xp = np.arange(n)  # 计算原始点
                samples = np.round(np.interp(xa, xp, samples)).astype(np.int16)  # 线性插值并转换数据类型

        self._write_samples(samples, {})  # 写入样本

    def _process_iq_samples(self, seq, samples, rssi, gps):
        if not self._squelch_status(seq, samples, rssi):  # 获取静噪状态
            return  # 如果静噪关闭，直接返回

        self._last_gps = gps  # 更新 GPS 数据

        s = np.zeros(2 * len(samples), dtype=np.int16)  # 初始化 IQ 样本数组
        s[0::2] = np.real(samples).astype(np.int16)  # 填充实部
        s[1::2] = np.imag(samples).astype(np.int16)  # 填充实部

        if self._options.resample > 0:  # 如果需要重采样
            if HAS_RESAMPLER:  # 如果有重采样库
                if self._resampler is None:  # 如果没有初始化重采样器
                    self._resampler = Resampler(channels=2, converter_type='sinc_best')  # 初始化重采样器
                    self._setup_resampler()  # 设置重采样器
                s = self._resampler.process(s.reshape(len(samples), 2), ratio=self._ratio)  # 重采样并转换数据类型
                s = np.round(s.flatten()).astype(np.int16)  # 展平并转换数据类型
            else:  # 如果没有重采样库
                n = len(samples)  # 获取样本数量
                m = int(round(n * self._ratio))  # 计算目标样本数量
                xa = np.arange(m) / self._ratio  # 计算插值点
                xp = np.arange(n)  # 计算原始点
                s = np.zeros(2 * m, dtype=np.int16)  # 初始化目标数组
                s[0::2] = np.round(np.interp(xa, xp, np.real(samples))).astype(np.int16)  # 线性插值实部
                s[1::2] = np.round(np.interp(xa, xp, np.imag(samples))).astype(np.int16)  # 线性插值虚部

        self._write_samples(s, gps)  # 写入样本

        last = gps['last_gps_solution']  # 获取最后的 GPS 解决方案
        if last == 255 or last == 254:  # 如果没有 GPS 或最近没有解决方案
            self._options.status = 3  # 设置状态

    def _update_wav_header(self):
        with open(self._get_output_filename(), 'r+b') as fp:  # 打开输出文件
            fp.seek(0, os.SEEK_END)  # 移动到文件末尾
            filesize = fp.tell()  # 获取文件大小
            fp.seek(0, os.SEEK_SET)  # 移动到文件开头

            if filesize >= 8:  # 如果文件大小大于等于 8 字节
                _write_wav_header(fp, filesize, self._output_sample_rate, self._num_channels, self._options.is_kiwi_wav)  # 更新 WAV 文件头

    def _write_samples(self, samples, *args):
        """将样本写入磁盘文件"""
        now = time.gmtime()  # 获取当前时间
        sec_of_day = lambda x: 3600 * x.tm_hour + 60 * x.tm_min + x.tm_sec  # 计算一天中的秒数
        dt_reached = self._options.dt != 0 and self._start_ts is not None and sec_of_day(now) // self._options.dt != sec_of_day(self._start_ts) // self._options.dt  # 检查是否达到新的时间间隔
        if self._start_ts is None or (self._options.filename == '' and dt_reached):  # 如果没有开始时间戳或达到新的时间间隔
            self._start_ts = now  # 更新开始时间
            self._start_time = time.time()  # 更新开始时间
            with open(self._get_output_filename(), 'wb') as fp:  # 打开输出文件
                _write_wav_header(fp, 100, self._output_sample_rate, self._num_channels, self._options.is_kiwi_wav)  # 写入静态 WAV 文件头
            if self._options.is_kiwi_tdoa:  # 如果是 TDoA 模式
                print("file=%d %s" % (self._options.idx, self._get_output_filename()))  # 打印文件信息
            else:
                logging.info("Started a new file: %s" % self._get_output_filename())  # 记录日志
        with open(self._get_output_filename(), 'ab') as fp:  # 追加写入输出文件
            if self._options.is_kiwi_wav:  # 如果是 Kiwi WAV 模式
                gps = args[0]  # 获取 GPS 数据
                self._gnss_performance.analyze(self._get_output_filename(), gps)  # 分析 GNSS 性能
                fp.write(struct.pack('<4sIBBII', b'kiwi', 10, gps['last_gps_solution'], 0, gps['gpssec'], gps['gpsnsec']))  # 写入 Kiwi 标头
                sample_size = samples.itemsize * len(samples)  # 计算样本大小
                fp.write(struct.pack('<4sI', b'data', sample_size))  # 写入数据标头
            samples.tofile(fp)  # 写入样本数据
        self._update_wav_header()  # 更新 WAV 文件头

    def _on_gnss_position(self, pos):
        """
        处理 GNSS 位置信息的方法。
        
        :param pos: 包含 GNSS 位置信息的元组 (经度, 纬度)
        """
        pos_record = False  # 初始化位置记录标志为 False

        if self._options.dir is not None:  # 如果指定了目录选项
            pos_dir = self._options.dir  # 使用指定的目录
            pos_record = True  # 标记需要记录位置
        else:
            if os.path.isdir('gnss_pos'):  # 如果默认目录 'gnss_pos' 存在
                pos_dir = 'gnss_pos'  # 使用默认目录
                pos_record = True  # 标记需要记录位置

        if pos_record:  # 如果需要记录位置
            station = 'kiwi_noname' if self._options.station is None else self._options.station  # 获取站点名称，如果没有指定则使用默认值 'kiwi_noname'
            pos_filename = pos_dir + '/' + station + '.txt'  # 构建位置文件的路径

            with open(pos_filename, 'w') as f:  # 打开位置文件进行写入
                station = station.replace('-', '_')  # 将站点名称中的 '-' 替换为 '_'，以符合 Octave 变量命名规则
                f.write("d.%s = struct('coord', [%f,%f], 'host', '%s', 'port', %d);\n"  # 写入位置信息
                        % (station,  # 站点名称
                        pos[0], pos[1],  # 经度和纬度
                        self._options.server_host,  # 服务器主机
                        self._options.server_port))  # 服务器端口
## -------------------------------------------------------------------------------------------------
class KiwiWaterfallRecorder(KiwiSDRStream):
    """
    KiwiWaterfallRecorder 类继承自 KiwiSDRStream，用于记录 Kiwi SDR 的瀑布图数据。
    """

    def __init__(self, options):
        """
        初始化方法，设置各种参数和变量。
        
        :param options: 包含配置选项的对象
        """
        super(KiwiWaterfallRecorder, self).__init__()
        self._options = options
        self._type = 'W/F'  # 瀑布图类型
        freq = options.frequency  # 设置频率
        # logging.info "%s:%s freq=%d" % (options.server_host, options.server_port, freq)
        self._freq = freq
        self._freq_offset = options.freq_offset  # 频率偏移
        self._start_ts = time.gmtime()  # 记录开始时间
        self._start_time = None
        self._last_gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], [0, 0, 0, 0]))  # GPS 相关信息
        self.wf_pass = 0  # 瀑布图处理次数
        self._rows = []  # 存储瀑布图行数据
        self._cmap_r = array.array('B')  # 红色通道颜色映射
        self._cmap_g = array.array('B')  # 绿色通道颜色映射
        self._cmap_b = array.array('B')  # 蓝色通道颜色映射

        # Kiwi 颜色映射
        for i in range(256):
            if i < 32:
                r = 0
                g = 0
                b = i * 255 / 31
            elif i < 72:
                r = 0
                g = (i - 32) * 255 / 39
                b = 255
            elif i < 96:
                r = 0
                g = 255
                b = 255 - (i - 72) * 255 / 23
            elif i < 116:
                r = (i - 96) * 255 / 19
                g = 255
                b = 0
            elif i < 184:
                r = 255
                g = 255 - (i - 116) * 255 / 67
                b = 0
            else:
                r = 255
                g = 0
                b = (i - 184) * 128 / 70

            self._cmap_r.append(clamp(int(round(r)), 0, 255))
            self._cmap_g.append(clamp(int(round(g)), 0, 255))
            self._cmap_b.append(clamp(int(round(b)), 0, 255))

    def _setup_rx_params(self):
        """
        设置接收参数。
        """
        baseband_freq = self._remove_freq_offset(self._freq)  # 去除频率偏移
        self._set_zoom_cf(self._options.zoom, baseband_freq)  # 设置缩放系数
        self._set_maxdb_mindb(-10, -110)  # 设置最大最小 dB 值
        self._set_wf_speed(self._options.speed)  # 设置瀑布图速度
        if self._options.no_api:
            self._setup_no_api()
            return
        # self._set_wf_comp(True)
        self._set_wf_comp(False)  # 设置瀑布图补偿
        self._set_wf_interp(self._options.interp)  # 设置瀑布图插值
        self.set_name(self._options.user)  # 设置用户名

        self._start_time = time.time()
        span = self.zoom_to_span(self._options.zoom)  # 计算跨度
        start = baseband_freq - span / 2  # 计算起始频率
        stop = baseband_freq + span / 2  # 计算结束频率
        if self._options.wf_cal is None:
            self._options.wf_cal = -13  # 兼容旧版本
        logging.info("wf samples: start|center|stop %.1f|%.1f|%.1f kHz, zoom %d, span %d kHz, rbw %.3f kHz, cal %d dB"
                     % (start, baseband_freq, stop, self._options.zoom, span, span / self.WF_BINS, self._options.wf_cal))
        if start < 0 or stop > self.MAX_FREQ:
            s = "Frequency and zoom values result in span outside 0 - %d kHz range" % (self.MAX_FREQ)
            raise Exception(s)
        if self._options.wf_png is True:
            logging.info("--wf_png: mindb %d, maxdb %d, cal %d dB" % (self._options.mindb, self._options.maxdb, self._options.wf_cal))

    def _waterfall_color_index_max_min(self, value):
        """
        将值转换为颜色索引。

        :param value: 输入值
        :return: 颜色索引
        """
        db_value = -(255 - value)  # 55..255 => -200..0 dBm
        db_value = clamp(db_value + self._options.wf_cal, self._options.mindb, self._options.maxdb)
        relative_value = db_value - self._options.mindb
        fullscale = self._options.maxdb - self._options.mindb
        fullscale = fullscale if fullscale != 0 else 1  # 不能为零
        value_percent = relative_value / fullscale
        return clamp(int(round(value_percent * 255)), 0, 255)

    def _process_waterfall_samples(self, seq, samples):
        """
        处理瀑布图样本。

        :param seq: 序列号
        :param samples: 样本数据
        """
        baseband_freq = self._remove_freq_offset(self._freq)  # 去除频率偏移
        nbins = len(samples)  # 样本数量
        bins = nbins - 1
        i = 0
        pwr = []
        pixels = array.array('B')
        do_wf = self._options.wf_png and (not self._options.wf_auto or (self._options.wf_auto and self.wf_pass != 0))

        for s in samples:
            dBm = s - 255  # 转换为 dBm
            if i > 2 and dBm > -190:  # 跳过 DC 偏移和掩蔽区域
                pwr.append({'dBm': dBm, 'i': i})
            i = i + 1

            if do_wf:
                ci = self._waterfall_color_index_max_min(s)  # 获取颜色索引
                pixels.append(self._cmap_r[ci])
                pixels.append(self._cmap_g[ci])
                pixels.append(self._cmap_b[ci])

        pwr.sort(key=by_dBm)  # 按 dBm 排序
        length = len(pwr)
        pmin = pwr[0]['dBm'] + self._options.wf_cal
        bmin = pwr[0]['i']
        pmax = pwr[length - 1]['dBm'] + self._options.wf_cal
        bmax = pwr[length - 1]['i']
        span = self.zoom_to_span(self._options.zoom)
        start = baseband_freq - span / 2

        if (not self._options.wf_png and not self._options.quiet) or (self._options.wf_png and self._options.not_quiet):
            logging.info("wf samples: %d bins, min %d dB @ %.2f kHz, max %d dB @ %.2f kHz"
                         % (nbins, pmin, start + span * bmin / bins, pmax, start + span * bmax / bins))

        if self._options.wf_peaks > 0:
            with open(self._get_output_filename("_peaks.txt"), 'a') as fp:
                for i in range(self._options.wf_peaks):
                    j = length - 1 - i
                    bin_i = pwr[j]['i']
                    bin_f = float(bin_i) / bins
                    fp.write("%d %.2f %d  " % (bin_i, start + span * bin_f, pwr[j]['dBm'] + self._options.wf_cal))
                fp.write("\n")

        if self._options.wf_png and self._options.wf_auto and self.wf_pass == 0:
            noise = pwr[int(0.50 * length)]['dBm']
            signal = pwr[int(0.95 * length)]['dBm']
            # 经验调整
            signal = signal + 30
            if signal < -80:
                signal = -80
            noise -= 10
            self._options.mindb = noise
            self._options.maxdb = signal
            logging.info("--wf_auto: mindb %d, maxdb %d, cal %d dB" % (self._options.mindb, self._options.maxdb, self._options.wf_cal))
        self.wf_pass = self.wf_pass + 1
        if do_wf is True:
            self._rows.append(pixels)

    def _close_func(self):
        """
        关闭功能，处理结束时的操作。
        """
        if self._options.wf_png is True:
            self._flush_rows()
        if self._options.wf_peaks > 0:
            logging.info("--wf-peaks: writing to file %s" % self._get_output_filename("_peaks.txt"))

    def _flush_rows(self):
        """
        将瀑布图行数据写入文件。
        """
        if not self._rows:
            return
        filename = self._get_output_filename(".png")
        logging.info("--wf_png: writing file %s" % filename)
        while True:
            with open(filename, 'wb') as fp:
                try:
                    png.Writer(len(self._rows[0]) // 3, len(self._rows)).write(fp, self._rows)
                    break
                except KeyboardInterrupt:
                    pass

"""class KiwiWaterfallRecorder(KiwiSDRStream):
    def __init__(self, options):
        super(KiwiWaterfallRecorder, self).__init__()  # 调用父类构造函数
        self._options = options  # 初始化选项
        self._type = 'W/F'  # 设置记录类型为瀑布图
        freq = options.frequency  # 获取频率选项
        # logging.info "%s:%s freq=%d" % (options.server_host, options.server_port, freq)  # 日志记录服务器信息和频率
        self._freq = freq  # 设置频率
        self._freq_offset = options.freq_offset  # 设置频率偏移
        self._start_ts = time.gmtime()  # 记录开始时间
        self._start_time = None  # 初始化开始时间
        self._last_gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], [0,0,0,0]))  # 初始化GPS信息
        self.wf_pass = 0  # 初始化瀑布图通过次数
        self._rows = []  # 初始化瀑布图行数据
        self._cmap_r = array.array('B')  # 初始化红色通道颜色映射
        self._cmap_g = array.array('B')  # 初始化绿色通道颜色映射
        self._cmap_b = array.array('B')  # 初始化蓝色通道颜色映射

        # Kiwi颜色映射
        for i in range(256):  # 遍历0到255的值
            if i < 32:
                r = 0  # 红色通道值
                g = 0  # 绿色通道值
                b = i * 255 / 31  # 蓝色通道值
            elif i < 72:
                r = 0
                g = (i - 32) * 255 / 39
                b = 255
            elif i < 96:
                r = 0
                g = 255
                b = 255 - (i - 72) * 255 / 23
            elif i < 116:
                r = (i - 96) * 255 / 19
                g = 255
                b = 0
            elif i < 184:
                r = 255
                g = 255 - (i - 116) * 255 / 67
                b = 0
            else:
                r = 255
                g = 0
                b = (i - 184) * 128 / 70

            self._cmap_r.append(clamp(int(round(r)), 0, 255))  # 将红色通道值添加到颜色映射
            self._cmap_g.append(clamp(int(round(g)), 0, 255))  # 将绿色通道值添加到颜色映射
            self._cmap_b.append(clamp(int(round(b)), 0, 255))  # 将蓝色通道值添加到颜色映射

    def _setup_rx_params(self):
        baseband_freq = self._remove_freq_offset(self._freq)  # 去除频率偏移
        self._set_zoom_cf(self._options.zoom, baseband_freq)  # 设置缩放中心频率
        self._set_maxdb_mindb(-10, -110)  # 设置最大最小dB值
        self._set_wf_speed(self._options.speed)  # 设置瀑布图速度
        if self._options.no_api:
            self._setup_no_api()  # 如果不使用API，设置无API模式
            return
        # self._set_wf_comp(True)  # 设置瀑布图压缩
        self._set_wf_comp(False)  # 设置瀑布图不压缩
        self._set_wf_interp(self._options.interp)  # 设置瀑布图插值
        self.set_name(self._options.user)  # 设置用户名

        self._start_time = time.time()  # 记录开始时间
        span = self.zoom_to_span(self._options.zoom)  # 计算跨度
        start = baseband_freq - span / 2  # 计算起始频率
        stop = baseband_freq + span / 2  # 计算结束频率
        if self._options.wf_cal is None:
            self._options.wf_cal = -13  # 兼容旧版本
        logging.info("wf samples: start|center|stop %.1f|%.1f|%.1f kHz, zoom %d, span %d kHz, rbw %.3f kHz, cal %d dB"
                     % (start, baseband_freq, stop, self._options.zoom, span, span / self.WF_BINS, self._options.wf_cal))
        if start < 0 or stop > self.MAX_FREQ:
            s = "Frequency and zoom values result in span outside 0 - %d kHz range" % (self.MAX_FREQ)
            raise Exception(s)  # 抛出异常
        if self._options.wf_png is True:
            logging.info("--wf_png: mindb %d, maxdb %d, cal %d dB" % (self._options.mindb, self._options.maxdb, self._options.wf_cal))

    def _waterfall_color_index_max_min(self, value):
        db_value = -(255 - value)  # 将值转换为dBm
        db_value = clamp(db_value + self._options.wf_cal, self._options.mindb, self._options.maxdb)  # 校准dB值
        relative_value = db_value - self._options.mindb  # 计算相对值
        fullscale = self._options.maxdb - self._options.mindb  # 计算全范围
        fullscale = fullscale if fullscale != 0 else 1  # 防止除以零
        value_percent = relative_value / fullscale  # 计算百分比
        return clamp(int(round(value_percent * 255)), 0, 255)  # 返回颜色索引

    def _process_waterfall_samples(self, seq, samples):
        baseband_freq = self._remove_freq_offset(self._freq)  # 去除频率偏移
        nbins = len(samples)  # 获取样本数量
        bins = nbins - 1  # 计算最后一个bin
        i = 0  # 初始化索引
        pwr = []  # 初始化功率列表
        pixels = array.array('B')  # 初始化像素数组
        do_wf = self._options.wf_png and (not self._options.wf_auto or (self._options.wf_auto and self.wf_pass != 0))  # 判断是否生成PNG

        for s in samples:
            dBm = s - 255  # 将值转换为dBm
            if i > 2 and dBm > -190:  # 跳过前两个bin和掩码区域
                pwr.append({'dBm': dBm, 'i': i})  # 添加功率信息
            i = i + 1

            if do_wf:
                ci = self._waterfall_color_index_max_min(s)  # 获取颜色索引
                pixels.append(self._cmap_r[ci])  # 添加红色通道值
                pixels.append(self._cmap_g[ci])  # 添加绿色通道值
                pixels.append(self._cmap_b[ci])  # 添加蓝色通道值

        pwr.sort(key=lambda x: x['dBm'])  # 按dBm排序
        length = len(pwr)  # 获取长度
        pmin = pwr[0]['dBm'] + self._options.wf_cal  # 获取最小功率
        bmin = pwr[0]['i']  # 获取最小功率的bin索引
        pmax = pwr[length - 1]['dBm'] + self._options.wf_cal  # 获取最大功率
        bmax = pwr[length - 1]['i']  # 获取最大功率的bin索引
        span = self.zoom_to_span(self._options.zoom)  # 计算跨度
        start = baseband_freq - span / 2  # 计算起始频率

        if (not self._options.wf_png and not self._options.quiet) or (self._options.wf_png and self._options.not_quiet):
            logging.info("wf samples: %d bins, min %d dB @ %.2f kHz, max %d dB @ %.2f kHz"
                         % (nbins, pmin, start + span * bmin / bins, pmax, start + span * bmax / bins))

        if self._options.wf_peaks > 0:
            with open(self._get_output_filename("_peaks.txt"), 'a') as fp:
                for i in range(self._options.wf_peaks):
                    j = length - 1 - i
                    bin_i = pwr[j]['i']
                    bin_f = float(bin_i) / bins
                    fp.write("%d %.2f %d  " % (bin_i, start + span * bin_f, pwr[j]['dBm'] + self._options.wf_cal))
                fp.write("\n")

        if self._options.wf_png and self._options.wf_auto and self.wf_pass == 0:
            noise = pwr[int(0.50 * length)]['dBm']
            signal = pwr[int(0.95 * length)]['dBm']
            # 经验调整
            signal = signal + 30
            if signal < -80:
                signal = -80
            noise -= 10
            self._options.mindb = noise
            self._options.maxdb = signal
            logging.info("--wf_auto: mindb %d, maxdb %d, cal %d dB" % (self._options.mindb, self._options.maxdb, self._options.wf_cal))
        self.wf_pass = self.wf_pass + 1
        if do_wf is True:
            self._rows.append(pixels)  # 添加像素行

    def _close_func(self):
        if self._options.wf_png is True:
            self._flush_rows()  # 写入PNG文件
        if self._options.wf_peaks > 0:
            logging.info("--wf-peaks: writing to file %s" % self._get_output_filename("_peaks.txt"))

    def _flush_rows(self):
        if not self._rows:
            return
        filename = self._get_output_filename(".png")  # 获取输出文件名
        logging.info("--wf_png: writing file %s" % filename)
        while True:
            with open(filename, 'wb') as fp:
                try:
                    png.Writer(len(self._rows[0]) // 3, len(self._rows)).write(fp, self._rows)  # 写入PNG文件
                    break
                except KeyboardInterrupt:
                    pass
"""## -------------------------------------------------------------------------------------------------

class KiwiExtensionRecorder(KiwiSDRStream):
    def __init__(self, options):
        super(KiwiExtensionRecorder, self).__init__()  # 调用父类构造函数
        self._options = options  # 初始化选项
        self._type = 'EXT'  # 设置记录类型为扩展
        self._freq = None  # 初始化频率
        self._start_ts = None  # 初始化开始时间戳
        self._start_time = time.time()  # 记录开始时间

    def _setup_rx_params(self):
        self.set_name(self._options.user)  # 设置用户名
        # rx_chan 已弃用，仅用于向后兼容
        self._send_message('SET ext_switch_to_client=%s first_time=1 rx_chan=0' % self._options.extension)

        if (self._options.extension == 'DRM'):
            if self._kiwi_version is not None and self._kiwi_version >= 1.550:
                self._send_message('SET lock_set')  # 设置锁定
                self._send_message('SET monitor=0')  # 关闭监控
                self._send_message('SET send_iq=0')  # 不发送IQ数据
                self._send_message('SET run=1')  # 开始运行
            else:
                raise Exception("KiwiSDR server v1.550 或更高版本要求用于 DRM")

        if self._options.ext_test:
            self._send_message('SET test=1')  # 设置测试模式

    def _process_ext_msg(self, log, name, value):
        prefix = "EXT %s = " % name if name != None else ""  # 构建消息前缀
        if log is True:
            logging.info("recv %s%s" % (prefix, value))  # 记录接收到的消息
        else:
            sys.stdout.write("%s%s%s\n" % ("\n" if self._need_nl else "", prefix, value))  # 输出消息
            self._need_nl = False if self._need_nl is True else False  # 更新换行标志

    def _process_ext(self, name, value):
        if self._options.extension == 'DRM':
            if self._options.stats and name == "drm_status_cb":
                self._process_ext_msg(False, None, value)  # 处理DRM状态回调消息
            elif name != "drm_status_cb" and name != "drm_bar_pct" and name != "annotate":
                self._process_ext_msg(True, name, value)  # 处理其他消息
            if name == "locked" and value != "1":
                raise Exception("没有 DRM 当 Kiwi 运行其他扩展或活动连接过多时")
        else:
            self._process_ext_msg(True, name, value)  # 处理扩展消息
## -------------------------------------------------------------------------------------------------
class KiwiNetcat(KiwiSDRStream):
    def __init__(self, options, reader):
        super(KiwiNetcat, self).__init__()  # 调用父类构造函数
        self._options = options  # 初始化选项
        self._type = 'W/F' if options.waterfall is True else 'SND'  # 设置记录类型为瀑布图或声音
        self._reader = reader  # 初始化读取器
        freq = options.frequency  # 获取频率
        #logging.info("%s:%s freq=%d" % (options.server_host, options.server_port, freq))  # 记录服务器地址和频率
        self._freq = freq  # 设置频率
        self._freq_offset = options.freq_offset  # 设置频率偏移
        self._start_ts = None  # 初始化开始时间戳
        #self._start_time = None  # 注释掉的开始时间
        self._start_time = time.time()  # 记录开始时间
        self._options.stats = None  # 初始化统计选项
        self._squelch = Squelch(self._options) if options.sq_thresh is not None else None  # 初始化静音门限
        self._last_gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], [0,0,0,0]))  # 初始化GPS信息
        self._fp_stdout = os.fdopen(sys.stdout.fileno(), 'wb')  # 打开标准输出为二进制写模式
        self._first = True  # 初始化第一个样本标志

    def _setup_rx_params(self):
        user = self._options.user  # 获取用户名
        if user == "kiwirecorder.py":
            user = "kiwi_nc.py"  # 替换用户名
        self.set_name(user)  # 设置用户名

        if self._type == 'SND':  # 如果类型为声音
            self.set_freq(self._freq)  # 设置频率

            if self._options.agc_gain != None:  # 固定增益（无AGC）
                self.set_agc(on=False, gain=self._options.agc_gain)  # 设置AGC关闭并设置增益
            elif self._options.agc_yaml_file != None:  # 自定义AGC参数从YAML文件
                self.set_agc(**self._options.agc_yaml)  # 设置AGC参数
            else:  # 默认为AGC开启（默认参数）
                self.set_agc(on=True)  # 设置AGC开启

            if self._options.compression is False:  # 如果压缩关闭
                self._set_snd_comp(False)  # 设置声音压缩关闭

            if self._options.nb is True or self._options.nb_test is True:  # 如果噪声门限开启或测试
                gate = self._options.nb_gate  # 获取噪声门限门限值
                if gate < 100 or gate > 5000:  # 检查门限值范围
                    gate = 100  # 设置默认门限值
                nb_thresh = self._options.nb_thresh  # 获取噪声门限阈值
                if nb_thresh < 0 or nb_thresh > 100:  # 检查阈值范围
                    nb_thresh = 50  # 设置默认阈值
                self.set_noise_blanker(gate, nb_thresh)  # 设置噪声门限

            if self._options.de_emp is True:  # 如果去加重开启
                self.set_de_emp(1)  # 设置去加重

        else:  # 如果类型为瀑布图
            self._set_maxdb_mindb(-10, -110)  # 设置最大最小dB值（不重要）
            self._set_zoom_cf(0, 0)  # 设置缩放中心频率
            self._set_wf_comp(False)  # 设置瀑布图压缩关闭
            self._set_wf_speed(1)  # 设置瀑布图更新速度为1Hz

    def _process_audio_samples_raw(self, seq, samples, rssi):
        if self._options.progress is True:  # 如果显示进度
            sys.stdout.write('\rBlock: %08x, RSSI: %6.1f' % (seq, rssi))  # 输出块序号和RSSI
            sys.stdout.flush()  # 刷新输出
        else:
            if self._squelch:  # 如果静音门限开启
                is_open = self._squelch.process(seq, rssi)  # 处理静音门限
                if not is_open:  # 如果未打开
                    self._start_ts = None  # 重置开始时间戳
                    self._start_time = None  # 重置开始时间
                    return  # 返回
            self._write_samples(samples, {})  # 写入样本

    def _process_iq_samples_raw(self, seq, data):
        if self._options.progress is True:  # 如果显示进度
            sys.stdout.write('\rBlock: %08x, RSSI: %6.1f' % (seq, rssi))  # 输出块序号和RSSI
            sys.stdout.flush()  # 刷新输出
        else:
            count = len(data) // 2  # 计算样本数量
            samples = np.ndarray(count, dtype='>h', buffer=data).astype(np.int16)  # 将数据转换为样本数组
            self._write_samples(samples, {})  # 写入样本

    def _process_waterfall_samples_raw(self, samples, seq):
        if self._options.progress is True:  # 如果显示进度
            nbins = len(samples)  # 获取样本数量
            bins = nbins-1  # 计算bins数量
            max = -1  # 初始化最大值
            min = 256  # 初始化最小值
            bmax = bmin = 0  # 初始化最大最小bin索引
            i = 0  # 初始化索引
            for s in samples:  # 遍历样本
                if s > max:  # 如果当前样本大于最大值
                    max = s  # 更新最大值
                    bmax = i  # 更新最大bin索引
                if s < min:  # 如果当前样本小于最小值
                    min = s  # 更新最小值
                    bmin = i  # 更新最小bin索引
                i += 1  # 增加索引
            span = 30000  # 设置跨度
            sys.stdout.write('\rwf samples %d bins %d..%d dB %.1f..%.1f kHz rbw %d kHz'
                  % (nbins, min-255, max-255, span*bmin/bins, span*bmax/bins, span/bins))  # 输出瀑布图信息
            sys.stdout.flush()  # 刷新输出
        else:
            self._fp_stdout.write(samples)  # 写入样本到标准输出
            self._fp_stdout.flush()  # 刷新输出

    def _write_samples(self, samples, *args):
        if self._options.progress is True:  # 如果显示进度
            return  # 返回
        if self._options.nc_wav and self._first == True:  # 如果输出WAV文件且为第一个样本
            _write_wav_header(self._fp_stdout, 0x7ffffff0, self._sample_rate, 2, False)  # 写入WAV头
            self._first = False  # 设置第一个样本标志为False
        self._fp_stdout.write(samples)  # 写入样本到标准输出
        self._fp_stdout.flush()  # 刷新输出

    def _writer_message(self):
        if self._options.writer_init == False:  # 如果写入器未初始化
            self._options.writer_init = True  # 设置写入器已初始化
            return 'init_msg'  # 返回初始化消息
        msg = sys.stdin.readline()  # 从标准输入读取消息（阻塞）
        return msg  # 返回消息
## -------------------------------------------------------------------------------------------------

def options_cross_product(options):
    """
    根据指定的服务器数量构建选项列表。
    对于每个服务器主机，生成一个新的选项对象，并设置相应的参数。
    """
    def _sel_entry(i, l):
        """
        如果l是列表，则返回索引i对应的元素；否则返回l本身。
        这个函数确保即使索引超出范围，也能安全地获取值。
        """
        return l[min(i, len(l)-1)] if type(l) == list else l

    l = []  # 初始化一个空列表，用于存储生成的选项
    multiple_connections = 0  # 初始化多个连接计数器
    for i, s in enumerate(options.server_host):  # 遍历服务器主机列表
        opt_single = copy(options)  # 复制原始选项对象
        opt_single.server_host = s  # 设置当前服务器主机
        opt_single.status = 0  # 初始化状态为0

        # time() returns seconds, so add pid and host index to make timestamp unique per connection
        opt_single.ws_timestamp = int(time.time() + os.getpid() + i) & 0xffffffff  # 生成唯一的WebSocket时间戳
        for x in ['server_port', 'password', 'tlimit_password', 'frequency', 'agc_gain', 'filename', 'station', 'user']:
            opt_single.__dict__[x] = _sel_entry(i, opt_single.__dict__[x])  # 根据索引选择相应的值
        l.append(opt_single)  # 将生成的选项添加到列表中
        multiple_connections = i  # 更新多个连接计数器
    return multiple_connections, l  # 返回连接计数器和生成的选项列表

def get_comma_separated_args(option, opt, value, parser, fn):
    """
    解析逗号分隔的命令行参数，并将结果设置到解析器的相应属性中。
    使用fn函数对每个值进行处理，然后将其存储为列表。
    """
    values = [fn(v.strip()) for v in value.split(',')]  # 将逗号分隔的字符串转换为列表，并对每个元素应用fn函数
    setattr(parser.values, option.dest, values)  # 将处理后的值设置到解析器的相应属性中
##    setattr(parser.values, option.dest, map(fn, value.split(',')))  # 备注掉的旧实现

def join_threads(snd, wf, ext):
    """
    设置并等待所有线程完成。
    通过设置事件来通知线程停止，并等待所有非当前线程结束。
    """
    [r._event.set() for r in snd]  # 设置snd列表中所有线程的事件
    [r._event.set() for r in wf]  # 设置wf列表中所有线程的事件
    [r._event.set() for r in ext]  # 设置ext列表中所有线程的事件
    [t.join() for t in threading.enumerate() if t is not threading.current_thread()]  # 等待所有非当前线程结束
def main():
    # extend the OptionParser so that we can print multiple paragraphs in
    # the help text
    class MyParser(OptionParser):
        def format_description(self, formatter):
            result = []
            for paragraph in self.description:
                result.append(formatter.format_description(paragraph))
            return "\n".join(result[:-1]) # drop last \n

        def format_epilog(self, formatter):
            result = []
            for paragraph in self.epilog:
                result.append(formatter.format_epilog(paragraph))
            return "".join(result)

    usage = "%prog -s SERVER -p PORT -f FREQ -m MODE [其他选项]"
    description = [
        "kiwirecorder.py 从一个或多个 KiwiSDRs 录制数据到您的磁盘。",
        "它接受许多选项作为输入，最基本的选项如上所示。",
        "要同时录制多个 KiwiSDR，使用相同的语法，但传递一个逗号分隔的值列表（适用时）而不是单个值。",
        "每个值列表应以逗号分隔且不带空格。例如，要录制一个位于 localhost 的 KiwiSDR（端口 80），以及另一个位于 example.com 的 KiwiSDR（端口 8073），运行以下命令：",
        "    kiwirecorder.py -s localhost,example.com -p 80,8073 -f 10000,10000 -m am",
        "在此示例中，两个 KiwiSDR 都将在 10,000 kHz (10 MHz) 上以 AM 模式录制。",
        "任何标明“可以是逗号分隔的列表”的选项也意味着单个值将被复制到多个连接。在上述示例中，可以使用更简单的“-f 10000”。"
    ]
    epilog = [] # 文本会出现在选项列表之后

    parser = MyParser(usage=usage, description=description, epilog=epilog)
    parser.add_option('-s', '--server-host',
                    dest='server_host',
                    type='string', default='localhost',
                    help='服务器主机（可以是逗号分隔的列表）',
                    action='callback',
                    callback_args=(str,),
                    callback=get_comma_separated_args)
    parser.add_option('-p', '--server-port',
                    dest='server_port',
                    type='string', default=8073,
                    help='服务器端口，默认 8073（可以是逗号分隔的列表）',
                    action='callback',
                    callback_args=(int,),
                    callback=get_comma_separated_args)
    parser.add_option('--pw', '--password',
                    dest='password',
                    type='string', default='',
                    help='Kiwi 登录密码（如果需要，可以是逗号分隔的列表）',
                    action='callback',
                    callback_args=(str,),
                    callback=get_comma_separated_args)
    parser.add_option('--tlimit-pw', '--tlimit-password',
                    dest='tlimit_password',
                    type='string', default='',
                    help='连接时间限制豁免密码（如果需要，可以是逗号分隔的列表）',
                    action='callback',
                    callback_args=(str,),
                    callback=get_comma_separated_args)
    parser.add_option('-u', '--user',
                    dest='user',
                    type='string', default='kiwirecorder.py',
                    help='Kiwi 连接用户名（可以是逗号分隔的列表）',
                    action='callback',
                    callback_args=(str,),
                    callback=get_comma_separated_args)
    parser.add_option('--station',
                    dest='station',
                    type='string', default=None,
                    help='附加到文件名的站点 ID（可以是逗号分隔的列表）',
                    action='callback',
                    callback_args=(str,),
                    callback=get_comma_separated_args)
    parser.add_option('--log', '--log-level', '--log_level',
                    dest='log_level',
                    type='choice', default='warn',
                    choices=['debug', 'info', 'warn', 'error', 'critical'],
                    help='日志级别：debug|info|warn(默认)|error|critical')
    parser.add_option('-q', '--quiet',
                    dest='quiet',
                    action='store_true', default=False,
                    help='不打印进度消息')
    parser.add_option('--nq', '--not-quiet',
                    dest='not_quiet',
                    action='store_true', default=False,
                    help='打印进度消息')
    parser.add_option('-d', '--dir',
                    dest='dir',
                    type='string', default=None,
                    help='可选的目标目录用于保存文件')
    parser.add_option('--fn', '--filename',
                    dest='filename',
                    type='string', default='',
                    help='使用固定文件名而不是生成的文件名（可选的站点 ID 将应用，可以是逗号分隔的列表）',
                    action='callback',
                    callback_args=(str,),
                    callback=get_comma_separated_args)
    parser.add_option('--tlimit', '--time-limit',
                    dest='tlimit',
                    type='float', default=None,
                    help='录制时间限制（秒）。当使用 --dt-sec 时忽略此选项。')
    parser.add_option('--dt-sec',
                    dest='dt',
                    type='int', default=0,
                    help='当 mod(一天中的秒数, dt) == 0 时开始新文件')
    parser.add_option('--launch-delay', '--launch_delay',
                    dest='launch_delay',
                    type='int', default=0,
                    help='启动多个连接的延迟（秒）')
    parser.add_option('--connect-timeout', '--connect_timeout',
                    dest='connect_timeout',
                    type='int', default=15,
                    help='重试超时（秒）连接到主机')
    parser.add_option('--connect-retries', '--connect_retries',
                    dest='connect_retries',
                    type='int', default=0,
                    help='连接到主机时的重试次数（默认无限重试）')
    parser.add_option('--busy-timeout', '--busy_timeout',
                    dest='busy_timeout',
                    type='int', default=15,
                    help='主机繁忙时的重试超时（秒）')
    parser.add_option('--busy-retries', '--busy_retries',
                    dest='busy_retries',
                    type='int', default=0,
                    help='主机繁忙时的重试次数（默认无限重试）')
    parser.add_option('-k', '--socket-timeout', '--socket_timeout',
                    dest='socket_timeout',
                    type='int', default=10,
                    help='数据传输期间的套接字超时（秒）')
    parser.add_option('--OV',
                    dest='ADC_OV',
                    action='store_true', default=False,
                    help='当 Kiwi ADC 超载时打印 "ADC OV" 消息')
    parser.add_option('--ts', '--tstamp', '--timestamp',
                    dest='tstamp',
                    action='store_true', default=False,
                    help='添加时间戳到输出。目前仅适用于 S-表模式。')
    parser.add_option('--stats',
                    dest='stats',
                    action='store_true', default=False,
                    help='打印额外的统计信息。适用于例如 S-表和扩展模式。')
    parser.add_option('-v', '-V', '--version',
                    dest='krec_version',
                    action='store_true', default=False,
                    help='打印版本号并退出')

    group = OptionGroup(parser, "音频连接选项", "")
    group.add_option('-f', '--freq',
                    dest='frequency',
                    type='string', default=15000,     # 15000 防止 --wf 模式跨度错误（zoom=0）
                    help='调谐频率，单位 kHz（可以是逗号分隔的列表）。对于边带模式（lsb/lsn/usb/usn/cw/cwn），这是载波频率。参见 --pbc 选项。还设置瀑布图模式中心频率。',
                    action='callback',
                    callback_args=(float,),
                    callback=get_comma_separated_args)
    group.add_option('--pbc', '--freq-pbc',
                    dest='freq_pbc',
                    action='store_true', default=False,
                    help='对于边带模式（lsb/lsn/usb/usn/cw/cwn），将 -f/--freq 频率解释为通带中心频率。')
    group.add_option('-o', '--offset', '--foffset',
                    dest='freq_offset',
                    type='int', default=0,
                    help='调谐频率减去的频率偏移（kHz）（适用于使用偏移的 KiwiSDR）')
    group.add_option('-m', '--mode', '--modulation',
                    dest='modulation',
                    type='string', default='am',
                    help='调制方式；可选值有 am/amn/amw, sam/sau/sal/sas/qam, lsb/lsn, usb/usn, cw/cwn, nbfm/nnfm, iq（默认通带，如果未指定 -L/-H）')
    group.add_option('--ncomp', '--no_compression', '--no_compression',
                    dest='compression',
                    action='store_false', default=True,
                    help='不使用音频压缩（IQ 模式从不使用压缩）')
    group.add_option('-L', '--lp-cutoff',
                    dest='lp_cut',
                    type='float', default=None,
                    help='低通截止频率，单位 Hz')
    group.add_option('-H', '--hp-cutoff',
                    dest='hp_cut',
                    type='float', default=None,
                    help='高通截止频率，单位 Hz')
    group.add_option('-r', '--resample',
                    dest='resample',
                    type='int', default=0,
                    help='重新采样输出文件到新的采样率（Hz）。重新采样比例必须在 [1/256, 256] 范围内')
    group.add_option('-T', '--squelch-threshold',
                    dest='sq_thresh',
                    type='float', default=None,
                    help='静噪阈值，单位 dB。')
    group.add_option('--squelch-tail',
                    dest='squelch_tail',
                    type='float', default=1,
                    help='信号低于阈值后静噪保持打开的时间（秒）。')
    group.add_option('-g', '--agc-gain',
                    dest='agc_gain',
                    type='string', default=None,
                    help='AGC 增益；如果设置，AGC 将关闭（可以是逗号分隔的列表）',
                    action='callback',
                    callback_args=(float,),
                    callback=get_comma_separated_args)
    group.add_option('--agc-yaml',
                    dest='agc_yaml_file',
                    type='string', default=None,
                    help='AGC 选项以 YAML 格式提供的文件')
    group.add_option('--scan-yaml',
                    dest='scan_yaml_file',
                    type='string', default=None,
                    help='扫描选项以 YAML 格式提供的文件')
    group.add_option('--nb',
                    dest='nb',
                    action='store_true', default=False,
                    help='启用标准噪声抑制器，默认参数。')
    group.add_option('--nb-gate',
                    dest='nb_gate',
                    type='int', default=100,
                    help='噪声抑制器门控时间（微秒，范围 100 到 5000，默认 100）')
    group.add_option('--nb-th', '--nb-thresh',
                    dest='nb_thresh',
                    type='int', default=50,
                    help='噪声抑制器阈值（百分比，范围 0 到 100，默认 50）')
    group.add_option('--nb-test',
                    dest='nb_test',
                    action='store_true', default=False,
                    help='启用噪声抑制器测试模式。')
    group.add_option('--de-emp',
                    dest='de_emp',
                    action='store_true', default=False,
                    help='启用去加重。')
    group.add_option('-w', '--kiwi-wav',
                    dest='is_kiwi_wav',
                    action='store_true', default=False,
                    help='在 wav 文件中包含 KIWI 头部，包含 GPS 时间戳（仅适用于 IQ 模式）')
    group.add_option('--kiwi-tdoa',
                    dest='is_kiwi_tdoa',
                    action='store_true', default=False,
                    help='当被 Kiwi TDoA 扩展调用时使用')
    group.add_option('--test-mode',
                    dest='test_mode',
                    action='store_true', default=False,
                    help='将 wav 数据写入 /dev/null（Linux）或 NUL（Windows）')
    group.add_option('--snd', '--sound',
                    dest='sound',
                    action='store_true', default=False,
                    help='在瀑布图或 S-表模式下也处理声音数据（声音连接选项适用）')
    group.add_option('--wb', '--wideband',
                    dest='wideband',
                    action='store_true', default=False,
                    help='打开宽频连接到 Kiwi（如果支持）')
    parser.add_option_group(group)

    group = OptionGroup(parser, "S-表模式选项", "")
    group.add_option('--S-meter', '--s-meter',
                    dest='S_meter',
                    type='int', default=-1,
                    help='报告 S-表（RSSI）值，在 S_METER 次平均后。S_METER=0 不进行平均，报告每次接收到的 RSSI 值。选项 --ts 和 --stats 适用。')
    group.add_option('--sdt-sec',
                    dest='sdt',
                    type='int', default=0,
                    help='S-表测量间隔')
    parser.add_option_group(group)

    group = OptionGroup(parser, "瀑布图连接选项", "")
    group.add_option('--wf', '--waterfall',
                    dest='waterfall',
                    action='store_true', default=False,
                    help='处理瀑布图数据而不是音频。中心频率由音频选项 --f/--freq 设置')
    group.add_option('-z', '--zoom',
                    dest='zoom',
                    type='int', default=0,
                    help='缩放级别 0-14')
    group.add_option('--speed',
                    dest='speed',
                    type='int', default=0,
                    help='瀑布图更新速度：1=1Hz, 2=慢, 3=中, 4=快')
    group.add_option('--interp', '--wf-interp',
                    dest='interp',
                    type='int', default=-1,
                    help='瀑布图显示插值 0-13')
    group.add_option('--wf-png',
                    dest='wf_png',
                    action='store_true', default=False,
                    help='创建瀑布图 .png 文件。--station 和 --filename 选项适用')
    group.add_option('--wf-peaks',
                    dest='wf_peaks',
                    type='int', default=0,
                    help='保存指定数量的瀑布图峰值到文件。--station 和 --filename 选项适用')
    group.add_option('--maxdb',
                    dest='maxdb',
                    type='int', default=-30,
                    help='瀑布图颜色映射最大 dB（-170 到 -10）')
    group.add_option('--mindb',
                    dest='mindb',
                    type='int', default=-155,
                    help='瀑布图颜色映射最小 dB（-190 到 -30）')
    group.add_option('--wf-auto',
                    dest='wf_auto',
                    action='store_true', default=False,
                    help='自动设置 mindb/maxdb')
    group.add_option('--wf-cal',
                    dest='wf_cal',
                    type='int', default=None,
                    help='瀑布图校准修正（覆盖 Kiwi 默认值）')
    parser.add_option_group(group)

    group = OptionGroup(parser, "扩展连接选项", "")
    group.add_option('--ext',
                    dest='extension',
                    type='string', default=None,
                    help='另外打开一个连接到扩展名称')
    group.add_option('--ext-test',
                    dest='ext_test',
                    action='store_true', default=False,
                    help='启动扩展的测试模式（如果适用）')
    parser.add_option_group(group)

    group = OptionGroup(parser, "Netcat 连接选项", "")
    group.add_option('--nc', '--netcat',
                    dest='netcat',
                    action='store_true', default=False,
                    help='打开 netcat 连接')
    group.add_option('--nc-wav', '--nc_wav',
                    dest='nc_wav',
                    action='store_true', default=False,
                    help='将输出格式化为连续的 wav 文件流')
    group.add_option('--fdx',
                    dest='fdx',
                    action='store_true', default=False,
                    help='连接是全双工（双向）')
    parser.add_option('--progress',
                    dest='progress',
                    action='store_true', default=False,
                    help='打印进度消息而不是二进制数据输出')
    parser.add_option_group(group)

    group = OptionGroup(parser, "KiwiSDR 开发选项", "")
    group.add_option('--gc-stats',
                    dest='gc_stats',
                    action='store_true', default=False,
                    help='打印垃圾回收统计信息')
    group.add_option('--nolocal',
                    dest='nolocal',
                    action='store_true', default=False,
                    help='使本地网络连接看起来是非本地的')
    group.add_option('--no-api',
                    dest='no_api',
                    action='store_true', default=False,
                    help='模拟连接到 Kiwi 使用不正确/不完整的 API')
    group.add_option('--devel',
                    dest='devel',
                    type='string', default=None,
                    help='设置开发参数 p0-p7 为浮点值。格式：[0-7]:浮点值, ...')
    parser.add_option_group(group)
    """
    解析命令行参数。
    获取命令行参数解析器的默认值。
    将解析结果与默认值进行合并，确保最终的 options 对象既包含用户提供的命令行参数，也包含未被用户覆盖的默认值。
    """
    # 创建一个 optparse.Values 对象 opts_no_defaults，用于存储解析后的命令行参数。
    opts_no_defaults = optparse.Values()
    # 使用 parser.parse_args 方法解析命令行参数，并将解析结果存储在 opts_no_defaults 对象中。parser 是一个 OptionParser 实例，用于定义和解析命令行选项。解析结果包括两个部分：
    # __：未被选项占用的剩余参数列表。args：解析后的命令行参数，存储在 opts_no_defaults 对象中。
    __, args = parser.parse_args(values=opts_no_defaults) # 解析命令行参数，将结果存储在 `opts_no_defaults` 中
    # 创建一个新的 optparse.Values 对象 options，并将其初始化为 parser 的默认值。parser.get_default_values() 返回一个包含所有默认值的 Values 对象，通过 .__dict__ 获取其字典表示形式。 
    options = optparse.Values(parser.get_default_values().__dict__)
    # 使用 _update_careful 方法将 opts_no_defaults 中的解析结果更新到 options 对象中。_update_careful 方法会谨慎地更新字典，确保不会覆盖已经存在的键值对。
    options._update_careful(opts_no_defaults.__dict__) 
    print("----------------------------------------")
    print("opts_no_defaults",opts_no_defaults)
    print("args",args)
    print("options",options)
    print("----------------------------------------")


    ## clean up OptionParser which has cyclic references
    # 清理 OptionParser 对象，以避免循环引用导致的内存泄漏。
    parser.destroy()

    if options.krec_version:
        print('kiwirecorder v1.4')
        sys.exit()
    # 定义日志格式
    FORMAT = '%(asctime)-15s pid %(process)5d %(message)s'
    # 配置日志记录的基本设置。
    logging.basicConfig(level=logging.getLevelName(options.log_level.upper()), format=FORMAT)
    if options.gc_stats: # --gc-stats 打印垃圾回收统计信息
        # gc 模块: Python 的垃圾收集模块，用于管理内存回收。set_debug 方法: 设置垃圾收集器的调试标志
        # gc.DEBUG_SAVEALL: 保存所有被垃圾收集器识别为不可达的对象。
        # gc.DEBUG_LEAK: 启用内存泄漏检测，记录所有被垃圾收集器识别为不可达的对象。
        # gc.DEBUG_UNCOLLECTABLE: 记录所有无法被垃圾收集器回收的对象。
        gc.set_debug(gc.DEBUG_SAVEALL | gc.DEBUG_LEAK | gc.DEBUG_UNCOLLECTABLE)

    # threading.Event(): 创建一个线程事件对象。Event 是一个简单的同步原语，用于线程间的通信。
    # run_event: 这是一个事件对象的引用，可以在线程之间共享。
    run_event = threading.Event()
    # set() 方法: 将事件的状态设置为“已设置”（即内部标志设为 True）。所有等待该事件的线程将被唤醒并继续执行。
    # 如果没有调用 set()，事件的初始状态是“未设置”（即内部标志为 False）。
    run_event.set()

    # 对命令行选项 --S-meter 和 --sdt-sec 的兼容性进行检查，并根据条件设置 options.quiet 标志。
    if options.S_meter >= 0:
        if options.S_meter > 0 and options.sdt != 0:
            raise Exception('Options --S-meter > 0 and --sdt-sec != 0 are incompatible. Did you mean to use --S-meter=0 ?')
        options.quiet = True

    if options.tlimit is not None and options.dt != 0:
        print('Warning: --tlimit ignored when --dt-sec option used')

    if options.wf_png is True:
        if options.waterfall is False:
            options.waterfall = True
            print('--wf-png note: assuming --wf')
        if options.speed == 0:
            options.speed = 4
            print('--wf-png note: no --speed specified, so using fast (=4)')
        options.quite = True    # specify "--not-quiet" to see all progress messages during --wf-png

    if options.wf_peaks > 0:
        if options.interp == -1:
            options.interp = 10
            print('--wf-peaks 注解：未指定 --wf-interp，因此使用 MAX+CIC (=10)')
            # print('--wf-peaks note: no --wf-interp specified, so using MAX+CIC (=10)')

    ### decode AGC YAML file options
    options.agc_yaml = None
    if options.agc_yaml_file:
        try:
            if not HAS_PyYAML:
                raise Exception('PyYAML not installed: sudo apt install python-yaml / sudo apt install python3-yaml / pip install pyyaml / pip3 install pyyaml')
            with open(options.agc_yaml_file) as yaml_file:
                documents = yaml.full_load(yaml_file)
                logging.debug('AGC file %s: %s' % (options.agc_yaml_file, documents))
                logging.debug('Got AGC parameters from file %s: %s' % (options.agc_yaml_file, documents['AGC']))
                options.agc_yaml = documents['AGC']
        except KeyError:
            logging.fatal('The YAML file does not contain AGC options')
            return
        except Exception as e:
            logging.fatal(e)
            return

    ### decode AGC YAML file options
    options.scan_yaml = None
    if options.scan_yaml_file:
        try:
            if not HAS_PyYAML:
                raise Exception('PyYAML not installed: sudo apt install python-yaml / sudo apt install python3-yaml / pip install pyyaml / pip3 install pyyaml')
            if hasattr(opts_no_defaults, 'frequency'):
                raise Exception('cannot specify frequency (-f, --freq) together with scan YAML (--scan-yaml)')
            with open(options.scan_yaml_file) as yaml_file:
                documents = yaml.full_load(yaml_file)
                logging.debug('Scan file %s: %s' % (options.scan_yaml_file, documents))
                logging.debug('Got Scan parameters from file %s: %s' % (options.scan_yaml_file, documents['Scan']))
                options.scan_yaml = documents['Scan']
                options.scan_state = 'WAIT'
                options.scan_time = time.time()
                options.scan_index = 0
                options.scan_yaml['frequencies'] = [float(f) for f in options.scan_yaml['frequencies']]
                options.frequency = options.scan_yaml['frequencies'][0]
        except KeyError:
            options.scan_yaml = None
            logging.fatal('The YAML file does not contain Scan options')
            return
        except Exception as e:
            options.scan_yaml = None
            logging.fatal(e)
            return

    options.raw = True if options.netcat else False
    options.rigctl_enabled = False
    
    options.maxdb = clamp(options.maxdb, -170, -10)
    options.mindb = clamp(options.mindb, -190, -30)
    if options.maxdb <= options.mindb:
        options.maxdb = options.mindb + 1

    gopt = options
    multiple_connections,options = options_cross_product(options)

    snd_recorders = []
    if not gopt.netcat and (not gopt.waterfall or (gopt.waterfall and gopt.sound)):
        for i,opt in enumerate(options):
            opt.multiple_connections = multiple_connections
            opt.idx = i
            snd_recorders.append(KiwiWorker(args=(KiwiSoundRecorder(opt),opt,run_event)))

    wf_recorders = []
    if not gopt.netcat and gopt.waterfall:
        for i,opt in enumerate(options):
            opt.multiple_connections = multiple_connections
            opt.idx = i
            wf_recorders.append(KiwiWorker(args=(KiwiWaterfallRecorder(opt),opt,run_event)))

    ext_recorders = []
    if not gopt.netcat and (gopt.extension is not None):
        for i,opt in enumerate(options):
            opt.multiple_connections = multiple_connections
            opt.idx = i
            ext_recorders.append(KiwiWorker(args=(KiwiExtensionRecorder(opt),opt,run_event)))

    nc_recorders = []
    if gopt.netcat:
        for i,opt in enumerate(options):
            opt.multiple_connections = multiple_connections
            opt.idx = 0
            nc_recorders.append(KiwiWorker(args=(KiwiNetcat(opt, True),opt,run_event)))
            if gopt.fdx:
                opt.writer_init = False
                opt.idx = 1
                nc_recorders.append(KiwiWorker(args=(KiwiNetcat(opt, False),opt,run_event)))
    try:
        for i,r in enumerate(snd_recorders):
            if opt.launch_delay != 0 and i != 0 and options[i-1].server_host == options[i].server_host:
                time.sleep(opt.launch_delay)
            r.start()
            #logging.info("started sound recorder %d, timestamp=%d" % (i, options[i].ws_timestamp))
            logging.info("started sound recorder %d" % i)

        for i,r in enumerate(wf_recorders):
            if i != 0 and options[i-1].server_host == options[i].server_host:
                time.sleep(opt.launch_delay)
            r.start()
            logging.info("started waterfall recorder %d" % i)

        for i,r in enumerate(ext_recorders):
            if i != 0 and options[i-1].server_host == options[i].server_host:
                time.sleep(opt.launch_delay)
            time.sleep(3)   # let snd/wf get established first
            r.start()
            logging.info("started extension recorder %d" % i)

        for i,r in enumerate(nc_recorders):
            if opt.launch_delay != 0 and i != 0 and options[i-1].server_host == options[i].server_host:
                time.sleep(opt.launch_delay)
            r.start()
            #logging.info("started netcat recorder %d, timestamp=%d" % (i, options[i].ws_timestamp))
            logging.info("started netcat recorder %d" % i)

        while run_event.is_set():
            time.sleep(.1)

    except KeyboardInterrupt:
        run_event.clear()
        join_threads(snd_recorders, wf_recorders, ext_recorders, nc_recorders)
        print("KeyboardInterrupt: threads successfully closed")
    except Exception as e:
        print_exc()
        run_event.clear()
        join_threads(snd_recorders, wf_recorders, ext_recorders, nc_recorders)
        print("Exception: threads successfully closed")

    if gopt.is_kiwi_tdoa:
      for i,opt in enumerate(options):
          # NB for TDoA support: MUST be a print (i.e. not a logging.info)
          print("status=%d,%d" % (i, opt.status))

    if gopt.gc_stats:
        logging.debug('gc %s' % gc.garbage)

if __name__ == '__main__':
    #import faulthandler
    #faulthandler.enable()
    main()
# EOF
