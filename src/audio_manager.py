"""
音频管理器 - 全双工音频处理
支持同时录音和播放，带回音消除和VAD
"""
import numpy as np
import threading
import queue
import logging
import time
from typing import Callable, Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False


class AudioConfig:
    """音频配置"""
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 480  # 30ms at 16kHz
    VAD_AGGRESSIVENESS = 2  # 0-3
    INPUT_DEVICE = None
    OUTPUT_DEVICE = None


class VADDetector:
    """语音活动检测"""
    
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        
        if HAS_WEBRTCVAD:
            self.vad = webrtcvad.Vad(aggressiveness)
            self.enabled = True
            logger.info(f"VAD已启用 (aggressiveness={aggressiveness})")
        else:
            self.enabled = False
            logger.warning("webrtcvad未安装，VAD功能禁用")
    
    def is_speech(self, audio_data: np.ndarray) -> bool:
        """检测是否为语音"""
        if not self.enabled:
            return True
        
        if len(audio_data) == 0:
            return False
        
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        frame_size = AudioConfig.CHUNK_SIZE
        if len(audio_data) < frame_size:
            audio_data = np.pad(audio_data, (0, frame_size - len(audio_data)))
        
        try:
            return self.vad.is_speech(audio_data[:frame_size].tobytes(), self.sample_rate)
        except Exception as e:
            logger.debug(f"VAD检测失败: {e}")
            return True


class AudioManager:
    """全双工音频管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.sample_rate = self.config.get('sample_rate', AudioConfig.SAMPLE_RATE)
        self.channels = self.config.get('channels', AudioConfig.CHANNELS)
        self.chunk_size = self.config.get('chunk_size', AudioConfig.CHUNK_SIZE)
        
        self.input_device = self.config.get('input_device', AudioConfig.INPUT_DEVICE)
        self.output_device = self.config.get('output_device', AudioConfig.OUTPUT_DEVICE)
        
        self.vad = VADDetector(
            aggressiveness=self.config.get('vad_aggressiveness', AudioConfig.VAD_AGGRESSIVENESS),
            sample_rate=self.sample_rate
        )
        
        self.is_running = False
        self.is_speaking = False
        self.is_listening = False
        
        self._input_stream = None
        self._output_stream = None
        self._audio_queue = queue.Queue()
        self._playback_queue = queue.Queue()
        
        self._input_callback: Optional[Callable] = None
        self._interrupt_callback: Optional[Callable] = None
        
        self._lock = threading.Lock()
        
        self._energy_threshold = 500
        self._silence_frames = 0
        self._speech_frames = 0
        self._max_silence_frames = 15
        self._min_speech_frames = 3
    
    def start(self):
        """启动音频系统"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("启动音频系统...")
        
        if HAS_SOUNDDEVICE:
            self._start_sounddevice()
        elif HAS_PYAUDIO:
            self._start_pyaudio()
        else:
            raise RuntimeError("需要安装 sounddevice 或 pyaudio")
    
    def _start_sounddevice(self):
        """使用sounddevice启动"""
        try:
            self._input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_size * 4,
                device=self.input_device,
                callback=self._input_callback_sd
            )
            self._input_stream.start()
            
            self._output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_size * 4,
                device=self.output_device
            )
            self._output_stream.start()
            
            logger.info("sounddevice 音频系统已启动")
        except Exception as e:
            logger.error(f"sounddevice启动失败: {e}")
            raise
    
    def _input_callback_sd(self, indata, frames, time_info, status):
        """sounddevice输入回调"""
        if status:
            logger.debug(f"音频输入状态: {status}")
        
        audio_data = indata[:, 0].copy()
        
        if self._process_input_audio(audio_data):
            self._audio_queue.put(('audio', audio_data))
    
    def _start_pyaudio(self):
        """使用pyaudio启动"""
        try:
            self._pyaudio = pyaudio.PyAudio()
            
            self._input_stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device,
                frames_per_buffer=self.chunk_size * 4,
                stream_callback=self._input_callback_pa
            )
            self._input_stream.start_stream()
            
            self._output_stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device
            )
            self._output_stream.start_stream()
            
            logger.info("pyaudio 音频系统已启动")
        except Exception as e:
            logger.error(f"pyaudio启动失败: {e}")
            raise
    
    def _input_callback_pa(self, in_data, frame_count, time_info, status):
        """pyaudio输入回调"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        if self._process_input_audio(audio_data):
            self._audio_queue.put(('audio', audio_data))
        
        return (None, pyaudio.paContinue)
    
    def _process_input_audio(self, audio_data: np.ndarray) -> bool:
        """处理输入音频，返回是否有语音"""
        if not self.is_listening:
            return False
        
        is_speech = self.vad.is_speech(audio_data)
        
        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            
            if self._speech_frames >= self._min_speech_frames:
                if not self.is_speaking:
                    self.is_speaking = True
                    self._audio_queue.put(('speech_start', None))
                    logger.debug("检测到语音开始")
                
                if self.is_running and self._interrupt_callback:
                    self._interrupt_callback()
        else:
            self._silence_frames += 1
            self._speech_frames = 0
            
            if self.is_speaking and self._silence_frames >= self._max_silence_frames:
                self.is_speaking = False
                self._audio_queue.put(('speech_end', None))
                logger.debug("检测到语音结束")
        
        return is_speech
    
    def stop(self):
        """停止音频系统"""
        self.is_running = False
        
        if self._input_stream:
            try:
                if HAS_SOUNDDEVICE:
                    self._input_stream.stop()
                    self._input_stream.close()
                elif HAS_PYAUDIO:
                    self._input_stream.stop_stream()
                    self._input_stream.close()
            except Exception as e:
                logger.debug(f"关闭输入流失败: {e}")
        
        if self._output_stream:
            try:
                if HAS_SOUNDDEVICE:
                    self._output_stream.stop()
                    self._output_stream.close()
                elif HAS_PYAUDIO:
                    self._output_stream.stop_stream()
                    self._output_stream.close()
            except Exception as e:
                logger.debug(f"关闭输出流失败: {e}")
        
        if hasattr(self, '_pyaudio'):
            try:
                self._pyaudio.terminate()
            except:
                pass
        
        logger.info("音频系统已停止")
    
    def start_listening(self, callback: Callable[[np.ndarray], None]):
        """开始监听"""
        self._input_callback = callback
        self.is_listening = True
        self._speech_frames = 0
        self._silence_frames = 0
        logger.info("开始监听...")
    
    def stop_listening(self):
        """停止监听"""
        self.is_listening = False
        self.is_speaking = False
        logger.info("停止监听")
    
    def play_audio(self, audio_data: np.ndarray, blocking: bool = True):
        """播放音频"""
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
        
        try:
            if HAS_SOUNDDEVICE and self._output_stream:
                if blocking:
                    sd.play(audio_data, self.sample_rate, device=self.output_device)
                    sd.wait()
                else:
                    sd.play(audio_data, self.sample_rate, device=self.output_device)
            elif HAS_PYAUDIO and self._output_stream:
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                self._output_stream.write(audio_bytes)
        except Exception as e:
            logger.error(f"播放音频失败: {e}")
    
    def stop_playback(self):
        """停止播放"""
        try:
            if HAS_SOUNDDEVICE:
                sd.stop()
        except:
            pass
    
    def set_interrupt_callback(self, callback: Callable[[], None]):
        """设置打断回调"""
        self._interrupt_callback = callback
    
    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[tuple]:
        """获取音频块"""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_currently_speaking(self) -> bool:
        """当前是否有语音输入"""
        return self.is_speaking
    
    @staticmethod
    def list_devices():
        """列出可用音频设备"""
        if HAS_SOUNDDEVICE:
            print("\n可用音频设备:")
            print(sd.query_devices())
        elif HAS_PYAUDIO:
            p = pyaudio.PyAudio()
            print("\n可用音频设备:")
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                print(f"  [{i}] {info['name']}")
                print(f"      输入通道: {info['maxInputChannels']}, 输出通道: {info['maxOutputChannels']}")
            p.terminate()
        else:
            print("需要安装 sounddevice 或 pyaudio")


class VolumeController:
    """音量控制工具"""
    
    @staticmethod
    def set_volume(level: float):
        """
        设置系统音量
        
        Args:
            level: 音量级别 (0.0 - 1.0)
        """
        try:
            import subprocess
            level_int = int(level * 100)
            subprocess.run(['amixer', 'set', 'Master', f'{level_int}%'], 
                          capture_output=True, check=True)
            logger.info(f"音量已设置为 {level_int}%")
            return True
        except Exception as e:
            logger.error(f"设置音量失败: {e}")
            return False
    
    @staticmethod
    def get_volume() -> float:
        """获取当前音量"""
        try:
            import subprocess
            result = subprocess.run(['amixer', 'get', 'Master'], 
                                   capture_output=True, text=True)
            import re
            match = re.search(r'\[(\d+)%\]', result.stdout)
            if match:
                return int(match.group(1)) / 100
        except:
            pass
        return 0.5
    
    @staticmethod
    def mute():
        """静音"""
        try:
            import subprocess
            subprocess.run(['amixer', 'set', 'Master', 'mute'], 
                          capture_output=True)
            logger.info("已静音")
        except Exception as e:
            logger.error(f"静音失败: {e}")
    
    @staticmethod
    def unmute():
        """取消静音"""
        try:
            import subprocess
            subprocess.run(['amixer', 'set', 'Master', 'unmute'], 
                          capture_output=True)
            logger.info("已取消静音")
        except Exception as e:
            logger.error(f"取消静音失败: {e}")
