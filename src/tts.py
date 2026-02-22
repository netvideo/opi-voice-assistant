"""
TTS模块 - 语音合成
使用 Qwen3-TTS-0.6B 模型
"""
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
import soundfile as sf
import logging
import os

logger = logging.getLogger(__name__)

class TTSModule:
    """语音合成模块"""
    
    def __init__(self, model_path, device="cpu"):
        """
        初始化TTS模块
        
        Args:
            model_path: 模型路径
            device: 运行设备
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.is_initialized = False
        
    def load_model(self):
        """加载模型"""
        try:
            logger.info(f"正在加载TTS模型: {self.model_path}")
            
            # 加载processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.is_initialized = True
            logger.info("✓ TTS模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"TTS模型加载失败: {e}")
            return False
    
    def synthesize(self, text, output_file=None, speaker_id=0, play_immediately=False):
        """
        语音合成
        
        Args:
            text: 要合成的文本
            output_file: 输出文件路径 (可选)
            speaker_id: 说话人ID
            play_immediately: 是否立即播放 (默认False)
            
        Returns:
            audio_data: 音频数据 (numpy array)
            sampling_rate: 采样率
        """
        if not self.is_initialized:
            logger.error("TTS模型未初始化")
            return None, None
        
        try:
            # 预处理文本
            inputs = self.processor(
                text=text,
                return_tensors="pt"
            )
            
            if self.device == "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            
            # 获取音频数据
            audio_data = outputs.cpu().numpy().squeeze()
            sampling_rate = 24000  # Qwen3-TTS使用24kHz
            
            # 保存到文件
            if output_file:
                sf.write(output_file, audio_data, sampling_rate)
                logger.info(f"音频已保存: {output_file}")
            
            # 立即播放
            if play_immediately:
                self.play_audio(audio_data, sampling_rate)
            
            return audio_data, sampling_rate
            
        except Exception as e:
            logger.error(f"语音合成失败: {e}")
            return None, None
    
    def play_audio(self, audio_data, sampling_rate=24000):
        """
        播放音频
        
        Args:
            audio_data: 音频数据
            sampling_rate: 采样率
        """
        try:
            import sounddevice as sd
            sd.play(audio_data, sampling_rate)
            sd.wait()
        except Exception as e:
            logger.error(f"音频播放失败: {e}")
            # 尝试使用pyaudio
            self._play_with_pyaudio(audio_data, sampling_rate)
    
    def _play_with_pyaudio(self, audio_data, sampling_rate):
        """使用pyaudio播放"""
        try:
            import pyaudio
            
            # 转换为16位整数
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sampling_rate,
                output=True
            )
            
            stream.write(audio_int16.tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            logger.error(f"pyaudio播放失败: {e}")


class TTSStreamHandler:
    """流式TTS处理"""
    
    def __init__(self, tts_module, cache_dir="cache"):
        """
        初始化流式处理器
        
        Args:
            tts_module: TTS模块实例
            cache_dir: 缓存目录
        """
        self.tts = tts_module
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def synthesize_and_play(self, text, play_immediately=True):
        """
        合成并播放
        
        Args:
            text: 要合成的文本
            play_immediately: 是否立即播放
            
        Returns:
            audio_file: 音频文件路径
        """
        # 生成缓存文件名
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        audio_file = os.path.join(self.cache_dir, f"tts_{text_hash}.wav")
        
        # 如果缓存存在，直接播放
        if os.path.exists(audio_file):
            if play_immediately:
                audio_data, sr = sf.read(audio_file)
                self.tts.play_audio(audio_data, sr)
            return audio_file
        
        # 合成新音频
        audio_data, sampling_rate = self.tts.synthesize(text, audio_file)
        
        if audio_data is not None and play_immediately:
            self.tts.play_audio(audio_data, sampling_rate)
        
        return audio_file
    
    def clear_cache(self):
        """清理缓存"""
        import glob
        cache_files = glob.glob(os.path.join(self.cache_dir, "tts_*.wav"))
        for f in cache_files:
            try:
                os.remove(f)
            except:
                pass
        logger.info(f"已清理 {len(cache_files)} 个缓存文件")
