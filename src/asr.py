"""
ASR模块 - 语音识别
使用 Qwen3-ASR-0.6B 模型
"""
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import threading
import queue
import logging

logger = logging.getLogger(__name__)

class ASRModule:
    """语音识别模块"""
    
    def __init__(self, model_path, device="cpu"):
        """
        初始化ASR模块
        
        Args:
            model_path: 模型路径
            device: 运行设备 (cpu/cuda)
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.is_initialized = False
        
    def load_model(self):
        """加载模型"""
        try:
            logger.info(f"正在加载ASR模型: {self.model_path}")
            
            # 加载processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            # 加载模型 (使用4bit量化减少内存占用)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            self.is_initialized = True
            logger.info("✓ ASR模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"ASR模型加载失败: {e}")
            return False
    
    def transcribe(self, audio_data, sampling_rate=16000):
        """
        语音识别
        
        Args:
            audio_data: 音频数据 (numpy array)
            sampling_rate: 采样率
            
        Returns:
            text: 识别文本
        """
        if not self.is_initialized:
            logger.error("ASR模型未初始化")
            return ""
        
        try:
            # 预处理音频
            inputs = self.processor(
                audio_data,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )
            
            if self.device == "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                predicted_ids = self.model.generate(**inputs)
            
            # 解码
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return ""
    
    def transcribe_file(self, audio_file):
        """
        识别音频文件
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            text: 识别文本
        """
        import soundfile as sf
        
        try:
            audio_data, sampling_rate = sf.read(audio_file)
            
            # 转换为单声道
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # 重采样到16kHz
            if sampling_rate != 16000:
                import librosa
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sampling_rate, 
                    target_sr=16000
                )
                sampling_rate = 16000
            
            return self.transcribe(audio_data, sampling_rate)
            
        except Exception as e:
            logger.error(f"音频文件读取失败: {e}")
            return ""


class ASRStreamHandler:
    """流式ASR处理"""
    
    def __init__(self, asr_module, chunk_duration=2.0):
        """
        初始化流式处理器
        
        Args:
            asr_module: ASR模块实例
            chunk_duration: 每次处理的音频长度（秒）
        """
        self.asr = asr_module
        self.chunk_duration = chunk_duration
        self.audio_buffer = []
        self.is_listening = False
        
    def start_listening(self):
        """开始监听"""
        self.is_listening = True
        self.audio_buffer = []
        
    def stop_listening(self):
        """停止监听并返回识别结果"""
        self.is_listening = False
        
        if len(self.audio_buffer) > 0:
            audio_data = np.concatenate(self.audio_buffer)
            result = self.asr.transcribe(audio_data)
            self.audio_buffer = []
            return result
        
        return ""
    
    def feed_audio(self, audio_chunk):
        """
        喂入音频块
        
        Args:
            audio_chunk: 音频数据块
        """
        if not self.is_listening:
            return
            
        self.audio_buffer.append(audio_chunk)
