"""
LLM模块 - 大语言模型推理
使用 DeepSeek-R1-Distill-Qwen-1.5B + RKLLM
基于 Rockchip RK3588 NPU
"""
import ctypes
import os
import logging
import threading
import queue
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


# ============================================================================
# RKLLM C API 定义
# ============================================================================

class RKLLMError(IntEnum):
    """RKLLM错误码"""
    SUCCESS = 0
    ERROR_INVALID_HANDLE = -1
    ERROR_INVALID_PARAM = -2
    ERROR_OUT_OF_MEMORY = -3
    ERROR_DEVICE_ERROR = -4
    ERROR_RUNTIME_ERROR = -5
    ERROR_INVALID_MODEL = -6


# RKLLM句柄类型
RKLLM_Handle_t = ctypes.c_void_p


class RKLLMParam(ctypes.Structure):
    """RKLLM推理参数"""
    _fields_ = [
        ("max_new_tokens", ctypes.c_int),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("stream", ctypes.c_bool),
    ]
    
    def __init__(self):
        super().__init__()
        self.max_new_tokens = 512
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.repeat_penalty = 1.1
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        self.stream = True


class RKLLMResult(ctypes.Structure):
    """RKLLM推理结果"""
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("is_end", ctypes.c_bool),
    ]


# 回调函数类型定义
# 参数: text_ptr (const char*), user_data (void*)
RKLLMCallback_t = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p,  # text
    ctypes.c_void_p   # user_data
)


class RKLLMRuntime:
    """RKLLM运行时封装 - 基于 Rockchip RK3588 NPU"""
    
    def __init__(self, model_path: str, max_context_len: int = 2048):
        """
        初始化RKLLM运行时
        
        Args:
            model_path: RKLLM模型文件路径 (.rkllm)
            max_context_len: 最大上下文长度
        """
        self.model_path = model_path
        self.max_context_len = max_context_len
        self.handle = None
        self.is_initialized = False
        self.lib = None
        
        # 流式输出回调相关
        self._callback_queue = queue.Queue()
        self._current_callback: Optional[Callable[[str], None]] = None
        
        # 加载RKLLM库
        self._load_library()
        
    def _load_library(self):
        """加载RKLLM运行时库"""
        try:
            # 尝试加载系统库
            self.lib = ctypes.CDLL("librkllmrt.so")
            logger.info("✓ RKLLM库加载成功 (系统)")
        except OSError:
            # 尝试加载本地库
            local_lib = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "librkllmrt.so"
            )
            if os.path.exists(local_lib):
                self.lib = ctypes.CDLL(local_lib)
                logger.info("✓ RKLLM库加载成功 (本地)")
            else:
                raise RuntimeError(
                    "找不到RKLLM运行时库 (librkllmrt.so). "
                    "请确保已安装RKLLM Runtime或手动下载librkllmrt.so到项目目录"
                )
        
        # 设置函数签名
        self._setup_function_signatures()
    
    def _setup_function_signatures(self):
        """设置C函数签名"""
        # rkllm_init
        self.lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),  # handle
            ctypes.c_char_p,                  # model_path
            ctypes.c_char_p                   # config (JSON string, optional)
        ]
        self.lib.rkllm_init.restype = ctypes.c_int
        
        # rkllm_run
        self.lib.rkllm_run.argtypes = [
            RKLLM_Handle_t,                   # handle
            ctypes.c_char_p,                  # prompt
            ctypes.POINTER(RKLLMParam),       # params
            RKLLMCallback_t,                  # callback
            ctypes.c_void_p                   # user_data
        ]
        self.lib.rkllm_run.restype = ctypes.c_int
        
        # rkllm_run_sync (同步版本，如果可用)
        if hasattr(self.lib, 'rkllm_run_sync'):
            self.lib.rkllm_run_sync.argtypes = [
                RKLLM_Handle_t,
                ctypes.c_char_p,
                ctypes.POINTER(RKLLMParam),
                ctypes.POINTER(ctypes.POINTER(RKLLMResult))
            ]
            self.lib.rkllm_run_sync.restype = ctypes.c_int
        
        # rkllm_destroy
        self.lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.lib.rkllm_destroy.restype = ctypes.c_int
        
        # rkllm_get_last_error (如果可用)
        if hasattr(self.lib, 'rkllm_get_last_error'):
            self.lib.rkllm_get_last_error.argtypes = []
            self.lib.rkllm_get_last_error.restype = ctypes.c_char_p
    
    def load_model(self) -> bool:
        """
        加载RKLLM模型
        
        Returns:
            bool: 是否加载成功
        """
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在: {self.model_path}")
            return False
        
        try:
            logger.info(f"正在加载RKLLM模型: {self.model_path}")
            
            # 调用rkllm_init
            handle = RKLLM_Handle_t()
            model_path_bytes = self.model_path.encode('utf-8')
            
            # 可选配置 (JSON格式)
            config = f'{{"max_context_len": {self.max_context_len}}}'
            config_bytes = config.encode('utf-8')
            
            ret = self.lib.rkllm_init(
                ctypes.byref(handle),
                model_path_bytes,
                config_bytes
            )
            
            if ret != RKLLMError.SUCCESS:
                error_msg = self._get_error_message(ret)
                logger.error(f"rkllm_init失败: {error_msg} (错误码: {ret})")
                return False
            
            self.handle = handle
            self.is_initialized = True
            logger.info("✓ RKLLM模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"RKLLM模型加载失败: {e}")
            return False
    
    def _get_error_message(self, error_code: int) -> str:
        """获取错误信息"""
        error_messages = {
            RKLLMError.SUCCESS: "成功",
            RKLLMError.ERROR_INVALID_HANDLE: "无效句柄",
            RKLLMError.ERROR_INVALID_PARAM: "无效参数",
            RKLLMError.ERROR_OUT_OF_MEMORY: "内存不足",
            RKLLMError.ERROR_DEVICE_ERROR: "设备错误",
            RKLLMError.ERROR_RUNTIME_ERROR: "运行时错误",
            RKLLMError.ERROR_INVALID_MODEL: "无效模型",
        }
        
        if hasattr(self.lib, 'rkllm_get_last_error'):
            try:
                last_error = self.lib.rkllm_get_last_error()
                if last_error:
                    return last_error.decode('utf-8')
            except (OSError, AttributeError, UnicodeDecodeError):
                pass
        
        return error_messages.get(error_code, f"未知错误 (代码: {error_code})")
    
    def _create_callback(self) -> RKLLMCallback_t:
        """创建C回调函数"""
        def callback_wrapper(text_ptr: ctypes.c_char_p, user_data: ctypes.c_void_p):
            if text_ptr:
                try:
                    text = text_ptr.decode('utf-8')
                    # 放入队列
                    self._callback_queue.put(('token', text))
                except:
                    pass
        
        return RKLLMCallback_t(callback_wrapper)
    
    def generate(self, 
                 prompt: str, 
                 callback: Optional[Callable[[str], None]] = None,
                 **kwargs) -> str:
        """
        生成回复
        
        Args:
            prompt: 输入提示
            callback: 流式回调函数，每生成一个token调用一次
            **kwargs: 额外参数 (max_new_tokens, temperature, top_p等)
            
        Returns:
            response: 生成的完整回复
        """
        if not self.is_initialized or not self.handle:
            logger.error("RKLLM模型未初始化")
            return ""
        
        try:
            # 设置当前回调
            self._current_callback = callback
            
            # 准备参数
            params = RKLLMParam()
            params.max_new_tokens = kwargs.get('max_new_tokens', 512)
            params.temperature = kwargs.get('temperature', 0.7)
            params.top_p = kwargs.get('top_p', 0.9)
            params.repeat_penalty = kwargs.get('repeat_penalty', 1.1)
            params.stream = callback is not None
            
            # 清空队列
            while not self._callback_queue.empty():
                self._callback_queue.get()
            
            # 创建回调函数
            c_callback = self._create_callback() if callback else None
            
            # 启动推理线程
            result = []
            def inference_thread():
                try:
                    ret = self.lib.rkllm_run(
                        self.handle,
                        prompt.encode('utf-8'),
                        ctypes.byref(params),
                        c_callback,
                        None
                    )
                    if ret != RKLLMError.SUCCESS:
                        error_msg = self._get_error_message(ret)
                        logger.error(f"rkllm_run失败: {error_msg}")
                        self._callback_queue.put(('error', error_msg))
                    else:
                        self._callback_queue.put(('end', ''))
                except Exception as e:
                    logger.error(f"推理线程异常: {e}")
                    self._callback_queue.put(('error', str(e)))
            
            thread = threading.Thread(target=inference_thread)
            thread.start()
            
            # 收集结果
            full_response = []
            while True:
                try:
                    item = self._callback_queue.get(timeout=0.1)
                    msg_type, msg_data = item
                    
                    if msg_type == 'token':
                        full_response.append(msg_data)
                        if callback:
                            callback(msg_data)
                    elif msg_type == 'end':
                        break
                    elif msg_type == 'error':
                        logger.error(f"推理错误: {msg_data}")
                        break
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    continue
            
            thread.join(timeout=5.0)
            
            return ''.join(full_response)
            
        except Exception as e:
            logger.error(f"RKLLM推理失败: {e}")
            return "抱歉，推理过程中出现错误。"
        finally:
            self._current_callback = None
    
    def chat(self, 
             messages: list, 
             callback: Optional[Callable[[str], None]] = None,
             **kwargs) -> str:
        """
        对话模式
        
        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}, ...]
            callback: 流式回调函数
            **kwargs: 额外参数
            
        Returns:
            response: 生成的回复
        """
        # 构建对话prompt
        prompt = self._build_chat_prompt(messages)
        return self.generate(prompt, callback, **kwargs)
    
    def _build_chat_prompt(self, messages: list) -> str:
        """
        构建对话prompt
        
        支持 DeepSeek-R1-Distill-Qwen 格式
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(prompt_parts)
    
    def release(self):
        """释放RKLLM资源"""
        if self.handle and self.lib:
            try:
                ret = self.lib.rkllm_destroy(self.handle)
                if ret != RKLLMError.SUCCESS:
                    logger.warning(f"rkllm_destroy返回错误: {ret}")
                else:
                    logger.info("✓ RKLLM资源已释放")
            except Exception as e:
                logger.error(f"释放RKLLM资源失败: {e}")
            finally:
                self.handle = None
                self.is_initialized = False


class SimpleLLM:
    """
    简化版LLM (使用transformers，用于测试和备选)
    当RKLLM不可用时，可以使用这个作为备选方案
    """
    
    def __init__(self, model_path: str, device="cpu"):
        """
        初始化简化版LLM
        
        Args:
            model_path: 模型路径 (HuggingFace格式)
            device: 运行设备 (cpu/cuda)
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self) -> bool:
        """
        加载Transformers模型
        
        Returns:
            bool: 是否加载成功
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"加载Transformers模型: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 设置padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("✓ Transformers模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"Transformers模型加载失败: {e}")
            return False
    
    def generate(self, 
                 prompt: str, 
                 max_new_tokens: int = 512,
                 callback: Optional[Callable[[str], None]] = None,
                 **kwargs) -> str:
        """
        生成回复
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            callback: 流式回调函数 (注意: transformers不支持真正的流式)
            **kwargs: 其他参数
            
        Returns:
            response: 生成的回复
        """
        import torch
        
        if not self.model or not self.tokenizer:
            logger.error("模型未加载")
            return ""
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=kwargs.get('temperature', 0.7),
                    top_p=kwargs.get('top_p', 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码生成的部分
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            
            response = self.tokenizer.decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            # 模拟流式输出
            if callback:
                for char in response:
                    callback(char)
            
            return response
            
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return ""
    
    def chat(self, 
             messages: list, 
             callback: Optional[Callable[[str], None]] = None,
             **kwargs) -> str:
        """对话模式"""
        # 构建简单prompt
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            else:
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        
        return self.generate(prompt, callback=callback, **kwargs)
    
    def release(self):
        """释放资源"""
        if self.model:
            import torch
            del self.model
            self.model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("Transformers模型资源已释放")


# ============================================================================
# 工厂函数
# ============================================================================

def create_llm(model_path: str, 
               use_rkllm: bool = True,
               device: str = "cpu",
               **kwargs) -> Optional[RKLLMRuntime]:
    """
    创建LLM实例
    
    Args:
        model_path: 模型路径
        use_rkllm: 是否使用RKLLM (如果失败会自动回退到Transformers)
        device: 设备 (仅用于Transformers)
        **kwargs: 其他参数
        
    Returns:
        llm: LLM实例或None
    """
    if use_rkllm:
        try:
            llm = RKLLMRuntime(model_path, **kwargs)
            if llm.load_model():
                return llm
            else:
                logger.warning("RKLLM加载失败，尝试使用Transformers")
        except Exception as e:
            logger.error(f"RKLLM初始化失败: {e}")
            logger.warning("尝试使用Transformers作为备选")
    
    # 回退到Transformers
    try:
        llm = SimpleLLM(model_path, device)
        if llm.load_model():
            return llm
    except Exception as e:
        logger.error(f"Transformers加载失败: {e}")
    
    return None
