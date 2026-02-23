"""
语音助手主程序
Orchestrates ASR -> LLM -> TTS pipeline
支持本地模型和云端API自动切换
"""
import os
import sys
import time
import yaml
import logging
import threading
import numpy as np
from pathlib import Path

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/voice_assistant.log')
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from typing import Optional, List, Dict, Any

from asr import ASRModule, ASRStreamHandler
from llm import RKLLMRuntime, SimpleLLM
from tts import TTSModule, TTSStreamHandler
from cloud_api import HybridLLM, CloudLLMClient, NetworkChecker
from audio_manager import AudioManager, VolumeController
from tools import get_default_tools, execute_tool


class VoiceAssistant:
    """语音助手主类"""
    
    asr: Optional[ASRModule]
    llm: Optional[Any]  # RKLLMRuntime | HybridLLM | SimpleLLM
    tts: Optional[TTSModule]
    local_llm: Optional[RKLLMRuntime]
    
    def __init__(self, config_path="config/config.yaml"):
        """初始化语音助手"""
        self.config = self._load_config(config_path)
        self.running = False
        
        self.asr = None
        self.llm = None
        self.tts = None
        self.local_llm = None
        
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = self.config.get('max_history', 10)
        
    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 从环境变量读取API key (如果配置中为空)
            cloud_config = config.get('cloud_api', {})
            if cloud_config.get('enabled') and not cloud_config.get('api_key'):
                provider = cloud_config.get('provider', '').upper()
                env_var = f"{provider}_API_KEY"
                api_key = os.environ.get(env_var, '')
                if api_key:
                    config['cloud_api']['api_key'] = api_key
                    logger.info(f"从环境变量 {env_var} 读取API key")
            
            return config
            
        except Exception as e:
            logger.warning(f"配置文件加载失败: {e}，使用默认配置")
            return self._default_config()
    
    def _default_config(self):
        """默认配置"""
        return {
            'models': {
                'asr': 'models/asr/Qwen3-ASR-0.6B',
                'llm': 'models/llm/Qwen3-0.6B-rk3588-w8a8.rkllm',
                'tts': 'models/tts/Qwen3-TTS-0.6B'
            },
            'cloud_api': {
                'enabled': False,
                'provider': 'dashscope',
                'api_key': '',
                'prefer_cloud': True,
                'auto_fallback': True
            },
            'device': 'cpu',
            'audio': {
                'sample_rate': 16000,
                'chunk_duration': 2.0,
                'vad_aggressiveness': 2
            },
            'llm': {
                'max_context_len': 2048,
                'temperature': 0.7,
                'max_new_tokens': 512
            },
            'max_history': 10
        }
    
    def initialize(self):
        """初始化所有模块"""
        logger.info("="*50)
        logger.info("初始化语音助手...")
        logger.info("="*50)
        
        # 初始化ASR
        logger.info("[1/3] 初始化ASR模块...")
        asr_path = self.config['models']['asr']
        if os.path.exists(asr_path):
            self.asr = ASRModule(asr_path, self.config.get('device', 'cpu'))
            if not self.asr.load_model():
                logger.error("ASR初始化失败")
                return False
        else:
            logger.error(f"ASR模型不存在: {asr_path}")
            return False
        
        # 初始化LLM (本地 + 云端混合)
        logger.info("[2/3] 初始化LLM模块...")
        if not self._init_llm():
            logger.error("LLM初始化失败")
            return False
        
        # 初始化TTS
        logger.info("[3/3] 初始化TTS模块...")
        tts_path = self.config['models']['tts']
        if os.path.exists(tts_path):
            self.tts = TTSModule(tts_path, self.config.get('device', 'cpu'))
            if not self.tts.load_model():
                logger.error("TTS初始化失败")
                return False
        else:
            logger.error(f"TTS模型不存在: {tts_path}")
            return False
        
        logger.info("✓ 所有模块初始化完成")
        
        # 检查网络状态
        self._check_network_and_update_mode()
        
        return True
    
    def _init_llm(self) -> bool:
        """初始化LLM (支持本地和云端混合)"""
        cloud_config = self.config.get('cloud_api', {})
        use_cloud = cloud_config.get('enabled', False)
        
        llm_path = self.config['models']['llm']
        if os.path.exists(llm_path):
            logger.info("加载本地LLM...")
            local_llm = RKLLMRuntime(
                llm_path, 
                self.config.get('llm', {}).get('max_context_len', 2048)
            )
            if local_llm.load_model():
                self.local_llm = local_llm
                logger.info("✓ 本地LLM加载完成")
            else:
                logger.warning("本地LLM加载失败")
        else:
            logger.warning(f"本地LLM模型不存在: {llm_path}")
        
        if use_cloud and cloud_config.get('api_key'):
            try:
                logger.info("配置云端LLM...")
                self.llm = HybridLLM(
                    local_llm=self.local_llm,
                    cloud_config=cloud_config,
                    prefer_cloud=cloud_config.get('prefer_cloud', True),
                    auto_fallback=cloud_config.get('auto_fallback', True)
                )
                logger.info("✓ 混合LLM配置完成")
            except Exception as e:
                logger.error(f"混合LLM配置失败: {e}")
                self.llm = self.local_llm
        else:
            self.llm = self.local_llm
            if use_cloud and not cloud_config.get('api_key'):
                logger.warning("云端API已启用但未配置API key")
        
        if self.llm is None:
            logger.error("无法初始化任何LLM (本地和云端都不可用)")
            return False
        
        return True
    
    def _check_network_and_update_mode(self):
        """检查网络并更新模式"""
        if isinstance(self.llm, HybridLLM):
            use_cloud = self.llm.check_network()
            mode = self.llm.get_current_mode()
            logger.info(f"当前模式: {mode}")
        else:
            has_network = NetworkChecker.check_internet()
            if has_network:
                logger.info("网络可用，但仅使用本地模型")
    
    def process_once(self, audio_file=None):
        """
        单次处理流程 (用于测试)
        
        Args:
            audio_file: 音频文件路径 (None则使用麦克风)
        """
        if self.asr is None or self.llm is None or self.tts is None:
            logger.error("模块未完全初始化")
            return
        
        assert self.asr is not None
        assert self.llm is not None
        assert self.tts is not None
        
        try:
            if audio_file:
                logger.info(f"处理音频文件: {audio_file}")
                user_text = self.asr.transcribe_file(audio_file)
            else:
                logger.error("麦克风输入尚未实现")
                return
            
            if not user_text:
                logger.warning("未识别到语音")
                return
            
            logger.info(f"用户说: {user_text}")
            
            self.conversation_history.append({"role": "user", "content": user_text})
            
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            messages = self._prepare_messages_with_system()
            
            logger.info("LLM思考中...")
            start_time = time.time()
            
            assistant_text = self._generate_llm_response(messages)
            
            elapsed = time.time() - start_time
            logger.info(f"LLM回复 ({elapsed:.2f}s): {assistant_text[:100]}...")
            
            self.conversation_history.append({"role": "assistant", "content": assistant_text})
            
            logger.info("合成语音...")
            self.tts.synthesize(assistant_text, play_immediately=True)
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
    
    def _prepare_messages_with_system(self) -> list:
        """准备带系统提示词的消息列表"""
        messages = []
        
        # 添加系统提示词
        system_prompt = self.config.get('system_prompt', '').strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加对话历史
        messages.extend(self.conversation_history)
        
        return messages
    
    def _generate_llm_response(self, messages: list) -> str:
        """使用LLM生成回复"""
        llm = self.llm
        if llm is None:
            return "LLM模块未初始化"
        
        llm_config = self.config.get('llm', {})
        cloud_config = self.config.get('cloud_api', {})
        
        kwargs = {
            'temperature': cloud_config.get('temperature', llm_config.get('temperature', 0.7)),
            'max_tokens': cloud_config.get('max_tokens', llm_config.get('max_new_tokens', 512))
        }
        
        if isinstance(llm, HybridLLM):
            return llm.chat(messages, **kwargs)
        elif isinstance(llm, RKLLMRuntime):
            return llm.chat(messages)
        else:
            prompt = self._build_simple_prompt(messages)
            return llm.generate(prompt)
    
    def _build_simple_prompt(self, messages):
        """构建简单prompt"""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"系统: {content}\n"
            elif role == "user":
                prompt += f"用户: {content}\n"
            else:
                prompt += f"助手: {content}\n"
        prompt += "助手: "
        return prompt
    
    def run_interactive(self):
        """交互模式运行"""
        logger.info("="*50)
        logger.info("语音助手已启动 - 交互模式")
        
        # 显示当前模式
        if isinstance(self.llm, HybridLLM):
            mode = self.llm.get_current_mode()
            logger.info(f"当前模式: {mode}")
        else:
            logger.info("当前模式: 本地 (RK3588)")
        
        logger.info("="*50)
        print("\n命令:")
        print("  text <消息>  - 文本输入")
        print("  file <路径>  - 音频文件输入")
        print("  mode         - 查看当前模式")
        print("  cloud        - 强制使用云端")
        print("  local        - 强制使用本地")
        print("  check        - 检查网络状态")
        print("  clear        - 清空对话历史")
        print("  quit/exit    - 退出")
        print()
        
        self.running = True
        
        while self.running:
            try:
                cmd = input("> ").strip()
                
                if not cmd:
                    continue
                
                if cmd.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif cmd.lower() == 'clear':
                    self.conversation_history = []
                    print("对话历史已清空")
                
                elif cmd.lower() == 'mode':
                    if isinstance(self.llm, HybridLLM):
                        mode = self.llm.get_current_mode()
                        print(f"当前模式: {mode}")
                    else:
                        print("当前模式: 本地 (RK3588)")
                
                elif cmd.lower() == 'cloud':
                    if isinstance(self.llm, HybridLLM):
                        self.llm.prefer_cloud = True
                        self.llm.check_network()
                        print("已切换到云端模式")
                    else:
                        print("云端功能未启用，请在配置中设置")
                
                elif cmd.lower() == 'local':
                    if isinstance(self.llm, HybridLLM):
                        self.llm.prefer_cloud = False
                        self.llm.use_cloud = False
                        print("已切换到本地模式")
                    else:
                        print("当前已经是本地模式")
                
                elif cmd.lower() == 'check':
                    has_network = NetworkChecker.check_internet()
                    if has_network:
                        print("✓ 网络连接正常")
                        if isinstance(self.llm, HybridLLM) and self.llm.cloud_client:
                            api_ok = self.llm.cloud_client.test_connection()
                            if api_ok:
                                print("✓ API服务正常")
                            else:
                                print("✗ API服务异常")
                    else:
                        print("✗ 无网络连接")
                
                elif cmd.startswith('text '):
                    text = cmd[5:].strip()
                    self._process_text(text)
                
                elif cmd.startswith('file '):
                    file_path = cmd[5:].strip()
                    if os.path.exists(file_path):
                        self.process_once(file_path)
                    else:
                        print(f"文件不存在: {file_path}")
                
                else:
            self._process_text(cmd)
    
    def run_duplex(self):
        """全双工模式运行 - 支持同时录音和播放"""
        logger.info("="*50)
        logger.info("语音助手已启动 - 全双工模式")
        logger.info("="*50)
        
        audio_config = self.config.get('audio', {})
        audio_manager = AudioManager(audio_config)
        
        try:
            audio_manager.start()
        except Exception as e:
            logger.error(f"音频系统启动失败: {e}")
            return
        
        self.running = True
        audio_buffer = []
        is_recording = False
        
        def on_interrupt():
            """打断回调"""
            if self.tts and audio_manager.is_running:
                logger.info("检测到用户说话，打断播放...")
                audio_manager.stop_playback()
        
        audio_manager.set_interrupt_callback(on_interrupt)
        
        tools = get_default_tools()
        logger.info("工具已加载: " + ", ".join([t['function']['name'] for t in tools]))
        
        print("\n全双工模式 - 说'小助手'唤醒")
        print("命令: quit 退出\n")
        
        wake_word = self.config.get('wake_word', '小助手')
        
        while self.running:
            try:
                chunk = audio_manager.get_audio_chunk(timeout=0.1)
                
                if chunk is None:
                    continue
                
                msg_type, data = chunk
                
                if msg_type == 'speech_start':
                    is_recording = True
                    audio_buffer = []
                    logger.info("开始录音...")
                    
                elif msg_type == 'speech_end' and is_recording:
                    is_recording = False
                    logger.info("录音结束，处理中...")
                    
                    if len(audio_buffer) > 0:
                        audio_data = np.concatenate(audio_buffer)
                        audio_buffer = []
                        
                        self._process_audio_duplex(audio_data, audio_manager, tools)
                
                elif msg_type == 'audio' and is_recording:
                    audio_buffer.append(data)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"处理错误: {e}")
        
        audio_manager.stop()
        logger.info("全双工模式结束")
    
    def _process_audio_duplex(self, audio_data: np.ndarray, audio_manager, tools: list):
        """处理音频 (全双工模式)"""
        if self.asr is None:
            return
        
        try:
            text = self.asr.transcribe(audio_data)
            
            if not text:
                logger.debug("未识别到语音")
                return
            
            logger.info(f"用户说: {text}")
            
            self.conversation_history.append({"role": "user", "content": text})
            
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            messages = self._prepare_messages_with_tools(tools)
            
            logger.info("LLM思考中...")
            start_time = time.time()
            
            response = self._generate_llm_response_with_tools(messages, tools)
            
            elapsed = time.time() - start_time
            logger.info(f"LLM回复 ({elapsed:.2f}s): {response[:100]}...")
            
            self.conversation_history.append({"role": "assistant", "content": response})
            
            if self.tts is not None:
                audio_manager.stop_listening()
                self.tts.synthesize(response)
                audio_data, sr = self.tts.synthesize(response, output_file=None)
                if audio_data is not None:
                    audio_manager.play_audio(audio_data, sr, blocking=False)
                audio_manager.start_listening(lambda x: None)
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
    
    def _prepare_messages_with_tools(self, tools: list) -> list:
        """准备带工具定义的消息"""
        messages = []
        
        system_prompt = self.config.get('system_prompt', '').strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(self.conversation_history)
        
        return messages
    
    def _generate_llm_response_with_tools(self, messages: list, tools: list) -> str:
        """生成LLM响应，支持工具调用"""
        llm = self.llm
        if llm is None:
            return "LLM模块未初始化"
        
        if isinstance(llm, RKLLMRuntime):
            response = llm.chat(messages, tools=tools)
        else:
            response = llm.chat(messages)
        
        tool_call = self._parse_tool_call(response)
        
        if tool_call:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})
            
            logger.info(f"调用工具: {tool_name}({tool_args})")
            
            result = execute_tool(tool_name, tool_args)
            
            if result.get("success"):
                return result.get("message", "操作成功")
            else:
                return f"操作失败: {result.get('error', '未知错误')}"
        
        return response
    
    def _parse_tool_call(self, response: str) -> dict:
        """解析工具调用"""
        import json
        import re
        
        pattern = r'\s*(\{.*?\})\s*'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            try:
                data = json.loads(match.group(1))
                if "name" in data:
                    return data
            except json.JSONDecodeError:
                pass
        
        return None
    
    def shutdown(self):
        """关闭语音助手"""
        logger.info("正在关闭语音助手...")
        self.running = False
        
        if self.local_llm:
            self.local_llm.release()
        
        logger.info("✓ 语音助手已关闭")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Orange Pi 5 Plus 语音助手')
    parser.add_argument('--config', '-c', default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--audio', '-a', help='音频文件路径 (测试模式)')
    parser.add_argument('--cloud', action='store_true', help='强制使用云端API')
    parser.add_argument('--local', action='store_true', help='强制使用本地模型')
    parser.add_argument('--duplex', '-d', action='store_true', help='全双工模式 (实时语音交互)')
    args = parser.parse_args()

    assistant = VoiceAssistant(args.config)
    
    if args.cloud:
        assistant.config['cloud_api']['prefer_cloud'] = True
        logger.info("命令行参数: 强制使用云端API")
    if args.local:
        assistant.config['cloud_api']['prefer_cloud'] = False
        logger.info("命令行参数: 强制使用本地模型")
    
    if not assistant.initialize():
        logger.error("初始化失败，请检查配置和模型文件")
        sys.exit(1)
    
    try:
        if args.audio:
            assistant.process_once(args.audio)
        elif args.duplex:
            assistant.run_duplex()
        else:
            assistant.run_interactive()
    finally:
        assistant.shutdown()


if __name__ == "__main__":
    main()
