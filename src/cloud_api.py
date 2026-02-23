"""
云端API模块
支持国内主流LLM云服务提供商
"""
import os
import requests
import json
import logging
from typing import Optional, Callable, List, Dict, Union
from enum import Enum

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """支持的云服务商"""
    # 国际
    OPENAI = "openai"
    
    # 阿里云
    DASHSCOPE = "dashscope"           # 灵积平台 - 通义千问
    
    # 百度
    QIANFAN = "qianfan"               # 千帆 - 文心一言
    
    # 腾讯
    HUNYUAN = "hunyuan"               # 混元大模型
    
    # 字节跳动
    DOUBAO = "doubao"                 # 豆包 - 火山引擎
    
    # 智谱AI
    ZHIPU = "zhipu"                   # ChatGLM
    
    # 月之暗面
    MOONSHOT = "moonshot"             # Kimi
    
    # MiniMax
    MINIMAX = "minimax"               # MiniMax
    
    # 零一万物
    YI = "yi"                         # Yi大模型
    
    # 科大讯飞
    SPARK = "spark"                   # 讯飞星火
    
    # DeepSeek
    DEEPSEEK = "deepseek"
    
    # 硅基流动
    SILICONFLOW = "siliconflow"
    
    # 自定义
    CUSTOM = "custom"


class CloudLLMClient:
    """云端LLM客户端 - 支持国内主流供应商"""
    
    # 供应商配置表 - 2025-2026年最新模型
    PROVIDER_CONFIGS = {
        # 阿里云 - 通义千问 (Qwen3.5系列 - 2026年2月最新)
        CloudProvider.DASHSCOPE: {
            "name": "阿里云灵积",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "default_model": "qwen3.5-plus",
            "models": [
                "qwen3.5-plus",                # Qwen3.5 - 2026.02.16发布，原生多模态
                "qwen3.5-max",                 # Qwen3.5 最强版
                "qwen-max",                    # Max旗舰版
                "qwen-max-latest",             # Max最新版
                "qwen-plus",                   # Plus均衡版
                "qwen-plus-latest",            # Plus最新版  
                "qwen-turbo",                  # 极速版 - 高性价比
                "qwen-long",                   # 长文本版 (100万token)
                "qwen-coder",                  # 代码专用
                "qwen-vl-max",                 # 视觉版
            ],
            "api_format": "openai",
            "stream_format": "openai"
        },
        
        # 百度 - 文心一言 (2025-2026年最新)
        CloudProvider.QIANFAN: {
            "name": "百度千帆",
            "base_url": "https://qianfan.baidubce.com/v2",
            "default_model": "ernie-4.5-turbo-128k",
            "models": [
                "ernie-5.0-thinking-preview",  # ERNIE 5.0 - 2025年旗舰
                "ernie-5.0-thinking-latest",   # 5.0最新版
                "ernie-x1.1",                  # X1.1 - 增强版
                "ernie-4.5-turbo-128k",        # 4.5 Turbo - 长文本
                "ernie-4.0-turbo-8k",          # 4.0 Turbo
                "ernie-speed-128k",            # Speed版 - 极速
                "ernie-speed-pro-128k",        # Speed Pro
                "ernie-lite-8k",               # Lite版
                "ernie-tiny-8k",               # Tiny版
                "deepseek-r1"                  # 接入DeepSeek-R1
            ],
            "api_format": "qianfan",
            "stream_format": "qianfan"
        },
        
        # 腾讯 - 混元 (2025-2026年最新)
        CloudProvider.HUNYUAN: {
            "name": "腾讯混元",
            "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
            "default_model": "hunyuan-2.0-instruct",
            "models": [
                "hunyuan-2.0-thinking",        # 2.0思考版 - 2025年旗舰
                "hunyuan-2.0-thinking-20251109", # 2.0特定版
                "hunyuan-2.0-instruct",        # 2.0指令版 - 推荐
                "hunyuan-2.0-instruct-20251111", # 2.0特定版
                "hunyuan-t1-latest",           # T1推理模型
                "hunyuan-a13b",                # A13B MoE模型
                "hunyuan-lite",                # Lite免费版
                "hunyuan-standard",            # 标准版
                "hunyuan-standard-256K",       # 长文本版
                "hunyuan-pro"                  # 专业版
            ],
            "api_format": "openai",
            "stream_format": "openai"
        },
        
        # 字节跳动 - 豆包/火山方舟 (2025-2026年最新)
        CloudProvider.DOUBAO: {
            "name": "豆包/火山引擎",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "default_model": "doubao-pro-32k",
            "models": [
                "doubao-pro-4k",               # Pro 4K - 专业版
                "doubao-pro-32k",              # Pro 32K - 推荐
                "doubao-pro-256k",             # Pro 256K - 超长文本
                "doubao-lite-4k",              # Lite 4K - 轻量
                "doubao-lite-32k",             # Lite 32K
                "doubao-1.8-4k",               # 豆包1.8 4K
                "doubao-1.8-32k",              # 豆包1.8 32K
                "doubao-1.8-256k",             # 豆包1.8 256K
                "doubao-vision",               # 视觉版
                "doubao-embedding"             # 向量化模型
            ],
            "api_format": "openai",
            "stream_format": "openai"
        },
        
        # 智谱AI - ChatGLM (2026年最新)
        CloudProvider.ZHIPU: {
            "name": "智谱AI",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "default_model": "glm-4-air",
            "models": [
                "glm-5",                       # GLM-5 - 开源之王
                "glm-4.7",                      # GLM-4.7旗舰
                "glm-4.7-flash",                # GLM-4.7极速版
                "glm-4-plus",                   # Plus版 - 旗舰
                "glm-4",                        # GLM-4标准版
                "glm-4-air",                    # Air版 - 推荐均衡
                "glm-4-airx",                   # AirX - 极速版
                "glm-4-flash",                  # Flash版 - 免费
                "glm-4v",                       # 视觉版
                "glm-4v-plus",                  # 视觉Plus版
                "glm-4-9b-chat"                 # 9B开源版
            ],
            "api_format": "zhipu",
            "stream_format": "zhipu"
        },
        
        # 月之暗面 - Kimi (2026年最新)
        CloudProvider.MOONSHOT: {
            "name": "月之暗面",
            "base_url": "https://api.moonshot.cn/v1",
            "default_model": "moonshot-v1-32k",
            "models": [
                "kimi-k2.5",                    # K2.5 - 最新版，推理能力强
                "kimi-k2.5-thinking",           # K2.5思考版
                "moonshot-v1-8k",               # 8K上下文
                "moonshot-v1-32k",              # 32K上下文 - 推荐
                "moonshot-v1-128k",             # 128K上下文 - 长文本
                "moonshot-v1-auto"              # 自动选择
            ],
            "api_format": "openai",
            "stream_format": "openai"
        },
        
        # MiniMax (2026年最新)
        CloudProvider.MINIMAX: {
            "name": "MiniMax",
            "base_url": "https://api.minimax.chat/v1",
            "default_model": "MiniMax-M2.5",
            "models": [
                "MiniMax-M2.5",                 # M2.5 - 最新旗舰，SWE-Bench 80.2%
                "MiniMax-M2.5-Lightning",       # M2.5极速版，100 TPS
                "abab6.5s-chat",                # 6.5s - 极速 (200K上下文)
                "abab6.5-chat",                 # 6.5 - 标准版
                "abab6.5t-chat",                # 6.5t - 万亿参数
            ],
            "api_format": "minimax",
            "stream_format": "minimax"
        },
        
        # 零一万物
        CloudProvider.YI: {
            "name": "零一万物",
            "base_url": "https://api.lingyiwanwu.com/v1",
            "default_model": "yi-lightning",
            "models": [
                "yi-lightning",         # 极速
                "yi-large",             # 旗舰
                "yi-medium",            # 标准
                "yi-spark"              # 轻量
            ],
            "api_format": "openai",     # 兼容OpenAI格式
            "stream_format": "openai"
        },
        
        # 科大讯飞 - 星火
        CloudProvider.SPARK: {
            "name": "讯飞星火",
            "base_url": "https://spark-api-open.xf-yun.com/v1",
            "default_model": "lite",
            "models": [
                "lite",                 # 免费版
                "generalv3",            # 标准版
                "generalv3.5",          # 增强版
                "4.0Ultra"              # 旗舰版
            ],
            "api_format": "openai",     # 兼容OpenAI格式
            "stream_format": "openai"
        },
        
        # DeepSeek (2025-2026年最新 - V3.2系列)
        CloudProvider.DEEPSEEK: {
            "name": "DeepSeek",
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat",
            "models": [
                "deepseek-chat",               # V3.2 - 通用对话 (128K上下文)
                "deepseek-reasoner"            # R1 - 推理模型 (128K上下文, 思维链)
            ],
            "api_format": "openai",
            "stream_format": "openai"
        },

        # 硅基流动 (2026年最新 - 100+开源模型)
        CloudProvider.SILICONFLOW: {
            "name": "硅基流动",
            "base_url": "https://api.siliconflow.cn/v1",
            "default_model": "Qwen/Qwen3-235B-A22B",
            "models": [
                # Qwen3系列
                "Qwen/Qwen3-235B-A22B",              # Qwen3 MoE旗舰
                "Qwen/Qwen3-30B-A3B",                # Qwen3 MoE轻量
                "Qwen/Qwen3-14B",                    # Qwen3 14B
                # DeepSeek系列
                "deepseek-ai/DeepSeek-V3.2",         # DeepSeek V3.2
                "deepseek-ai/DeepSeek-R1",           # DeepSeek R1推理
                "deepseek-ai/DeepSeek-V3",           # DeepSeek V3
                # GLM系列
                "THUDM/glm-4-9b-chat",               # GLM-4 9B - 免费
                "zai-org/GLM-4.7",                   # GLM-4.7
                # InternLM系列
                "internlm/internlm2_5-7b-chat",      # InternLM2.5 7B - 免费
                # Mistral系列
                "mistralai/Mistral-7B-Instruct-v0.2", # Mistral 7B - 免费
                # 专业版模型(付费)
                "Pro/deepseek-ai/DeepSeek-V3.2",     # Pro V3.2
                "Pro/zai-org/GLM-4.7"                # Pro GLM-4.7
            ],
            "api_format": "openai",
            "stream_format": "openai"
        },
        
        # OpenAI (2026年最新)
        CloudProvider.OPENAI: {
            "name": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o",
            "models": [
                "gpt-5.2",                      # GPT-5.2 - 最新
                "gpt-4o",                       # GPT-4o
                "gpt-4-turbo",                  # GPT-4 Turbo
                "gpt-4",                        # GPT-4
                "gpt-3.5-turbo"                 # GPT-3.5
            ],
            "api_format": "openai",
            "stream_format": "openai"
        },
        
        # 自定义
        CloudProvider.CUSTOM: {
            "name": "自定义",
            "base_url": "",
            "default_model": "",
            "models": [],
            "api_format": "openai",     # 默认兼容OpenAI
            "stream_format": "openai"
        }
    }
    
    def __init__(self, provider: str, api_key: str, **kwargs):
        """
        初始化云端LLM客户端
        
        Args:
            provider: 服务商名称 (dashscope/qianfan/hunyuan/doubao/zhipu/moonshot/minimax/yi/spark/deepseek/siliconflow/openai/custom)
            api_key: API密钥
            **kwargs: 其他配置参数
                - model: 模型名称 (可选)
                - base_url: 自定义API地址 (仅custom)
                - secret_key: 部分平台需要 (如百度千帆)
                - app_id: 部分平台需要 (如科大讯飞)
        """
        try:
            self.provider = CloudProvider(provider.lower())
        except ValueError:
            logger.error(f"不支持的供应商: {provider}")
            logger.info(f"支持的供应商: {[p.value for p in CloudProvider]}")
            raise
        
        self.api_key = api_key
        self.secret_key = kwargs.get('secret_key', '')
        self.app_id = kwargs.get('app_id', '')
        self.config = kwargs
        
        # 获取供应商配置
        self.provider_config = self.PROVIDER_CONFIGS[self.provider]
        
        # 设置端点
        self._setup_endpoint()
        
        logger.info(f"初始化云端LLM: {self.provider_config['name']} ({self.provider.value})")
        logger.info(f"使用模型: {self.model}")
    
    def _setup_endpoint(self):
        """配置API端点"""
        # 基础URL
        if self.provider == CloudProvider.CUSTOM:
            self.base_url = self.config.get('base_url', '')
        else:
            self.base_url = self.provider_config['base_url']
        
        # 聊天端点
        api_format = self.provider_config['api_format']
        if api_format == 'qianfan':
            self.chat_endpoint = '/chat/completions'
        else:
            self.chat_endpoint = '/chat/completions'
        
        # 模型名称
        self.model = self.config.get('model', self.provider_config['default_model'])
    
    @classmethod
    def list_supported_providers(cls) -> List[Dict]:
        """列出所有支持的供应商"""
        providers = []
        for provider, config in cls.PROVIDER_CONFIGS.items():
            providers.append({
                'provider': provider.value,
                'name': config['name'],
                'default_model': config['default_model'],
                'models': config['models']
            })
        return providers
    
    @classmethod
    def list_models(cls, provider: str) -> List[str]:
        """列出指定供应商的所有模型"""
        try:
            provider_enum = CloudProvider(provider.lower())
            return cls.PROVIDER_CONFIGS[provider_enum]['models']
        except ValueError:
            return []
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             callback: Optional[Callable[[str], None]] = None,
             **kwargs) -> str:
        """
        对话生成
        
        Args:
            messages: 消息列表
            callback: 流式回调函数
            **kwargs: 额外参数 (temperature, max_tokens等)
            
        Returns:
            response: 生成的回复
        """
        try:
            if callback:
                return self._chat_streaming(messages, callback, **kwargs)
            else:
                return self._chat_non_streaming(messages, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求失败: {e}")
            raise Exception(f"API连接失败: {e}")
        except Exception as e:
            logger.error(f"云端API调用失败: {e}")
            raise
    
    def _chat_non_streaming(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """非流式对话"""
        headers = self._get_headers()
        data = self._build_request_data(messages, stream=False, **kwargs)
        
        url = f"{self.base_url}{self.chat_endpoint}"
        
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=120
        )
        
        if response.status_code != 200:
            logger.error(f"API错误: {response.status_code} - {response.text}")
            response.raise_for_status()
        
        result = response.json()
        return self._parse_response(result)
    
    def _chat_streaming(self, 
                       messages: List[Dict[str, str]], 
                       callback: Callable[[str], None],
                       **kwargs) -> str:
        """流式对话"""
        headers = self._get_headers()
        data = self._build_request_data(messages, stream=True, **kwargs)
        
        url = f"{self.base_url}{self.chat_endpoint}"
        
        full_response = ""
        
        response = requests.post(
            url,
            headers=headers,
            json=data,
            stream=True,
            timeout=120
        )
        
        response.raise_for_status()
        
        stream_format = self.provider_config['stream_format']
        
        for line in response.iter_lines():
            if not line:
                continue
                
            line = line.decode('utf-8')
            
            if stream_format == 'openai':
                if line.startswith('data: '):
                    line = line[6:]
                
                if line == '[DONE]':
                    break
                
                try:
                    chunk = json.loads(line)
                    content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if content:
                        full_response += content
                        callback(content)
                except json.JSONDecodeError:
                    continue
                    
            elif stream_format == 'qianfan':
                if line.startswith('data: '):
                    line = line[6:]
                
                try:
                    chunk = json.loads(line)
                    content = chunk.get('result', '')
                    if content:
                        full_response += content
                        callback(content)
                except (json.JSONDecodeError, KeyError):
                    continue
                    
            elif stream_format == 'zhipu':
                if line.startswith('data: '):
                    line = line[6:]
                
                try:
                    chunk = json.loads(line)
                    content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if content:
                        full_response += content
                        callback(content)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return full_response
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "Content-Type": "application/json"
        }
        
        api_format = self.provider_config['api_format']
        
        if api_format == 'openai':
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        elif api_format == 'qianfan':
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        elif api_format == 'zhipu':
            headers["Authorization"] = self.api_key
            
        elif api_format == 'hunyuan':
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        elif api_format == 'minimax':
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["Group-Id"] = self.config.get('group_id', '')
        
        return headers
    
    def _build_request_data(self, 
                           messages: List[Dict[str, str]], 
                           stream: bool = False,
                           **kwargs) -> Dict:
        """构建请求数据"""
        api_format = self.provider_config['api_format']
        
        # 通用参数
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 512)
        
        if api_format == 'openai':
            return {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
        elif api_format == 'qianfan':
            return {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
        elif api_format == 'zhipu':
            return {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
        elif api_format == 'minimax':
            return {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
        elif api_format == 'hunyuan':
            return {
                "Model": self.model,
                "Messages": messages,
                "Stream": stream,
                "Temperature": temperature,
                "MaxTokens": max_tokens
            }
        
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    
    def _parse_response(self, result: Dict) -> str:
        """解析响应"""
        api_format = self.provider_config['api_format']
        
        try:
            if api_format == 'qianfan':
                return result.get("result", "")
            
            elif api_format == 'hunyuan':
                return result.get("Choices", [{}])[0].get("Message", {}).get("Content", "")
            
            elif api_format == 'minimax':
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            else:
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            return ""
    
    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            test_messages = [{"role": "user", "content": "你好"}]
            response = self._chat_non_streaming(test_messages, max_tokens=10)
            return len(response) > 0
        except Exception as e:
            logger.error(f"API连接测试失败: {e}")
            return False
    
    def get_provider_info(self) -> Dict:
        """获取供应商信息"""
        return {
            'provider': self.provider.value,
            'name': self.provider_config['name'],
            'model': self.model,
            'available_models': self.provider_config['models']
        }


class NetworkChecker:
    """网络连接检查器"""
    
    @staticmethod
    def check_internet(timeout: int = 3) -> bool:
        """检查是否有互联网连接"""
        try:
            test_urls = [
                "https://www.baidu.com",
                "https://www.aliyun.com",
                "https://8.8.8.8"
            ]
            
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=timeout)
                    if response.status_code == 200:
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"网络检查失败: {e}")
            return False
    
    @staticmethod
    def check_api_reachable(api_url: str, timeout: int = 5) -> bool:
        """检查API是否可达"""
        try:
            response = requests.get(api_url, timeout=timeout)
            return response.status_code < 500
        except:
            return False


class HybridLLM:
    """混合LLM管理器 - 本地/云端自动切换"""
    
    def __init__(self, 
                 local_llm,
                 cloud_config: Dict,
                 prefer_cloud: bool = True,
                 auto_fallback: bool = True):
        """
        初始化混合LLM
        
        Args:
            local_llm: 本地LLM实例
            cloud_config: 云端API配置
            prefer_cloud: 优先使用云端
            auto_fallback: 自动回退到本地
        """
        self.local_llm = local_llm
        self.cloud_config = cloud_config
        self.prefer_cloud = prefer_cloud
        self.auto_fallback = auto_fallback
        
        self.cloud_client = None
        self.has_network = False
        self.use_cloud = False
        
        self._init_cloud_client()
    
    def _init_cloud_client(self):
        """初始化云端客户端"""
        if not self.cloud_config.get("enabled", False):
            logger.info("云端API未启用")
            return
        
        api_key = self.cloud_config.get("api_key", "")
        if not api_key:
            logger.warning("未设置API key，无法使用云端服务")
            return
        
        try:
            self.cloud_client = CloudLLMClient(
                provider=self.cloud_config.get("provider", "openai"),
                api_key=api_key,
                model=self.cloud_config.get("model"),
                base_url=self.cloud_config.get("base_url"),
                secret_key=self.cloud_config.get("secret_key"),
                app_id=self.cloud_config.get("app_id")
            )
            logger.info("云端LLM客户端初始化成功")
        except Exception as e:
            logger.error(f"云端LLM初始化失败: {e}")
    
    def check_network(self) -> bool:
        """检查网络状态"""
        self.has_network = NetworkChecker.check_internet()
        
        if self.has_network and self.cloud_client:
            api_reachable = self.cloud_client.test_connection()
            
            if api_reachable:
                self.use_cloud = self.prefer_cloud
                logger.info("网络可用，使用云端LLM")
            else:
                self.use_cloud = False
                logger.warning("API不可达，使用本地LLM")
        else:
            self.use_cloud = False
            if not self.has_network:
                logger.info("无网络连接，使用本地LLM")
        
        return self.use_cloud
    
    def chat(self, messages: List[Dict], callback: Callable = None, **kwargs) -> str:
        """对话生成 - 自动选择云端或本地"""
        if self.auto_fallback:
            self.check_network()
        
        if self.use_cloud and self.cloud_client:
            try:
                logger.info("使用云端LLM生成回复...")
                return self.cloud_client.chat(messages, callback, **kwargs)
            except Exception as e:
                logger.error(f"云端API调用失败: {e}")
                if self.auto_fallback and self.local_llm:
                    logger.info("回退到本地LLM...")
                    return self._use_local_llm(messages, callback)
                else:
                    return "抱歉，服务暂时不可用。"
        else:
            return self._use_local_llm(messages, callback)
    
    def _use_local_llm(self, messages: List[Dict], callback: Callable = None) -> str:
        """使用本地LLM"""
        if not self.local_llm:
            return "本地模型未加载"
        
        logger.info("使用本地LLM生成回复...")
        
        if hasattr(self.local_llm, 'chat'):
            return self.local_llm.chat(messages, callback)
        else:
            prompt = self._build_prompt(messages)
            return self.local_llm.generate(prompt)
    
    def _build_prompt(self, messages: List[Dict]) -> str:
        """构建prompt"""
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
        return prompt
    
    def get_current_mode(self) -> str:
        """获取当前模式"""
        if self.use_cloud and self.cloud_client:
            info = self.cloud_client.get_provider_info()
            return f"云端 ({info['name']} - {info['model']})"
        else:
            return "本地 (RK3588)"
    
    def list_available_cloud_models(self) -> List[str]:
        """列出可用的云端模型"""
        if self.cloud_client:
            return self.cloud_client.provider_config['models']
        return []
