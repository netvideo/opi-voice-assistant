# Orange Pi 5 Plus 中文语音助手

基于 RK3588 NPU 的 ASR + LLM + TTS 本地部署方案

## 硬件要求
- **设备**: Orange Pi 5 Plus
- **SoC**: RK3588 (6 TOPS NPU)
- **内存**: 4GB+ (推荐8GB)
- **存储**: 16GB+ eMMC/SD卡
- **音频**: USB麦克风 + 扬声器

## 技术栈

| 组件 | 模型 | 大小 | 框架 | 备注 |
|------|------|------|------|------|
| ASR | qwen3-asr-0.6b | ~1.2GB | transformers | 支持方言 |
| LLM | DeepSeek-R1-Distill-Qwen-1.5B | ~1GB (w4a16) | RKLLM | 本地优先 |
| LLM | 云端API | - | HTTP | 阿里云/DeepSeek等 |
| TTS | qwen3-tts-0.6b | ~1.2GB | transformers | 支持克隆 |

## 项目结构

```
.
├── models/              # 模型文件目录
│   ├── asr/            # ASR模型
│   ├── llm/            # LLM模型 (RKLLM格式)
│   └── tts/            # TTS模型
├── src/                # 源代码
│   ├── asr.py          # 语音识别模块
│   ├── llm.py          # 大语言模型模块
│   ├── tts.py          # 语音合成模块
│   └── main.py         # 主程序入口
├── scripts/            # 脚本工具
│   ├── install.sh          # 环境安装
│   ├── download_models.sh  # 模型下载
│   ├── convert_llm.py      # LLM模型转换
│   └── test_env.py         # 环境检测
├── config/             # 配置文件
│   └── config.yaml     # 主配置 (含云端API设置)
├── docs/               # 文档
│   └── deployment.md   # 部署指南
└── README.md           # 本文件
```

## 快速开始

### 1. 克隆项目
```bash
git clone <your-repo>
cd opi-voice-assistant
```

### 2. 运行安装脚本
```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

### 3. 下载模型
```bash
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

### 4. 启动语音助手
```bash
# 交互模式
python3 src/main.py

# 强制使用云端API (需配置API key)
python3 src/main.py --cloud

# 强制使用本地模型
python3 src/main.py --local
```

## 功能特性

### 🎯 混合推理模式
- **本地优先**: 无网络时使用RK3588本地模型
- **云端增强**: 有网络时自动切换到云端API (通义千问/DeepSeek等)
- **智能回退**: 云端故障时自动回退到本地模型

### 🎙️ 语音识别
- 支持22种中国方言
- 离线识别，保护隐私
- 实时流式识别

### 🧠 大语言模型
- **本地**: DeepSeek-R1-Distill-Qwen-1.5B (1.5B参数，支持NPU加速)
- **云端**: 支持12+国内主流AI平台 (2026年最新模型)
  - 阿里云 (通义千问Qwen3/3.5系列)
  - 百度千帆 (文心一言ERNIE 5.0/X1.1/4.5系列)
  - 腾讯混元 (2.0/T1/A13B系列)
  - 字节豆包/火山引擎 (1.8/Pro/Lite系列)
  - 智谱AI (ChatGLM-4/4.5/4.6/4.7系列)
  - 月之暗面 (Kimi)
  - MiniMax (abab6.5系列)
  - 零一万物 (Yi Lightning/Large系列)
  - 讯飞星火 (4.0Ultra系列)
  - DeepSeek (V3.2/R1系列)
  - 硅基流动 (100+开源模型)
  - OpenAI (GPT系列)
- 支持多轮对话和上下文记忆
- 智能本地/云端切换

## 支持的云端API供应商 (2026年最新)

| 供应商 | Provider名称 | 推荐模型 | 特点 |
|--------|-------------|---------|------|
| **阿里云** | `dashscope` | `qwen-plus`, `qwen-max` | 通义千问3.5系列，百万token上下文 |
| **百度千帆** | `qianfan` | `ernie-5.0-thinking-preview` | ERNIE 5.0原生全模态 |
| **腾讯混元** | `hunyuan` | `hunyuan-2.0-thinking` | 混元2.0 MoE架构 |
| **字节豆包** | `doubao` | `doubao-pro-32k` | 火山方舟，256K上下文 |
| **智谱AI** | `zhipu` | `glm-4-air`, `glm-4-plus` | GLM-4系列 |
| **月之暗面** | `moonshot` | `moonshot-v1-32k` | Kimi，超长上下文 |
| **MiniMax** | `minimax` | `abab6.5s-chat` | 200K上下文，MoE架构 |
| **零一万物** | `yi` | `yi-lightning` | Lightning极速版 |
| **讯飞星火** | `spark` | `4.0Ultra` | 4.0Ultra旗舰 |
| **DeepSeek** | `deepseek` | `deepseek-chat`, `deepseek-reasoner` | V3.2/R1推理模型 |
| **硅基流动** | `siliconflow` | `deepseek-ai/DeepSeek-V3` | 100+开源模型 |
| **OpenAI** | `openai` | `gpt-4o`, `gpt-3.5-turbo` | GPT系列 |

### 配置示例

```yaml
cloud_api:
  enabled: true
  provider: dashscope              # 选择供应商
  api_key: "your-api-key"          # API密钥
  model: "qwen-plus"               # 选择模型 (可选)
  prefer_cloud: true               # 优先使用云端
  auto_fallback: true              # 失败时回退本地
```

### 环境变量设置 (推荐)

```bash
# 阿里云
export DASHSCOPE_API_KEY="your-dashscope-key"

# 百度千帆
export QIANFAN_API_KEY="your-qianfan-key"

# 腾讯混元
export HUNYUAN_API_KEY="your-hunyuan-key"

# 字节豆包
export DOUBAO_API_KEY="your-doubao-key"

# 其他供应商类似...
# 支持的变量名详见 config/config.yaml
```

### 🔊 语音合成
- 支持语音克隆 (3秒音频克隆音色)
- 支持多说话人
- 流式合成，低延迟

## 手动安装步骤

详见 [docs/deployment.md](docs/deployment.md)

## 许可证

MIT License
