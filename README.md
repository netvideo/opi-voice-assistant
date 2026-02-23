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
| ASR | Qwen3-ASR-0.6B | ~1.2GB | transformers | 支持方言 |
| LLM | Qwen3-0.6B | ~700MB | RKLLM | 4GB内存, 支持Function Calling |
| LLM | Qwen3-1.7B | ~1.8GB | RKLLM | 8GB内存, 支持Function Calling [推荐] |
| LLM | 云端API | - | HTTP | Qwen3.5/GLM-5/DeepSeek V3.2等 |
| TTS | Qwen3-TTS-0.6B | ~1.2GB | transformers | 支持克隆 |

### LLM 模型选择

#### 本地部署 (RK3588 NPU)

| 模型 | 内存占用 | 推理速度 | Function Calling | 推荐场景 |
|------|----------|----------|------------------|----------|
| **Qwen3-0.6B** | ~700MB | 15-20 t/s | ✅ | 4GB内存设备 |
| **Qwen3-1.7B** | ~1.8GB | 8-12 t/s | ✅ | 8GB内存设备 [推荐] |

#### 云端API (2026最新模型)

| 模型 | 提供商 | 特点 |
|------|--------|------|
| **Qwen3.5-397B-A17B** | 阿里云 | 原生多模态，性能超越GPT-5.2 |
| **GLM-5** | 智谱AI | 开源之王，编程/推理极强 |
| **DeepSeek V3.2** | DeepSeek | 极致性价比，1M上下文 |
| **Kimi K2.5** | 月之暗面 | 超长上下文，推理能力强 |
| **GPT-5.2** | OpenAI | 生态完善，工具调用稳定 |

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
│   ├── audio_manager.py # 全双工音频管理器
│   ├── tools.py        # 工具调用模块
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
# 交互模式 (文本/音频文件)
python3 src/main.py

# 全双工模式 (实时语音交互，推荐)
python3 src/main.py --duplex
# 或
python3 src/main.py -d

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

### 🎙️ 全双工音频 (新增)
- **同时录音和播放**: 打断传统"说-等-听"模式
- **VAD语音检测**: 自动检测用户说话开始/结束
- **打断机制 (Barge-in)**: 用户说话时自动停止TTS播放
- **回音消除**: WebRTC VAD过滤播放回声
- **实时交互**: 更自然的对话体验

### 🔧 工具调用 (Function Calling)
- **音量调节**: "把音量调大"、"静音"、"音量调到50%"
- 支持扩展更多工具

### 🎙️ 语音识别
- 支持22种中国方言
- 离线识别，保护隐私
- 实时流式识别

### 🧠 大语言模型
- **本地**: Qwen3-0.6B 或 Qwen3-1.7B (支持NPU加速和Function Calling)
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

## 支持的云端API供应商 (2026年2月最新)

| 供应商 | Provider名称 | 推荐模型 | 特点 |
|--------|-------------|---------|------|
| **阿里云** | `dashscope` | `qwen3.5-plus`, `qwen3.5-max` | Qwen3.5原生多模态，性能超GPT-5.2 |
| **智谱AI** | `zhipu` | `glm-5`, `glm-4.7-flash` | GLM-5开源之王，编程/推理极强 |
| **DeepSeek** | `deepseek` | `deepseek-chat`, `deepseek-reasoner` | V3.2极致性价比，1M上下文 |
| **月之暗面** | `moonshot` | `kimi-k2.5`, `kimi-k2.5-thinking` | K2.5推理能力强，全球开源第四 |
| **百度千帆** | `qianfan` | `ernie-5.0-thinking-preview` | ERNIE 5.0原生全模态 |
| **腾讯混元** | `hunyuan` | `hunyuan-2.0-thinking` | 混元2.0 MoE架构 |
| **硅基流动** | `siliconflow` | `Qwen/Qwen3-235B-A22B` | 100+开源模型，免费额度 |
| **字节豆包** | `doubao` | `doubao-pro-32k` | 火山方舟，256K上下文 |
| **MiniMax** | `minimax` | `MiniMax-M2.5` | SWE-Bench 80.2%，Agent能力强 |
| **零一万物** | `yi` | `yi-lightning` | Lightning极速版 |
| **讯飞星火** | `spark` | `4.0Ultra` | 4.0Ultra旗舰 |
| **OpenAI** | `openai` | `gpt-5.2`, `gpt-4o` | GPT系列，生态完善 |

### 配置示例

```yaml
cloud_api:
  enabled: true
  provider: dashscope              # 选择供应商
  api_key: "your-api-key"          # API密钥
  model: "qwen3.5-plus"            # 2026最新模型 (可选)
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

## 音频模式对比

| 模式 | 命令 | 特点 | 适用场景 |
|------|------|------|----------|
| **交互模式** | `python3 src/main.py` | 文本/文件输入，半双工 | 测试、调试 |
| **全双工模式** | `python3 src/main.py -d` | 实时语音，可打断 | 实际使用 [推荐] |

### 全双工模式特性

```
传统模式 (半双工):
用户说 → 等待识别 → LLM处理 → TTS播放 → 结束

全双工模式:
┌─────────────────────────────────────────┐
│  持续监听 ←→ 检测语音 ←→ 打断播放        │
│     ↓           ↓           ↓           │
│  录音缓冲    VAD检测    即时响应          │
└─────────────────────────────────────────┘
```

**支持的语音指令示例**:
- "把音量调大一点" → 自动调高音量
- "静音" → 系统静音
- "音量调到50%" → 设置精确音量

## 手动安装步骤

详见 [docs/deployment.md](docs/deployment.md)

## 许可证

MIT License
