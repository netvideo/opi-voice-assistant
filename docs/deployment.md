# Orange Pi 5 Plus 语音助手部署指南

## 目录
1. [硬件准备](#硬件准备)
2. [系统安装](#系统安装)
3. [环境配置](#环境配置)
4. [模型准备](#模型准备)
5. [运行测试](#运行测试)
6. [性能优化](#性能优化)
7. [故障排除](#故障排除)

---

## 硬件准备

### 必需
- Orange Pi 5 Plus (4GB+内存)
- 电源适配器 (5V/4A)
- USB麦克风
- 扬声器或耳机
- MicroSD卡 (32GB+) 或 eMMC模块

### 可选
- USB-C Hub (扩展接口)
- 散热风扇 (推荐)
- 外壳

---

## 系统安装

### 1. 下载系统镜像
推荐系统：**Ubuntu 22.04 LTS** (带RK3588 NPU驱动)

从Orange Pi官网下载:
```bash
# 官方镜像
http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-5-plus.html
```

### 2. 烧录镜像
使用 balenaEtcher 或 dd 命令烧录到SD卡:
```bash
# Linux/macOS
sudo dd if=ubuntu-22.04.img of=/dev/sdX bs=4M status=progress
```

### 3. 首次启动
1. 插入SD卡
2. 连接HDMI显示器
3. 连接USB键盘鼠标
4. 上电启动
5. 按照提示完成初始化设置

---

## 环境配置

### 1. 更新系统
```bash
sudo apt update
sudo apt upgrade -y
```

### 2. 检查NPU驱动
```bash
# 检查NPU是否可用
cat /sys/kernel/debug/rknpu/version

# 检查NPU负载 (需要root)
sudo cat /sys/kernel/debug/rknpu/load
```

如果显示版本信息，说明NPU驱动已安装。

### 3. 安装项目
```bash
# 克隆项目 (或使用SCP上传)
cd ~
git clone <项目仓库> opi-voice-assistant
cd opi-voice-assistant

# 运行安装脚本
chmod +x scripts/install.sh
./scripts/install.sh
```

---

## 模型准备

### 方案1: 使用预转换模型 (推荐)

运行下载脚本，选择模型：
```bash
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

可选择的LLM模型：

| 模型 | 内存占用 | 速度 | Function Calling | 推荐场景 |
|------|----------|------|------------------|----------|
| Qwen3-0.6B | ~700MB | 15-20 t/s | ✅ | 4GB内存设备 |
| Qwen3-1.7B | ~1.8GB | 8-12 t/s | ✅ | 8GB内存设备 |

### 方案2: 手动下载模型

#### Qwen3-1.7B (推荐)
```bash
# 从HuggingFace下载预转换模型
pip install huggingface_hub

python3 << 'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3",
    filename="Qwen3-1.7B-rk3588-w8a8.rkllm",
    local_dir="./models/llm"
)
EOF
```

#### Qwen3-0.6B
```bash
python3 << 'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="dulimov/Qwen3-0.6B-rk3588-1.2.1-unsloth-16k",
    filename="Qwen3-0.6B-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.5.rkllm",
    local_dir="./models/llm"
)
EOF
```

### 方案3: 手动转换模型 (高级)

#### 在PC端准备 (需要Linux x86_64)

1. **安装RKLLM Toolkit**
```bash
# 创建conda环境
conda create -n rkllm python=3.8
conda activate rkllm

# 下载RKLLM Toolkit
wget https://github.com/airockchip/rknn-llm/releases/download/v1.2.3/rkllm-toolkit-1.2.3-cp38-cp38-linux_x86_64.whl

# 安装
pip install rkllm-toolkit-1.2.3-cp38-cp38-linux_x86_64.whl
```

2. **下载原始模型**
```bash
# 安装git-lfs
git lfs install

# Qwen3-1.7B
git clone https://huggingface.co/Qwen/Qwen3-1.7B

# 或 Qwen3-0.6B
git clone https://huggingface.co/Qwen/Qwen3-0.6B
```

3. **转换脚本** (save as `convert_to_rkllm.py`)
```python
from rkllm.api import RKLLM

# 配置 - Qwen3-1.7B
modelpath = './Qwen3-1.7B'
llm = RKLLM()

# 加载模型
ret = llm.load_huggingface(model=modelpath, device='cuda')
if ret != 0:
    print('加载失败')
    exit()

# 构建RKLLM模型 (w8a8量化)
ret = llm.build(
    do_quantization=True,
    optimization_level=1,
    target_platform='rk3588',
    quantization_type='w8a8'
)
if ret != 0:
    print('构建失败')
    exit()

# 导出
ret = llm.export_rkllm('./Qwen3-1.7B-rk3588-w8a8.rkllm')
if ret != 0:
    print('导出失败')
    exit()

print('转换成功!')
```

4. **运行转换**
```bash
python convert_to_rkllm.py
```

5. **传输到Orange Pi**
```bash
scp Qwen3-1.7B-rk3588-w8a8.rkllm orangepi@<ip>:~/opi-voice-assistant/models/llm/
```

#### 下载ASR和TTS模型
```bash
# 在Orange Pi上运行
cd ~/opi-voice-assistant
./scripts/download_models.sh
```

---

## 运行测试

### 1. 文本交互模式
```bash
# 激活虚拟环境
source venv/bin/activate

# 运行
python3 src/main.py
```

### 2. 测试命令
```
> 你好
> 今天天气怎么样
> clear          # 清空历史
> quit           # 退出
```

### 3. 音频文件测试
```bash
# 准备测试音频 (16kHz, 单声道, WAV格式)
python3 src/main.py --audio test.wav
```

---

## 性能优化

### 1. CPU调频 (提升性能)
```bash
# 设置为性能模式
echo performance | sudo tee /sys/bus/cpu/devices/cpu*/cpufreq/scaling_governor
```

### 2. 内存优化
```bash
# 关闭不必要的服务
sudo systemctl disable snapd
sudo systemctl disable bluetooth

# 增加swap (如果内存不足)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 3. NPU监控
```bash
# 实时查看NPU负载
watch -n 1 sudo cat /sys/kernel/debug/rknpu/load

# 查看CPU频率
watch -n 1 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

---

## 故障排除

### 问题1: RKLLM库找不到
**症状**: `OSError: librkllmrt.so: cannot open shared object file`

**解决**:
```bash
# 检查库文件
ls -la librkllmrt.so

# 复制到系统目录
sudo cp librkllmrt.so /usr/lib/
sudo ldconfig
```

### 问题2: 内存不足
**症状**: `RuntimeError: out of memory`

**解决**:
- 关闭其他程序
- 减少 `max_context_len` 到 1024
- 使用swap空间

### 问题3: 模型加载失败
**症状**: 模型文件不存在或格式错误

**解决**:
```bash
# 检查模型文件
ls -lh models/llm/*.rkllm
ls -lh models/asr/
ls -lh models/tts/

# 重新下载
./scripts/download_models.sh
```

### 问题4: 音频设备错误
**症状**: 无法录音或播放

**解决**:
```bash
# 列出音频设备
arecord -l
aplay -l

# 测试录音
arecord -D plughw:1,0 -d 5 test.wav

# 测试播放
aplay test.wav
```

### 问题5: 推理速度过慢
**症状**: token生成速度 < 5 t/s

**解决**:
1. 确保使用RKLLM而不是纯CPU
2. 检查NPU是否被调用: `sudo cat /sys/kernel/debug/rknpu/load`
3. 降低模型精度 (如果支持)
4. 增加CPU频率

---

## 云端API配置 (可选)

语音助手支持混合模式：本地模型 + 云端API。当网络可用时，可以自动切换到云端API以获得更强的性能。

### 支持的服务商

| 服务商 | provider名称 | 推荐模型 |
|--------|-------------|----------|
| 阿里云灵积 | `dashscope` | qwen-turbo, qwen-plus |
| OpenAI | `openai` | gpt-3.5-turbo |
| DeepSeek | `deepseek` | deepseek-chat |
| 硅基流动 | `siliconflow` | Qwen/Qwen2.5-7B-Instruct |
| 自定义 | `custom` | 自定义 |

### 配置步骤

#### 1. 获取API Key

**阿里云灵积 (推荐)**:
1. 访问 https://dashscope.aliyun.com/
2. 注册/登录账号
3. 进入"API-KEY管理"创建新Key
4. 新用户有免费额度

**DeepSeek**:
1. 访问 https://platform.deepseek.com/
2. 注册账号
3. 在API Keys页面创建新Key

**SiliconFlow**:
1. 访问 https://siliconflow.cn/
2. 注册账号获取API Key

#### 2. 配置config.yaml

编辑 `config/config.yaml`:

```yaml
cloud_api:
  enabled: true                    # 启用云端API
  provider: dashscope              # 服务商
  api_key: "your-api-key-here"     # 你的API key
  model: "qwen-turbo"              # 模型名称 (可选)
  prefer_cloud: true               # 优先使用云端
  auto_fallback: true              # 失败时自动回退到本地
  temperature: 0.7
  max_tokens: 512
```

#### 3. 使用环境变量 (更安全)

将API key保存在环境变量中，避免写入配置文件:

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export DASHSCOPE_API_KEY="your-api-key"

# 立即生效
source ~/.bashrc
```

然后在 `config.yaml` 中不填写 `api_key`:
```yaml
cloud_api:
  enabled: true
  provider: dashscope
  api_key: ""  # 从环境变量读取
```

#### 4. 测试云端API

启动语音助手后，输入命令:

```
> check          # 检查网络和API状态
> mode           # 查看当前模式
> cloud          # 强制切换到云端
> local          # 强制切换到本地
```

### 命令行参数

```bash
# 强制使用云端API
python3 src/main.py --cloud

# 强制使用本地模型
python3 src/main.py --local
```

### 云端 vs 本地对比

| 特性 | 云端API | 本地模型 |
|------|---------|----------|
| **推理速度** | ⚡ 快 (10-50 t/s) | 🐢 较慢 (5-10 t/s) |
| **网络依赖** | 📡 需要 | ❌ 不需要 |
| **隐私** | ☁️ 数据上传 | 🔒 本地处理 |
| **成本** | 💰 按量付费 | 🆓 免费 |
| **可用性** | ⏰ 依赖服务商 | ✅ 随时可用 |

### 故障排除

**API连接失败**:
```bash
# 检查网络
curl https://dashscope.aliyuncs.com

# 检查API key是否有效
curl -X POST https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-turbo", "input": {"messages": [{"role": "user", "content": "你好"}]}}'
```

**自动回退不工作**:
- 确保 `auto_fallback: true` 已设置
- 检查日志中的错误信息
- 本地模型必须正确加载才能回退

---

## 参考资料

- [RKLLM官方文档](https://github.com/airockchip/rknn-llm)
- [Qwen3-ASR文档](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-TTS文档](https://huggingface.co/Qwen/Qwen3-TTS-0.6B)
- [DeepSeek-R1文档](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

---

## 后续优化

1. **唤醒词检测**: 集成Snowboy或Porcupine
2. **流式ASR**: 实现实时语音识别
3. **多轮对话**: 优化上下文管理
4. **语音克隆**: 使用Qwen3-TTS的voice cloning功能
5. **Web界面**: 添加Flask/FastAPI Web控制面板

---

**祝部署顺利!** 🚀
