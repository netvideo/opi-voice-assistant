# RKLLM 使用指南

本文档介绍如何在 Orange Pi 5 Plus 上使用 RKLLM (Rockchip LLM Runtime) 进行大语言模型推理。

## 目录

1. [RKLLM简介](#rkllm简介)
2. [环境准备](#环境准备)
3. [模型转换](#模型转换)
4. [Python API使用](#python-api使用)
5. [测试和验证](#测试和验证)
6. [常见问题](#常见问题)

---

## RKLLM简介

RKLLM 是瑞芯微 (Rockchip) 为 RK3588/RK3576 等芯片提供的 LLM 推理框架，具有以下特点：

- **NPU加速**: 利用 RK3588 的 6 TOPS NPU 算力
- **低功耗**: 相比 GPU 推理功耗更低
- **本地部署**: 无需联网，保护隐私
- **支持量化**: w4a16、w8a8 等量化格式，节省内存

### 支持的模型

- Qwen/Qwen2 系列
- Llama/Llama2 系列
- DeepSeek 系列
- Phi-2/Phi-3 系列
- 其他 HuggingFace 格式的 LLM

---

## 环境准备

### 1. 硬件要求

- **设备**: Orange Pi 5 Plus (RK3588)
- **内存**: 4GB+ (推荐 8GB)
- **存储**: 16GB+ 可用空间
- **NPU驱动**: 确保 NPU 驱动版本 >= 0.9.6

检查 NPU 驱动：
```bash
cat /sys/kernel/debug/rknpu/version
```

### 2. 安装RKLLM Runtime

#### 方法1: 自动安装 (推荐)

运行项目提供的安装脚本：
```bash
./scripts/install.sh
```

脚本会自动下载 `librkllmrt.so` 并安装到系统目录。

#### 方法2: 手动安装

从 GitHub 下载最新版本：
```bash
# 下载 RKLLM Runtime
wget https://github.com/airockchip/rknn-llm/releases/download/v1.2.0/rkllm-runtime-1.2.0-linux-aarch64.zip

# 解压
unzip rkllm-runtime-1.2.0-linux-aarch64.zip

# 复制到系统目录
sudo cp librkllmrt.so /usr/lib/
sudo ldconfig

# 或者复制到项目目录
cp librkllmrt.so ~/opi-voice-assistant/
```

### 3. 验证安装

检查库是否正确加载：
```bash
python3 scripts/test_rkllm.py --check
```

成功输出：
```
✓ 系统库 librkllmrt.so 存在
  ✓ 函数 rkllm_init 可用
  ✓ 函数 rkllm_run 可用
  ✓ 函数 rkllm_destroy 可用
```

---

## 模型转换

### 为什么需要转换？

RK3588 只能运行 `.rkllm` 格式的模型，需要将 HuggingFace 格式的模型进行转换。

### 在PC端转换 (推荐)

由于转换过程需要较多内存和算力，建议在 PC (x86_64 Linux) 上进行。

#### 步骤1: 安装RKLLM Toolkit

```bash
# 创建conda环境
conda create -n rkllm python=3.8
conda activate rkllm

# 下载并安装 toolkit
wget https://github.com/airockchip/rknn-llm/releases/download/v1.2.0/rkllm-toolkit-1.2.0-cp38-cp38-linux_x86_64.whl
pip install rkllm-toolkit-1.2.0-cp38-cp38-linux_x86_64.whl
```

#### 步骤2: 下载原始模型

```bash
# 安装git-lfs
git lfs install

# 下载模型
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
cd DeepSeek-R1-Distill-Qwen-1.5B
git lfs pull
```

#### 步骤3: 创建转换脚本

创建 `convert_to_rkllm.py`：

```python
from rkllm.api import RKLLM
import os

# 配置
modelpath = './DeepSeek-R1-Distill-Qwen-1.5B'
llm = RKLLM()

# 加载HuggingFace模型
print("加载模型...")
ret = llm.load_huggingface(model=modelpath, device='cuda')
if ret != 0:
    print('加载失败')
    exit()

# 构建RKLLM模型 (w4a16量化)
print("构建RKLLM模型...")
ret = llm.build(
    do_quantization=True,
    optimization_level=1,
    target_platform='rk3588',
    quantization_type='w4a16'
)
if ret != 0:
    print('构建失败')
    exit()

# 导出
output_path = './DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3588.rkllm'
print(f"导出模型到: {output_path}")
ret = llm.export_rkllm(output_path)
if ret != 0:
    print('导出失败')
    exit()

print('✓ 转换成功!')
```

#### 步骤4: 运行转换

```bash
python convert_to_rkllm.py
```

转换完成后，会生成 `.rkllm` 文件 (约 1GB)。

#### 步骤5: 传输到Orange Pi

```bash
scp DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3588.rkllm \
    orangepi@<orange-pi-ip>:~/opi-voice-assistant/models/llm/
```

### 使用预转换模型

如果不想自己转换，可以下载社区预转换的模型：

```bash
# 从ModelScope下载
pip install modelscope
python3 -c "
from modelscope import snapshot_download
snapshot_download(
    'radxa/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM',
    local_dir='./models/llm/radxa_models'
)
"

# 复制到正确位置
cp models/llm/radxa_models/*.rkllm models/llm/
```

---

## Python API使用

### 基础用法

```python
from src.llm import RKLLMRuntime

# 1. 创建实例
llm = RKLLMRuntime(
    model_path="models/llm/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3588.rkllm",
    max_context_len=2048
)

# 2. 加载模型
if llm.load_model():
    print("模型加载成功!")
else:
    print("模型加载失败")
    exit()

# 3. 单轮对话
response = llm.generate("你好，请介绍一下自己。")
print(response)

# 4. 流式输出 (推荐)
def on_token(token):
    print(token, end="", flush=True)

response = llm.generate(
    "解释一下量子计算",
    callback=on_token
)

# 5. 多轮对话
messages = [
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是..."},
    {"role": "user", "content": "能举个例子吗？"}
]

response = llm.chat(messages, callback=on_token)

# 6. 释放资源
llm.release()
```

### 高级用法

#### 自定义推理参数

```python
response = llm.generate(
    "你好",
    max_new_tokens=512,    # 最大生成token数
    temperature=0.7,        # 温度 (创造性)
    top_p=0.9,             # Top-p采样
    repeat_penalty=1.1,    # 重复惩罚
)
```

#### 备选方案 (Transformers)

如果 RKLLM 不可用，会自动回退到 Transformers：

```python
from src.llm import SimpleLLM

# 使用HuggingFace格式模型
llm = SimpleLLM(
    model_path="models/llm/DeepSeek-R1-Distill-Qwen-1.5B",
    device="cpu"
)

llm.load_model()
response = llm.generate("你好")
```

#### 自动选择模式

```python
from src.llm import create_llm

# 自动选择最佳后端 (优先RKLLM)
llm = create_llm(
    model_path="models/llm/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3588.rkllm",
    use_rkllm=True,
    max_context_len=2048
)

if llm:
    response = llm.generate("你好")
    llm.release()
```

---

## 测试和验证

### 运行测试脚本

```bash
# 检查RKLLM库
python3 scripts/test_rkllm.py --check

# 测试RKLLM模型 (如果有.rkllm文件)
python3 scripts/test_rkllm.py

# 测试Transformers备选 (如果有HF格式模型)
python3 scripts/test_rkllm.py --transformers

# 测试自动选择
python3 scripts/test_rkllm.py --auto
```

### 集成到语音助手

确保模型文件就位后，启动语音助手：

```bash
# 使用本地RKLLM模型
python3 src/main.py

# 强制使用云端API
python3 src/main.py --cloud

# 强制使用本地模型
python3 src/main.py --local
```

---

## 常见问题

### Q1: `librkllmrt.so: cannot open shared object file`

**原因**: RKLLM Runtime库未找到

**解决**:
```bash
# 方法1: 下载库到项目目录
wget https://github.com/airockchip/rknn-llm/releases/download/v1.2.0/rkllm-runtime-1.2.0-linux-aarch64.zip
unzip rkllm-runtime-1.2.0-linux-aarch64.zip
cp librkllmrt.so ./

# 方法2: 安装到系统
sudo cp librkllmrt.so /usr/lib/
sudo ldconfig
```

### Q2: `rkllm_init failed: 无效模型`

**原因**: 模型格式不正确或损坏

**解决**:
- 检查模型文件是否完整
- 重新转换模型
- 确保使用正确的 `target_platform='rk3588'`

### Q3: 推理速度很慢 (< 5 tokens/s)

**原因**: 
1. 没有使用NPU加速
2. NPU驱动版本过旧
3. 系统负载过高

**解决**:
```bash
# 检查NPU驱动
cat /sys/kernel/debug/rknpu/version

# 检查NPU负载
watch -n 1 sudo cat /sys/kernel/debug/rknpu/load

# 设置CPU高性能模式
echo performance | sudo tee /sys/bus/cpu/devices/cpu*/cpufreq/scaling_governor
```

### Q4: 内存不足 (OOM)

**原因**: 4GB内存不足以运行大型模型

**解决**:
1. 使用更大的swap空间：
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

2. 减小 `max_context_len`
3. 关闭其他程序

### Q5: `Segmentation fault`

**原因**: 
1. 模型与RKLLM版本不兼容
2. NPU驱动问题

**解决**:
- 更新 NPU 驱动到最新版本
- 使用与 RKLLM Runtime 匹配的模型版本
- 检查模型转换时使用的 RKLLM Toolkit 版本

---

## 性能参考

在 Orange Pi 5 Plus (RK3588, 4GB RAM) 上的典型性能：

| 模型 | 量化 | 内存占用 | 推理速度 | 推荐场景 |
|------|------|----------|----------|----------|
| DeepSeek-R1-1.5B | w4a16 | ~1.5GB | 8-12 t/s | 日常对话 |
| Qwen2-1.5B | w4a16 | ~1.5GB | 8-12 t/s | 通用任务 |
| TinyLlama-1.1B | w4a16 | ~1.2GB | 10-15 t/s | 简单任务 |

*注: t/s = tokens/second*

---

## 参考链接

- [RKNN-LLM GitHub](https://github.com/airockchip/rknn-llm)
- [RKLLM Toolkit 文档](https://github.com/airockchip/rknn-llm/tree/main/rkllm-toolkit)
- [RKLLM Runtime 示例](https://github.com/airockchip/rknn-llm/tree/main/rkllm-runtime/examples)
- [Orange Pi 5 Plus 官方文档](http://www.orangepi.cn/)

---

如有问题，请查看项目文档或在 GitHub Issues 中提问。
