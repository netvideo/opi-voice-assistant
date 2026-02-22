#!/bin/bash
# Orange Pi 5 Plus 语音助手环境安装脚本

set -e

echo "=========================================="
echo "  Orange Pi 5 Plus 语音助手 - 环境安装"
echo "=========================================="

# 检查是否在Orange Pi上
if ! grep -q "rk3588" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️  警告: 未检测到RK3588芯片"
    echo "此脚本专为Orange Pi 5 Plus / RK3588设备设计"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查内存
echo ""
echo "[1/6] 检查系统资源..."
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
echo "  内存: ${TOTAL_MEM}MB"
if [ $TOTAL_MEM -lt 4096 ]; then
    echo "  ⚠️  警告: 内存小于4GB，可能影响性能"
fi

# 更新系统
echo ""
echo "[2/6] 更新系统软件包..."
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-venv \
    git \
    wget \
    cmake \
    build-essential \
    libsndfile1 \
    portaudio19-dev \
    libportaudio2 \
    python3-pyaudio

# 创建虚拟环境
echo ""
echo "[3/6] 创建Python虚拟环境..."
cd /home/kwh/pi/opi-voice-assistant
python3 -m venv venv
source venv/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel

# 安装Python依赖
echo ""
echo "[4/6] 安装Python依赖..."
pip install -r requirements.txt

# 安装RKLLM Runtime (RK3588)
echo ""
echo "[5/6] 安装RKLLM Runtime..."
if [ ! -f "librkllmrt.so" ]; then
    echo "  下载RKLLM Runtime..."
    # 从GitHub下载RKLLM runtime
    RKLLM_VERSION="1.2.0"
    wget -q --show-progress \
        "https://github.com/airockchip/rknn-llm/releases/download/v${RKLLM_VERSION}/rkllm-runtime-${RKLLM_VERSION}-linux-aarch64.zip" \
        -O /tmp/rkllm-runtime.zip
    
    unzip -q /tmp/rkllm-runtime.zip -d /tmp/rkllm
    cp /tmp/rkllm/librkllmrt.so ./
    sudo cp /tmp/rkllm/librkllmrt.so /usr/lib/
    sudo ldconfig
    rm -rf /tmp/rkllm /tmp/rkllm-runtime.zip
    echo "  ✓ RKLLM Runtime安装完成"
else
    echo "  ✓ RKLLM Runtime已存在"
fi

# 设置音频
echo ""
echo "[6/6] 配置音频设备..."
# 检测音频设备
if arecord -l 2>/dev/null | grep -q "card"; then
    echo "  ✓ 检测到录音设备:"
    arecord -l | grep "card"
else
    echo "  ⚠️  未检测到录音设备，请连接USB麦克风"
fi

if aplay -l 2>/dev/null | grep -q "card"; then
    echo "  ✓ 检测到播放设备:"
    aplay -l | grep "card"
else
    echo "  ⚠️  未检测到播放设备，请连接扬声器或耳机"
fi

# 创建必要的目录
echo ""
echo "[完成] 创建项目目录..."
mkdir -p models/{asr,llm,tts}
mkdir -p cache
mkdir -p logs

# 设置权限
chmod +x scripts/*.sh

echo ""
echo "=========================================="
echo "  ✅ 环境安装完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 下载模型: ./scripts/download_models.sh"
echo "  2. 启动助手: python3 src/main.py"
echo ""
echo "注意: 如果遇到问题，请查看 docs/deployment.md"
echo ""
