#!/bin/bash
# 模型下载脚本

set -e

echo "=========================================="
echo "  下载语音助手所需模型"
echo "=========================================="

# 配置
MODELS_DIR="/home/kwh/pi/opi-voice-assistant/models"
HF_MIRROR="https://hf-mirror.com"  # 国内镜像

# 创建目录
mkdir -p ${MODELS_DIR}/{asr,llm,tts}

echo ""
echo "[1/3] 下载 ASR 模型 (qwen3-asr-0.6b)..."
cd ${MODELS_DIR}/asr
if [ ! -d "Qwen3-ASR-0.6B" ]; then
    echo "  正在下载..."
    # 使用git-lfs下载
    if ! command -v git-lfs &> /dev/null; then
        sudo apt install -y git-lfs
        git lfs install
    fi
    
    # 使用镜像加速
    GIT_LFS_SKIP_SMUDGE=1 git clone ${HF_MIRROR}/Qwen/Qwen3-ASR-0.6B Qwen3-ASR-0.6B || {
        echo "  镜像下载失败，尝试官方源..."
        GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen3-ASR-0.6B Qwen3-ASR-0.6B
    }
    cd Qwen3-ASR-0.6B
    git lfs pull
    cd ..
    echo "  ✓ ASR模型下载完成"
else
    echo "  ✓ ASR模型已存在"
fi

echo ""
echo "[2/3] 下载 TTS 模型 (qwen3-tts-0.6b)..."
cd ${MODELS_DIR}/tts
if [ ! -d "Qwen3-TTS-0.6B" ]; then
    echo "  正在下载..."
    GIT_LFS_SKIP_SMUDGE=1 git clone ${HF_MIRROR}/Qwen/Qwen3-TTS-0.6B Qwen3-TTS-0.6B || {
        echo "  镜像下载失败，尝试官方源..."
        GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen3-TTS-0.6B Qwen3-TTS-0.6B
    }
    cd Qwen3-TTS-0.6B
    git lfs pull
    cd ..
    echo "  ✓ TTS模型下载完成"
else
    echo "  ✓ TTS模型已存在"
fi

echo ""
echo "[3/3] 下载 LLM 模型 (DeepSeek-R1-Distill-Qwen-1.5B)..."
cd ${MODELS_DIR}/llm

# 检查是否已有RKLLM格式模型
if [ ! -f "DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3588.rkllm" ]; then
    echo "  方法1: 尝试下载预转换的RKLLM模型..."
    
    # 尝试从ModelScope下载（国内更快）
    pip install modelscope -q
    python3 << 'EOF'
from modelscope import snapshot_download
import os

try:
    model_dir = snapshot_download(
        'radxa/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM',
        local_dir='/home/kwh/pi/opi-voice-assistant/models/llm/radxa_models'
    )
    print(f"✓ 从ModelScope下载成功: {model_dir}")
except Exception as e:
    print(f"ModelScope下载失败: {e}")
    print("将使用手动转换方式")
EOF
    
    # 查找下载的模型文件
    if [ -d "radxa_models" ]; then
        RKLLM_FILE=$(find radxa_models -name "*.rkllm" | head -1)
        if [ -n "$RKLLM_FILE" ]; then
            cp "$RKLLM_FILE" DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3588.rkllm
            echo "  ✓ RKLLM模型准备完成"
        fi
    fi
fi

# 如果没有RKLLM格式，下载原始模型
if [ ! -f "DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3588.rkllm" ]; then
    echo ""
    echo "  方法2: 下载原始模型并转换..."
    
    if [ ! -d "DeepSeek-R1-Distill-Qwen-1.5B" ]; then
        echo "  下载原始模型..."
        GIT_LFS_SKIP_SMUDGE=1 git clone ${HF_MIRROR}/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B DeepSeek-R1-Distill-Qwen-1.5B || {
            echo "  镜像失败，尝试官方源..."
            GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B DeepSeek-R1-Distill-Qwen-1.5B
        }
        cd DeepSeek-R1-Distill-Qwen-1.5B
        git lfs pull
        cd ..
    fi
    
    echo ""
    echo "  ⚠️  需要在PC端转换模型为RKLLM格式"
    echo "     请参考 docs/deployment.md 中的转换指南"
    echo "     或者使用预转换的模型文件"
fi

cd ${MODELS_DIR}/llm

echo ""
echo "=========================================="
echo "  模型下载完成!"
echo "=========================================="
echo ""
echo "模型位置:"
echo "  ASR: ${MODELS_DIR}/asr/Qwen3-ASR-0.6B"
echo "  TTS: ${MODELS_DIR}/tts/Qwen3-TTS-0.6B"
echo "  LLM: ${MODELS_DIR}/llm/"
ls -lh ${MODELS_DIR}/llm/*.rkllm 2>/dev/null || echo "  (LLM模型需要转换或下载预转换版本)"
echo ""
echo "下一步:"
echo "  如果LLM模型未准备就绪，请参考 docs/deployment.md 进行转换"
echo "  或直接下载预转换模型"
echo ""
