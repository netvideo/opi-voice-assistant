#!/bin/bash
# 模型下载脚本
# 支持选择不同的LLM模型

set -e

echo "=========================================="
echo "  下载语音助手所需模型"
echo "=========================================="

# 配置
MODELS_DIR="/home/kwh/pi/opi-voice-assistant/models"
HF_MIRROR="https://hf-mirror.com"

# 创建目录
mkdir -p ${MODELS_DIR}/{asr,llm,tts}

# ===========================================
# 选择LLM模型
# ===========================================
echo ""
echo "请选择LLM模型:"
echo "  1) Qwen3-0.6B  (~700MB,  适合4GB内存, 支持function calling)"
echo "  2) Qwen3-1.7B  (~1.8GB,  适合8GB内存, 支持function calling) [推荐]"
echo "  3) DeepSeek-R1-Distill-Qwen-1.5B (~1GB, 无function calling)"
echo ""
read -p "请输入选择 [1-3, 默认2]: " llm_choice
llm_choice=${llm_choice:-2}

case $llm_choice in
    1)
        LLM_MODEL="Qwen3-0.6B"
        LLM_FILE="Qwen3-0.6B-rk3588-w8a8.rkllm"
        LLM_HF="dulimov/Qwen3-0.6B-rk3588-1.2.1-unsloth-16k"
        ;;
    2)
        LLM_MODEL="Qwen3-1.7B"
        LLM_FILE="Qwen3-1.7B-rk3588-w8a8.rkllm"
        LLM_HF="GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3"
        ;;
    3)
        LLM_MODEL="DeepSeek-R1-Distill-Qwen-1.5B"
        LLM_FILE="DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3588.rkllm"
        LLM_HF=""
        ;;
    *)
        echo "无效选择，使用默认 Qwen3-1.7B"
        LLM_MODEL="Qwen3-1.7B"
        LLM_FILE="Qwen3-1.7B-rk3588-w8a8.rkllm"
        LLM_HF="GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3"
        ;;
esac

echo ""
echo "已选择: ${LLM_MODEL}"

# ===========================================
# 下载ASR模型
# ===========================================
echo ""
echo "[1/3] 下载 ASR 模型 (Qwen3-ASR-0.6B)..."
cd ${MODELS_DIR}/asr
if [ ! -d "Qwen3-ASR-0.6B" ]; then
    echo "  正在下载..."
    if ! command -v git-lfs &> /dev/null; then
        sudo apt install -y git-lfs
        git lfs install
    fi
    
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

# ===========================================
# 下载TTS模型
# ===========================================
echo ""
echo "[2/3] 下载 TTS 模型 (Qwen3-TTS-0.6B)..."
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

# ===========================================
# 下载LLM模型
# ===========================================
echo ""
echo "[3/3] 下载 LLM 模型 (${LLM_MODEL})..."
cd ${MODELS_DIR}/llm

if [ -f "${LLM_FILE}" ]; then
    echo "  ✓ LLM模型已存在: ${LLM_FILE}"
else
    echo "  正在下载 ${LLM_MODEL} RKLLM模型..."
    
    pip install modelscope huggingface_hub -q 2>/dev/null || true
    
    python3 << EOF
import os
import sys

llm_file = "${LLM_FILE}"
llm_hf = "${LLM_HF}"
llm_model = "${LLM_MODEL}"

if llm_model == "DeepSeek-R1-Distill-Qwen-1.5B":
    # DeepSeek模型从ModelScope下载
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download(
            'radxa/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM',
            local_dir='/home/kwh/pi/opi-voice-assistant/models/llm/deepseek_tmp'
        )
        import glob
        rkllm_files = glob.glob(f"{model_dir}/**/*.rkllm", recursive=True)
        if rkllm_files:
            import shutil
            shutil.copy(rkllm_files[0], f"/home/kwh/pi/opi-voice-assistant/models/llm/{llm_file}")
            print(f"✓ 下载成功: {llm_file}")
    except Exception as e:
        print(f"ModelScope下载失败: {e}")
        sys.exit(1)
else:
    # Qwen3模型从HuggingFace下载
    try:
        from huggingface_hub import hf_hub_download
        hf_mirror = "${HF_MIRROR}"
        
        # 获取仓库中的rkllm文件列表
        from huggingface_hub import list_repo_files
        files = list_repo_files(llm_hf, endpoint=hf_mirror)
        rkllm_files = [f for f in files if f.endswith('.rkllm')]
        
        if rkllm_files:
            # 下载第一个rkllm文件
            downloaded = hf_hub_download(
                repo_id=llm_hf,
                filename=rkllm_files[0],
                local_dir="/home/kwh/pi/opi-voice-assistant/models/llm",
                endpoint=hf_mirror
            )
            # 重命名为标准名称
            import shutil
            target = f"/home/kwh/pi/opi-voice-assistant/models/llm/{llm_file}"
            if downloaded != target:
                shutil.move(downloaded, target)
            print(f"✓ 下载成功: {llm_file}")
        else:
            print("未找到RKLLM文件")
            sys.exit(1)
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动从以下地址下载:")
        print(f"  https://huggingface.co/{llm_hf}")
        sys.exit(1)
EOF
    
    if [ -f "${LLM_FILE}" ]; then
        echo "  ✓ LLM模型下载完成"
    else
        echo "  ✗ LLM模型下载失败"
        echo "  请手动从以下地址下载:"
        if [ -n "${LLM_HF}" ]; then
            echo "    https://huggingface.co/${LLM_HF}"
        else
            echo "    https://modelscope.cn/models/radxa/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM"
        fi
    fi
fi

# ===========================================
# 更新配置文件
# ===========================================
CONFIG_FILE="/home/kwh/pi/opi-voice-assistant/config/config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    sed -i "s|llm: models/llm/.*\.rkllm|llm: models/llm/${LLM_FILE}|g" "$CONFIG_FILE"
    echo ""
    echo "  ✓ 已更新配置文件"
fi

# ===========================================
# 完成
# ===========================================
echo ""
echo "=========================================="
echo "  模型下载完成!"
echo "=========================================="
echo ""
echo "模型位置:"
echo "  ASR: ${MODELS_DIR}/asr/Qwen3-ASR-0.6B"
echo "  TTS: ${MODELS_DIR}/tts/Qwen3-TTS-0.6B"
echo "  LLM: ${MODELS_DIR}/llm/${LLM_FILE}"
echo ""
ls -lh ${MODELS_DIR}/llm/*.rkllm 2>/dev/null
echo ""
echo "内存占用估算:"
case $llm_choice in
    1) echo "  ~700MB  (适合4GB内存设备)" ;;
    2) echo "  ~1.8GB  (适合8GB内存设备)" ;;
    3) echo "  ~1GB    (适合4GB内存设备)" ;;
esac
echo ""
echo "下一步:"
echo "  python3 src/main.py"
echo ""
