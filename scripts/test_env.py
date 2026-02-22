#!/usr/bin/env python3
"""
快速测试脚本 - 验证环境是否正确配置
"""
import sys
import os

def check_python_version():
    """检查Python版本"""
    print("[1/6] 检查Python版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python版本过低: {version.major}.{version.minor}")
        print("  需要 Python 3.8+")
        return False

def check_packages():
    """检查必要的包"""
    print("\n[2/6] 检查Python包...")
    packages = [
        'torch',
        'transformers',
        'pyyaml',
        'soundfile',
        'numpy'
    ]
    
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (未安装)")
            missing.append(pkg)
    
    if missing:
        print(f"\n  请安装缺失的包: pip install {' '.join(missing)}")
        return False
    return True

def check_models():
    """检查模型文件"""
    print("\n[3/6] 检查模型文件...")
    
    models = {
        'ASR': 'models/asr/Qwen3-ASR-0.6B',
        'LLM': 'models/llm/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3588.rkllm',
        'TTS': 'models/tts/Qwen3-TTS-0.6B'
    }
    
    all_exist = True
    for name, path in models.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} (不存在)")
            all_exist = False
    
    if not all_exist:
        print("\n  请运行: ./scripts/download_models.sh")
    
    return all_exist

def check_rkllm():
    """检查RKLLM库"""
    print("\n[4/6] 检查RKLLM运行时...")
    
    # 检查本地库
    if os.path.exists('librkllmrt.so'):
        print("  ✓ 本地库: librkllmrt.so")
        return True
    
    # 检查系统库
    import ctypes
    try:
        ctypes.CDLL("librkllmrt.so")
        print("  ✓ 系统库: librkllmrt.so")
        return True
    except OSError:
        print("  ✗ RKLLM库未找到")
        print("  请运行: ./scripts/install.sh")
        return False

def check_audio():
    """检查音频设备"""
    print("\n[5/6] 检查音频设备...")
    
    try:
        import subprocess
        
        # 检查录音设备
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        if result.returncode == 0 and 'card' in result.stdout:
            print("  ✓ 录音设备已找到")
        else:
            print("  ⚠ 未找到录音设备")
        
        # 检查播放设备
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
        if result.returncode == 0 and 'card' in result.stdout:
            print("  ✓ 播放设备已找到")
        else:
            print("  ⚠ 未找到播放设备")
        
        return True
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False

def check_npu():
    """检查NPU"""
    print("\n[6/6] 检查NPU...")
    
    npu_version = '/sys/kernel/debug/rknpu/version'
    if os.path.exists(npu_version):
        try:
            with open(npu_version, 'r') as f:
                version = f.read().strip()
                print(f"  ✓ NPU驱动: {version}")
                return True
        except:
            pass
    
    print("  ✗ NPU驱动未找到 (仅在RK3588设备上可用)")
    return False

def main():
    """主函数"""
    print("="*50)
    print("Orange Pi 5 Plus 语音助手 - 环境检查")
    print("="*50)
    
    checks = [
        check_python_version(),
        check_packages(),
        check_models(),
        check_rkllm(),
        check_audio(),
        check_npu()
    ]
    
    print("\n" + "="*50)
    if all(checks):
        print("✅ 所有检查通过! 可以运行语音助手了。")
        print("\n运行: python3 src/main.py")
    else:
        print("⚠️  部分检查未通过，请根据提示修复问题。")
    print("="*50)

if __name__ == "__main__":
    main()
