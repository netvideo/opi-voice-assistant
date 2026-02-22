#!/usr/bin/env python3
"""
RKLLM 测试和示例脚本
测试本地RKLLM模型的初始化和推理
"""
import sys
import os
import time
import logging

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm import RKLLMRuntime, SimpleLLM, create_llm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_rkllm():
    """测试RKLLM模型"""
    print("=" * 60)
    print("RKLLM 模型测试")
    print("=" * 60)
    
    # 模型路径 (请根据实际情况修改)
    model_path = "models/llm/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3588.rkllm"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先下载或转换RKLLM模型")
        return False
    
    try:
        print(f"\n1. 初始化RKLLM运行时...")
        print(f"   模型路径: {model_path}")
        
        llm = RKLLMRuntime(model_path, max_context_len=2048)
        
        print("\n2. 加载模型...")
        if not llm.load_model():
            print("❌ 模型加载失败")
            return False
        
        print("\n3. 测试单轮对话...")
        prompt = "你好，请介绍一下自己。"
        print(f"   用户: {prompt}")
        print(f"   助手: ", end="", flush=True)
        
        response = llm.generate(prompt)
        print(response)
        
        print("\n4. 测试流式输出...")
        prompt2 = "解释一下量子计算。"
        print(f"   用户: {prompt2}")
        print(f"   助手: ", end="", flush=True)
        
        def stream_callback(token):
            print(token, end="", flush=True)
        
        start_time = time.time()
        response2 = llm.generate(prompt2, callback=stream_callback)
        elapsed = time.time() - start_time
        
        print(f"\n   (生成耗时: {elapsed:.2f}s)")
        print(f"   (生成速度: {len(response2)/elapsed:.1f} chars/s)")
        
        print("\n5. 测试多轮对话...")
        messages = [
            {"role": "user", "content": "什么是机器学习？"},
            {"role": "assistant", "content": "机器学习是人工智能的一个分支..."},
            {"role": "user", "content": "能举个例子吗？"}
        ]
        
        print(f"   对话历史:")
        for msg in messages:
            print(f"     {msg['role']}: {msg['content'][:50]}...")
        
        print(f"   助手: ", end="", flush=True)
        response3 = llm.chat(messages)
        print(response3)
        
        print("\n6. 释放资源...")
        llm.release()
        
        print("\n✅ RKLLM测试完成!")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformers_fallback():
    """测试Transformers备选方案"""
    print("\n" + "=" * 60)
    print("Transformers 备选方案测试")
    print("=" * 60)
    
    # 注意: 这需要下载完整的HuggingFace模型 (约3GB)
    model_path = "models/llm/DeepSeek-R1-Distill-Qwen-1.5B"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型目录不存在: {model_path}")
        print("请先下载HuggingFace格式的模型")
        return False
    
    try:
        print(f"\n1. 加载Transformers模型...")
        llm = SimpleLLM(model_path, device="cpu")
        
        if not llm.load_model():
            print("❌ 模型加载失败")
            return False
        
        print("\n2. 测试生成...")
        prompt = "你好"
        print(f"   用户: {prompt}")
        print(f"   助手: ", end="", flush=True)
        
        start_time = time.time()
        response = llm.generate(prompt, max_new_tokens=100)
        elapsed = time.time() - start_time
        
        print(response)
        print(f"   (生成耗时: {elapsed:.2f}s)")
        
        print("\n3. 释放资源...")
        llm.release()
        
        print("\n✅ Transformers测试完成!")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_select():
    """测试自动选择模式"""
    print("\n" + "=" * 60)
    print("自动选择模式测试")
    print("=" * 60)
    
    # 尝试自动选择 (优先RKLLM，失败则回退到Transformers)
    model_path_rkllm = "models/llm/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3588.rkllm"
    model_path_hf = "models/llm/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # 使用存在的模型路径
    model_path = model_path_rkllm if os.path.exists(model_path_rkllm) else model_path_hf
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到任何可用模型")
        return False
    
    try:
        print(f"\n1. 自动创建LLM实例...")
        llm = create_llm(model_path, use_rkllm=True)
        
        if not llm:
            print("❌ 创建LLM实例失败")
            return False
        
        print(f"   使用的LLM类型: {type(llm).__name__}")
        
        print("\n2. 测试推理...")
        prompt = "你好"
        print(f"   用户: {prompt}")
        print(f"   助手: ", end="", flush=True)
        
        response = llm.generate(prompt)
        print(response)
        
        print("\n3. 释放资源...")
        llm.release()
        
        print("\n✅ 自动选择测试完成!")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_rkllm_library():
    """检查RKLLM库是否可用"""
    print("=" * 60)
    print("RKLLM 库检查")
    print("=" * 60)
    
    import ctypes
    
    # 检查系统库
    try:
        lib = ctypes.CDLL("librkllmrt.so")
        print("✓ 系统库 librkllmrt.so 存在")
        
        # 检查关键函数
        functions = ['rkllm_init', 'rkllm_run', 'rkllm_destroy']
        for func in functions:
            if hasattr(lib, func):
                print(f"  ✓ 函数 {func} 可用")
            else:
                print(f"  ⚠ 函数 {func} 不可用")
        
        return True
        
    except OSError:
        print("❌ 系统库 librkllmrt.so 不存在")
        
        # 检查本地库
        local_lib = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'librkllmrt.so'
        )
        if os.path.exists(local_lib):
            print(f"✓ 本地库存在: {local_lib}")
            return True
        else:
            print(f"❌ 本地库也不存在")
            print("\n请从以下地址下载RKLLM Runtime:")
            print("https://github.com/airockchip/rknn-llm/releases")
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RKLLM 测试脚本')
    parser.add_argument('--check', action='store_true', help='只检查库')
    parser.add_argument('--transformers', action='store_true', help='测试Transformers')
    parser.add_argument('--auto', action='store_true', help='测试自动选择模式')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("RKLLM Python 接口测试")
    print("=" * 60 + "\n")
    
    # 检查库
    has_library = check_rkllm_library()
    
    if args.check:
        return
    
    results = []
    
    # 测试RKLLM
    if has_library and not args.transformers:
        results.append(("RKLLM", test_rkllm()))
    
    # 测试Transformers
    if args.transformers or not has_library:
        results.append(("Transformers", test_transformers_fallback()))
    
    # 测试自动选择
    if args.auto:
        results.append(("自动选择", test_auto_select()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s} {status}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
