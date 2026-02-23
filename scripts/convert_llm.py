#!/usr/bin/env python3
"""
LLM模型转换脚本
将HuggingFace格式的模型转换为RKLLM格式 (用于RK3588 NPU)

注意: 此脚本需要在x86_64 Linux PC上运行，需要安装RKLLM Toolkit
"""
import os
import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """检查运行环境"""
    import platform
    if platform.machine() != 'x86_64':
        logger.warning("此脚本应在x86_64 Linux PC上运行")
        logger.warning("当前架构: " + platform.machine())
        return False
    
    try:
        from rkllm.api import RKLLM
        logger.info("✓ RKLLM Toolkit 已安装")
        return True
    except ImportError:
        logger.error("✗ 未安装 RKLLM Toolkit")
        logger.error("请运行: pip install rkllm-toolkit")
        return False


def convert_model(model_path: str, output_path: str, target_platform: str = 'rk3588'):
    """
    转换模型为RKLLM格式
    
    Args:
        model_path: HuggingFace模型路径
        output_path: 输出的RKLLM文件路径
        target_platform: 目标平台 (rk3588/rk3576)
    """
    try:
        from rkllm.api import RKLLM
    except ImportError:
        logger.error("请先安装 RKLLM Toolkit")
        return False
    
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return False
    
    logger.info("=" * 60)
    logger.info("开始转换模型")
    logger.info("=" * 60)
    logger.info(f"输入模型: {model_path}")
    logger.info(f"输出路径: {output_path}")
    logger.info(f"目标平台: {target_platform}")
    
    llm = RKLLM()
    
    logger.info("\n[1/3] 加载HuggingFace模型...")
    ret = llm.load_huggingface(model=model_path, device='cuda')
    if ret != 0:
        logger.error('加载模型失败')
        return False
    logger.info("✓ 模型加载成功")
    
    logger.info("\n[2/3] 构建RKLLM模型 (w4a16量化)...")
    ret = llm.build(
        do_quantization=True,
        optimization_level=1,
        target_platform=target_platform,
        quantization_type='w4a16'
    )
    if ret != 0:
        logger.error('构建失败')
        return False
    logger.info("✓ 模型构建成功")
    
    logger.info("\n[3/3] 导出RKLLM模型...")
    ret = llm.export_rkllm(output_path)
    if ret != 0:
        logger.error('导出失败')
        return False
    logger.info(f"✓ 模型已导出到: {output_path}")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"  文件大小: {file_size:.1f} MB")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ 转换完成!")
    logger.info("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='将HuggingFace模型转换为RKLLM格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换本地模型
  python convert_llm.py ./DeepSeek-R1-Distill-Qwen-1.5B -o ./model.rkllm

  # 指定目标平台
  python convert_llm.py ./model -o ./model.rkllm --platform rk3588

安装RKLLM Toolkit:
  pip install rkllm-toolkit-1.2.0-cp38-cp38-linux_x86_64.whl

下载地址:
  https://github.com/airockchip/rknn-llm/releases
        """
    )
    
    parser.add_argument('model_path', help='HuggingFace模型路径')
    parser.add_argument('-o', '--output', required=True, help='输出RKLLM文件路径')
    parser.add_argument('--platform', default='rk3588', 
                        choices=['rk3588', 'rk3576'],
                        help='目标平台 (默认: rk3588)')
    parser.add_argument('--skip-check', action='store_true', 
                        help='跳过环境检查')
    
    args = parser.parse_args()
    
    if not args.skip_check:
        if not check_environment():
            sys.exit(1)
    
    success = convert_model(
        model_path=args.model_path,
        output_path=args.output,
        target_platform=args.platform
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
