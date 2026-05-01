#!/usr/bin/env python
"""环境设置脚本 - 使用 uv 管理 Python 环境

支持的 CUDA 版本:
  cu121  - CUDA 12.1  (适用于 RTX 4070/4080/4090 等 Ada Lovelace 显卡)
  cu118  - CUDA 11.8  (适用于 RTX 3090/A100 等 Ampere 显卡)
  cpu    - CPU 仅版本

示例用法:
  python setup_env.py --env benchmark           # 安装评估框架
  python setup_env.py --env glim --cuda cu121   # 安装 GLIM + CUDA 12.1
  python setup_env.py --env all --cuda cu121    # 安装全部 + CUDA 12.1
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# CUDA 版本对应的 PyTorch wheel index URL
CUDA_INDEX_URLS = {
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu117": "https://download.pytorch.org/whl/cu117",
    "cpu":   "https://download.pytorch.org/whl/cpu",
}


def check_uv():
    """检查 uv 是否已安装"""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)
        print(f"uv: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("错误: uv 未安装")
        print("安装 uv:")
        print("  Windows: winget install uv")
        print("  macOS: brew install uv")
        print("  Linux: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False


def detect_cuda():
    """尝试自动检测 CUDA 版本"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # 简单估算: 驱动版本 >= 525 对应 CUDA 12.x
            driver_ver = result.stdout.strip().split("\n")[0].split(".")[0]
            if int(driver_ver) >= 525:
                return "cu121"
            elif int(driver_ver) >= 450:
                return "cu118"
    except Exception:
        pass
    return None


def install_torch(cuda="cu121"):
    """安装 PyTorch with 正确的 CUDA wheel"""
    print(f"\n安装 PyTorch (目标环境: {cuda})...")
    if cuda == "cpu" or cuda not in CUDA_INDEX_URLS:
        cmd = ["uv", "pip", "install", "torch>=2.0.0", "torchvision", "torchaudio"]
    else:
        index_url = CUDA_INDEX_URLS[cuda]
        cmd = [
            "uv", "pip", "install",
            "torch>=2.0.0", "torchvision", "torchaudio",
            "--extra-index-url", index_url
        ]
    print(f"  命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("✔ PyTorch 安装完成!")


def install_env(name, cuda="cu121"):
    """安装指定环境"""
    os.chdir(PROJECT_ROOT)
    install_torch(cuda)

    paths = {
        "benchmark": "benchmark_eval",
        "eeg-to-text": "models/EEG-To-Text-main",
        "eeg2text": "models/EEG2Text-main",
        "cet-mae": "models/CET-MAE",
        "glim": "models/GLIM-main",
    }

    if name not in paths:
        print(f"未知环境: {name}")
        return

    path = paths[name]
    print(f"\n安装项目依赖: {path}...")
    subprocess.run(["uv", "pip", "install", "-e", path], check=True)
    print(f"✔ 安装完成: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="环境设置（使用 uv）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python setup_env.py --env benchmark                # 安装评估框架
  python setup_env.py --env glim --cuda cu121        # RTX 4070 用 CUDA 12.1
  python setup_env.py --env all --cuda cu118         # A100服务器用 CUDA 11.8
  python setup_env.py --env eeg-to-text --cuda cpu   # 无显卡用 CPU 版
"""
    )
    parser.add_argument(
        "--env", default="list",
        choices=["list", "benchmark", "eeg-to-text", "eeg2text", "cet-mae", "glim", "all"],
        help="要安装的环境"
    )
    parser.add_argument(
        "--cuda", default=None,
        choices=list(CUDA_INDEX_URLS.keys()),
        help="CUDA 版本 (RTX 4070 请用 cu121, 默认自动检测)"
    )
    args = parser.parse_args()

    if not check_uv():
        return 1

    # 自动检测 CUDA 版本
    cuda = args.cuda
    if cuda is None:
        detected = detect_cuda()
        if detected:
            print(f"自动检测到 CUDA 环境: {detected}")
            cuda = detected
        else:
            cuda = "cu121"  # 默认 CUDA 12.1
            print(f"未能自动检测 CUDA，使用默认: {cuda}")
    print(f"目标 CUDA 版本: {cuda}")

    if args.env == "list":
        print("\n可用环境:")
        print("  benchmark   - 基准评估框架")
        print("  eeg-to-text - EEG-To-Text 模型  (原始 1.9+cu111 不支持 RTX 4070, 已升级 >=2.0)")
        print("  eeg2text    - EEG2Text 模型    (原始 1.12+cu118 不支持 RTX 4070, 已升级 >=2.0)")
        print("  cet-mae     - CET-MAE 模型     (无版本限制, 直接兼容)")
        print("  glim        - GLIM 模型        (PyTorch>=2.0, Lightning 2.4, BF16 原生支持)")
        print("  all         - 安装全部")
        print(f"\n建议 RTX 4070 用户: --cuda cu121")
    elif args.env == "all":
        install_env("benchmark", cuda)
        for e in ["eeg-to-text", "eeg2text", "cet-mae", "glim"]:
            install_env(e, cuda)
    else:
        install_env(args.env, cuda)

    print("\n✅ 环境设置完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
