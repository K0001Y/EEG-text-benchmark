#!/bin/bash
# EEG2Text RTX 4090 快速安装脚本
# 使用方法: bash install.sh

set -e

echo "=========================================="
echo "EEG2Text CUDA 12 安装脚本"
echo "=========================================="

# 检查 CUDA 版本
echo "[1/5] 检查 CUDA 环境..."
nvidia-smi || echo "警告: 未检测到 NVIDIA GPU"

# 创建并激活 conda 环境
echo "[2/5] 创建 conda 环境..."
conda env create -f environment.yml

# 激活环境
echo "[3/5] 激活环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate EEGToText-CUDA12

# 安装额外的依赖
# pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html  # 如果需要 DGL，取消注释

# 验证安装
echo "[5/5] 验证安装..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('安装完成！')
"

echo ""
echo "=========================================="
echo "使用说明:"
echo "=========================================="
echo "激活环境: conda activate EEGToText-CUDA12"
echo ""
echo "运行训练示例:"
echo "python et_decoding_pretrain_robert.py -m BrainTranslator -t task1 ..."
