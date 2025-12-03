#!/bin/bash
#SBATCH -J safety_accel
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH -o logs/safety_accel_%j.out
#SBATCH -e logs/safety_accel_%j.err
#SBATCH --time=12:00:00

# 创建日志目录
mkdir -p logs

echo "=========================================="
echo "🚀 Accelerate Multi-GPU Safety Evaluation"
echo "    Model: Qwen2.5-32B-Instruct"
echo "    GPUs: 8 (Accelerate 自动并行)"
echo "    Batch Size: 16"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

# 验证环境
echo -e "\n🔍 Python 环境:"
which python
python --version

echo -e "\n📦 检查 accelerate:"
pip show accelerate || echo "⚠️  需要安装: pip install accelerate"

# 显示 GPU 信息
echo -e "\n📊 GPU 信息:"
nvidia-smi

# 设置环境变量
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# 运行评估（使用 accelerate launch）
echo -e "\n🔥 开始运行 Accelerate 安全评估..."
cd /mnt/petrelfs/liuhaoze/main/xlam_60k

accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision no \
    --multi_gpu \
    evaluate_xlam_trustworthy_llm.py \
    --dataset /mnt/petrelfs/liuhaoze/datasets/xlam_60k.jsonl \
    --local \
    --model /mnt/petrelfs/liuhaoze/models1/Qwen2.5-32B-Instruct \
    --batch-size 16 \
    --accelerate

echo -e "\n=========================================="
echo "✅ 任务完成"
echo "End Time: $(date)"
echo "=========================================="

