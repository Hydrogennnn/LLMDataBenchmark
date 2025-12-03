#!/bin/bash
#SBATCH -J safety_ddp
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH -o logs/safety_ddp_%j.out
#SBATCH -e logs/safety_ddp_%j.err
#SBATCH --time=12:00:00

# 创建日志目录
mkdir -p logs

echo "=========================================="
echo "🚀 Multi-GPU DDP Safety Evaluation"
echo "    Model: Qwen2.5-32B-Instruct"
echo "    GPUs: 8 (真正的数据并行)"
echo "    Batch per GPU: 8"
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

# 显示 GPU 信息
echo -e "\n📊 GPU 信息:"
nvidia-smi

# 设置环境变量
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# 运行评估
echo -e "\n🔥 开始运行 DDP 安全评估..."
cd /mnt/petrelfs/liuhaoze/main/xlam_60k

python ddp.py \
    --dataset /mnt/petrelfs/liuhaoze/datasets/xlam_60k.jsonl \
    --model /mnt/petrelfs/liuhaoze/models1/Qwen2.5-32B-Instruct \
    --num-gpus 8 \
    --batch-size 8

echo -e "\n=========================================="
echo "✅ 任务完成"
echo "End Time: $(date)"
echo "=========================================="

