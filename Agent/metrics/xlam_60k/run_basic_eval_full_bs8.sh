#!/bin/bash
#SBATCH -J basic_eval_full_bs8
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH -o logs/basic_eval_%j.out
#SBATCH -e logs/basic_eval_%j.err
#SBATCH --time=12:00:00

# 创建日志目录
mkdir -p logs

echo "=========================================="
echo "🚀 XLam Dataset Semantic Executability Evaluation (Batch=8)"
echo "    Dataset: xlam_60k.jsonl (60,000 samples)"
echo "    Metric: Semantic Executability (Full Dataset)"
echo "    Model: Qwen2.5-32B-Instruct"
echo "    GPUs: 8 (Accelerate Multi-GPU)"
echo "    Batch Size: 8"
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

echo -e "\n📦 检查依赖:"
pip show transformers torch accelerate || echo "⚠️  需要安装依赖"

# 显示 GPU 信息
echo -e "\n📊 GPU 信息:"
nvidia-smi

# 设置环境变量
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# 运行完整评估
echo -e "\n🔥 开始运行完整基础评估（batch=8）..."
cd /mnt/petrelfs/liuhaoze/main/xlam_60k

accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision no \
    --multi_gpu \
    evaluate_xlam_basic.py \
    --dataset /mnt/petrelfs/liuhaoze/datasets/xlam_60k.jsonl \
    --metric semantic \
    --semantic-model /mnt/petrelfs/liuhaoze/models1/Qwen2.5-32B-Instruct \
    --semantic-max-samples 0 \
    --semantic-batch-size 8 \
    --semantic-max-new-tokens 512 \
    --accelerate

echo -e "\n=========================================="
echo "✅ 任务完成"
echo "End Time: $(date)"
echo "=========================================="
echo -e "\n📁 结果文件位置:"
echo "   /mnt/petrelfs/liuhaoze/datasets/xlam_60k_eval_logs/base_metric/"
echo "=========================================="


#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH -o logs/basic_eval_%j.out
#SBATCH -e logs/basic_eval_%j.err
#SBATCH --time=12:00:00

# 创建日志目录
mkdir -p logs

echo "=========================================="
echo "🚀 XLam Dataset Semantic Executability Evaluation"
echo "    Dataset: xlam_60k.jsonl (60,000 samples)"
echo "    Metric: Semantic Executability (Full Dataset)"
echo "    Model: Qwen2.5-32B-Instruct"
echo "    GPUs: 8 (Accelerate Multi-GPU)"
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

echo -e "\n📦 检查依赖:"
pip show transformers torch accelerate || echo "⚠️  需要安装依赖"

# 显示 GPU 信息
echo -e "\n📊 GPU 信息:"
nvidia-smi

# 设置环境变量
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# 运行完整评估
echo -e "\n🔥 开始运行完整基础评估..."
cd /mnt/petrelfs/liuhaoze/main/xlam_60k

accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision no \
    --multi_gpu \
    evaluate_xlam_basic.py \
    --dataset /mnt/petrelfs/liuhaoze/datasets/xlam_60k.jsonl \
    --metric semantic \
    --semantic-model /mnt/petrelfs/liuhaoze/models1/Qwen2.5-32B-Instruct \
    --semantic-max-samples 0 \
    --semantic-batch-size 16 \
    --semantic-max-new-tokens 512 \
    --accelerate

echo -e "\n=========================================="
echo "✅ 任务完成"
echo "End Time: $(date)"
echo "=========================================="
echo -e "\n📁 结果文件位置:"
echo "   /mnt/petrelfs/liuhaoze/datasets/xlam_60k_eval_logs/base_metric/"
echo "=========================================="





