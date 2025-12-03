#!/bin/bash
#SBATCH -J safety_eval_llm
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH -o logs/safety_eval_llm_%j.out
#SBATCH -e logs/safety_eval_llm_%j.err
#SBATCH --time=12:00:00

# 创建日志目录
mkdir -p logs

echo "=========================================="
echo "🛡️  Safety Evaluation (SafeToolBench Framework)"
echo "    Model: Qwen2.5-32B-Instruct"
echo "    GPUs: 8 (batch inference)"
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

# 运行安全评估脚本（使用batch推理加速）
echo -e "\n🔥 开始运行安全评估 (batch_size=32)..."
cd /mnt/petrelfs/liuhaoze/main/xlam_60k

python evaluate_xlam_trustworthy_llm.py \
    --dataset /mnt/petrelfs/liuhaoze/datasets/xlam_60k.jsonl \
    --local \
    --model /mnt/petrelfs/liuhaoze/models1/Qwen2.5-32B-Instruct \
    --batch-size 32

echo -e "\n=========================================="
echo "✅ 任务完成"
echo "End Time: $(date)"
echo "=========================================="

