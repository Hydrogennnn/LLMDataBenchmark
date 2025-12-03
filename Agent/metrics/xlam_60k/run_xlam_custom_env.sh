#!/bin/bash
#SBATCH -J xlam_test
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o logs/xlam_gpu_%j.out
#SBATCH -e logs/xlam_gpu_%j.err
#SBATCH --time=01:00:00

# 创建日志目录
mkdir -p logs

echo "=========================================="
echo "🚀 xLAM-1b-fc-r GPU 推理任务"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# ============================================
# 方法 1: 激活指定的 conda 环境
# ============================================
# 初始化 conda（如果 .bashrc 没有自动初始化）
source ~/anaconda3/etc/profile.d/conda.sh

# 激活你想要的环境（替换成你的环境名）
conda activate my_env_name

# 验证环境
echo -e "\n🔍 当前 Python 环境:"
which python
conda info --envs | grep "*"

# ============================================
# 方法 2: 直接使用环境的绝对路径（不需要激活）
# ============================================
# /mnt/petrelfs/liuhaoze/anaconda3/envs/my_env_name/bin/python3 test_xlam_gpu_quick.py

# 显示 GPU 信息
echo -e "\n📊 GPU 信息:"
nvidia-smi

# 运行测试脚本
echo -e "\n🔥 开始运行模型测试..."
cd /mnt/petrelfs/liuhaoze
python3 test_xlam_gpu_quick.py

echo -e "\n=========================================="
echo "✅ 任务完成"
echo "End Time: $(date)"
echo "=========================================="

