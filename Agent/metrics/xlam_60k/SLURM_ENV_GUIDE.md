# Slurm 任务中使用不同 Conda 环境的方法

## 方法 1: 使用 `conda activate`（推荐）

```bash
#!/bin/bash
#SBATCH -J my_job
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --gres=gpu:1

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh

# 激活指定环境
conda activate my_env_name

# 运行脚本
python3 my_script.py
```

## 方法 2: 使用绝对路径（无需激活）

```bash
#!/bin/bash
#SBATCH -J my_job
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --gres=gpu:1

# 直接使用环境的 Python 解释器
/mnt/petrelfs/liuhaoze/anaconda3/envs/my_env_name/bin/python3 my_script.py
```

## 方法 3: 在脚本开头指定 shebang

在 Python 脚本第一行：
```python
#!/mnt/petrelfs/liuhaoze/anaconda3/envs/my_env_name/bin/python3
```

然后在 Slurm 脚本中直接运行：
```bash
./my_script.py
```

## 查看可用环境

```bash
# 列出所有 conda 环境
conda env list

# 或
conda info --envs
```

## 常见问题

### Q1: 为什么 `conda activate` 不工作？
**A:** 需要先初始化 conda：
```bash
source ~/anaconda3/etc/profile.d/conda.sh
```

### Q2: 如何在同一个任务中使用多个环境？
**A:** 可以多次激活：
```bash
conda activate env1
python3 script1.py

conda activate env2
python3 script2.py
```

### Q3: 如何验证当前使用的环境？
**A:** 在脚本中添加：
```bash
echo "当前 Python: $(which python)"
echo "当前环境: $(conda info --envs | grep '*')"
```

## 示例：多环境任务

```bash
#!/bin/bash
#SBATCH -J multi_env_job
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --gres=gpu:1

source ~/anaconda3/etc/profile.d/conda.sh

# 使用 PyTorch 环境做训练
conda activate pytorch_env
echo "Training with: $(which python)"
python3 train.py

# 使用 TensorFlow 环境做评估
conda activate tf_env
echo "Evaluating with: $(which python)"
python3 evaluate.py

# 使用 base 环境做后处理
conda activate base
echo "Post-processing with: $(which python)"
python3 postprocess.py
```

## 最佳实践

1. **明确指定环境**：即使使用 base，也建议显式激活
2. **验证环境**：在运行主程序前打印 Python 路径
3. **记录依赖**：使用 `conda env export > environment.yml`
4. **隔离环境**：不同项目使用不同环境，避免依赖冲突






