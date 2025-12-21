#!/bin/bash

###### Part 1: SLURM 配置 ######

# 注意：SLURM 指令必须在脚本最前面，且不能使用变量
# 日志路径是相对于提交作业时的工作目录
# 建议：始终从项目根目录提交：sbatch sub/run_pipeline.sh
# 如果从 sub 目录提交，日志会写到 sub/logs/ 目录

#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=higgsgpu
#SBATCH --job-name=hep_resonance_ad
#SBATCH --ntasks=1
#SBATCH --output=logs/run_pipeline_%j.log
#SBATCH --error=logs/run_pipeline_%j.err
#SBATCH --mem-per-cpu=24576
#SBATCH --gres=gpu:v100:1

###### Part 2: 环境设置 ######

# 显示分配的节点和GPU
srun -l hostname
/usr/bin/nvidia-smi -L
echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"

# 启用错误时退出
set -e

# 首先设置项目根目录（脚本所在目录的父目录）
# 这样后续的路径设置都基于项目根目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# 切换到项目根目录（确保所有相对路径都正确）
cd ${PROJECT_DIR}

# 如果从 sub 目录提交，日志路径可能不对，这里创建正确的日志目录
mkdir -p ${PROJECT_DIR}/logs

# 激活 conda 环境（根据实际情况修改路径）
# 方式1: 使用 conda activate（如果已配置）
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# 方式2: 直接使用 conda 可执行文件路径（推荐）
# 请根据实际环境修改以下路径
CONDA_BASE="/hpcfs/cepc/higgsgpu/wangxinzhu/miniconda3"
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate strange

# 或者如果使用 module load
# module load python/3.x
# module load cuda/11.x

echo "=========================================="
echo "Script directory: ${SCRIPT_DIR}"
echo "Project directory: ${PROJECT_DIR}"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version 2>/dev/null || echo 'nvcc not found')"
echo "=========================================="
echo "Note: If submitted from sub/ directory, logs may be in sub/logs/"
echo "      Current log will be written to: ${PROJECT_DIR}/logs/"
echo "=========================================="

###### Part 3: 配置参数 ######

# 配置文件路径
CONFIG_FILE="configs/upsilon_reproduction.yaml"

# 随机种子
SEED=42

# 是否使用快速模式（减少迭代次数，加快计算）
# 设置为 "true" 使用快速模式，设置为 "false" 使用完整模式
FAST_MODE="false"

# 是否计算所有variations（不同拟合类型和bin数）
# 设置为 "true" 计算所有variations，设置为 "false" 只计算默认配置
ALL_VARIATIONS="true"

# 训练参数
EPOCHS=""  # 留空使用配置文件中的默认值，或指定具体数值如 "100"

# 评估参数
MAX_ITER=""  # 留空使用默认值（15090），快速模式会自动设置为400

###### Part 4: 创建必要的目录 ######

# 确保在项目根目录创建目录
mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${PROJECT_DIR}/outputs
mkdir -p ${PROJECT_DIR}/data/precompiled_data

###### Part 5: 执行完整流程 ######

echo "=========================================="
echo "Step 1: 数据下载（如果需要）"
echo "=========================================="
# 如果数据已存在，可以跳过此步骤
# 如果需要下载数据，取消下面的注释
# python scripts/download_data.py --data-id skimmed_data_2016H_30555

echo "=========================================="
echo "Step 2: 数据加载"
echo "=========================================="
python scripts/load_data.py --config ${CONFIG_FILE}

echo "=========================================="
echo "Step 3: 区域定义"
echo "=========================================="
python scripts/define_regions.py --config ${CONFIG_FILE}

echo "=========================================="
echo "Step 4: 训练 CATHODE"
echo "=========================================="
TRAIN_CMD="python scripts/train_cathode.py --config ${CONFIG_FILE} --seed ${SEED}"
if [ -n "${EPOCHS}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --epochs ${EPOCHS}"
fi
${TRAIN_CMD}

echo "=========================================="
echo "Step 5: 评估和 Bump Hunt"
echo "=========================================="
EVAL_CMD="python scripts/evaluate.py --config ${CONFIG_FILE} --seed ${SEED}"
if [ "${FAST_MODE}" = "true" ]; then
    EVAL_CMD="${EVAL_CMD} --fast"
    echo "Using fast mode (reduced iterations)"
fi
if [ "${ALL_VARIATIONS}" = "true" ]; then
    EVAL_CMD="${EVAL_CMD} --all-variations"
    echo "Computing all variations (cubic/quintic/septic × 8/12/16 bins)"
fi
if [ -n "${MAX_ITER}" ]; then
    EVAL_CMD="${EVAL_CMD} --max-iter ${MAX_ITER}"
fi
${EVAL_CMD}

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Evaluation results saved to:"
echo "  outputs/*/evaluation/bump_hunt_results_seed${SEED}.pkl"
echo ""
echo "Next step: Run plot.sh to generate figures"
echo "  bash sub/plot.sh"

