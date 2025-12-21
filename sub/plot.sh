#!/bin/bash

# 画图脚本（不需要提交到SLURM，在服务器上手动运行）
# 用途：在评估完成后生成所有论文图

set -e

# 激活 conda 环境（与 run_pipeline.sh 保持一致）
CONDA_BASE="/hpcfs/cepc/higgsgpu/wangxinzhu/miniconda3"
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate strange

# 或者如果使用 module load
# module load python/3.x

# 设置项目根目录
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd ${PROJECT_DIR}

echo "=========================================="
echo "Project directory: ${PROJECT_DIR}"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "=========================================="

###### 配置参数 ######

# 配置文件路径
CONFIG_FILE="configs/upsilon_reproduction.yaml"

# 随机种子（必须与 run_pipeline.sh 中的 SEED 一致）
SEED=42

# 评估结果文件路径（根据实际输出路径修改）
# 默认路径格式: outputs/{analysis_name}/evaluation/bump_hunt_results_seed{SEED}.pkl
# 例如: outputs/upsilon_iso_12_21/evaluation/bump_hunt_results_seed42.pkl
# 注意: 路径基于 analysis_keywords.name，不是 dataset_id
EVAL_RESULTS=""  # 留空自动检测，或手动指定路径

# 拟合类型（用于主要图）
FIT_TYPE="quintic"  # cubic, quintic, septic

# SR bin 数量（用于主要图）
NUM_BINS_SR=12  # 8, 12, 16

# 是否包含训练损失图
INCLUDE_TRAINING_PLOTS="true"

# 输出目录（留空使用默认路径）
OUTPUT_DIR=""  # 留空使用 outputs/{analysis_name}/figures（基于 analysis_keywords.name）

###### 自动检测评估结果文件 ######

if [ -z "${EVAL_RESULTS}" ]; then
    # 尝试自动查找最新的评估结果文件
    echo "Auto-detecting evaluation results file..."
    
    # 查找所有可能的评估结果文件
    POSSIBLE_FILES=$(find outputs -name "bump_hunt_results_seed${SEED}.pkl" -type f 2>/dev/null | head -1)
    
    if [ -n "${POSSIBLE_FILES}" ]; then
        EVAL_RESULTS="${POSSIBLE_FILES}"
        echo "Found evaluation results: ${EVAL_RESULTS}"
    else
        echo "Error: Could not find evaluation results file."
        echo "Please specify the path manually by setting EVAL_RESULTS variable."
        echo ""
        echo "Example:"
        echo "  EVAL_RESULTS=\"outputs/upsilon_iso_12_21/evaluation/bump_hunt_results_seed42.pkl\""
        echo ""
        echo "Note: Path is based on analysis_keywords.name, not dataset_id"
        exit 1
    fi
fi

# 检查文件是否存在
if [ ! -f "${EVAL_RESULTS}" ]; then
    echo "Error: Evaluation results file not found: ${EVAL_RESULTS}"
    exit 1
fi

echo "Using evaluation results: ${EVAL_RESULTS}"

###### 生成所有论文图 ######

echo "=========================================="
echo "Generating all paper figures..."
echo "=========================================="

# 构建命令
PLOT_CMD="python scripts/generate_figures.py"
PLOT_CMD="${PLOT_CMD} --config ${CONFIG_FILE}"
PLOT_CMD="${PLOT_CMD} --eval-results ${EVAL_RESULTS}"
PLOT_CMD="${PLOT_CMD} --fit-type ${FIT_TYPE}"
PLOT_CMD="${PLOT_CMD} --num-bins-SR ${NUM_BINS_SR}"
PLOT_CMD="${PLOT_CMD} --seed ${SEED}"

if [ "${INCLUDE_TRAINING_PLOTS}" = "true" ]; then
    PLOT_CMD="${PLOT_CMD} --include-training-plots"
fi

if [ -n "${OUTPUT_DIR}" ]; then
    PLOT_CMD="${PLOT_CMD} --output-dir ${OUTPUT_DIR}"
fi

echo "Command: ${PLOT_CMD}"
echo ""

# 执行画图
${PLOT_CMD}

echo ""
echo "=========================================="
echo "Figure generation completed!"
echo "=========================================="
echo "Figures saved to:"
if [ -n "${OUTPUT_DIR}" ]; then
    echo "  ${OUTPUT_DIR}"
else
    echo "  outputs/*/figures/"
fi
echo ""
echo "Generated figures:"
echo "  - histogram_cuts.pdf (Cut histograms)"
echo "  - features.pdf (Feature distributions)"
echo "  - significance.pdf (Significance vs FPR)"
echo "  - training_losses.pdf (Training/validation losses)"
echo "  - roc_curve.pdf (ROC curve)"
echo "  - significance_variations.pdf (Variations plot, if --all-variations was used)"

