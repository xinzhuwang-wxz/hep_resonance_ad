#!/bin/bash

###### SLURM Configuration ######

#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=higgsgpu
#SBATCH --job-name=evaluate_cathode
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --output=./logs/evaluate_%j.log
#SBATCH --error=./logs/evaluate_%j.err

# Create logs directory if it doesn't exist
mkdir -p ./logs

###### Environment Setup ######

# Load modules if needed (uncomment and modify as necessary)
# module load cuda/11.8
# module load python/3.10

# Activate conda environment
source /hpcfs/cepc/higgsgpu/wangxinzhu/miniconda3/bin/activate
conda activate strange

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# PyTorch specific optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

###### Job Execution ######

# Change to working directory
cd /hpcfs/cepc/higgsgpu/wangxinzhu/hep_resonance_ad

# Print job information
echo "Job started at $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Python environment: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "CUDA device count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "CUDA device name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# Run evaluation script
python scripts/evaluate.py \
    --config configs/upsilon_reproduction.yaml \
    --model-dir outputs/training_run_20241221_120000 \
    --seed 42 \
    --all-variations \
    --output-dir outputs/evaluation_run_$(date +%Y%m%d_%H%M%S)

echo "Job finished at $(date)"
