#!/bin/bash

###### SLURM Configuration ######

#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=higgsgpu
#SBATCH --job-name=generate_figures
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=06:00:00
#SBATCH --output=./logs/generate_figures_%j.log
#SBATCH --error=./logs/generate_figures_%j.err

# Create logs directory if it doesn't exist
mkdir -p ./logs

###### Environment Setup ######

# Load modules if needed (uncomment and modify as necessary)
# module load cuda/11.8
# module load python/3.10

# Activate conda environment
source /hpcfs/cepc/higgsgpu/wangxinzhu/miniconda3/bin/activate
conda activate strange

# Set environment variables
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

# Run figure generation script
python scripts/generate_figures.py \
    --config configs/upsilon_reproduction.yaml \
    --evaluation-results outputs/evaluation_run_20241221_120000/evaluation/bump_hunt_results_seed42.pkl \
    --fit-type quintic \
    --num-bins-SR 8 \
    --include-training-plots \
    --seed 42 \
    --output-dir outputs/figures_run_$(date +%Y%m%d_%H%M%S)

echo "Job finished at $(date)"
