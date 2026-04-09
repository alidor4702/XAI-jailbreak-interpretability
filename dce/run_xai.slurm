#!/bin/bash
#SBATCH --job-name=xai_analysis
#SBATCH --output=outputs/xai_%j.out
#SBATCH --error=outputs/xai_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_prod_long
#SBATCH --exclude=sh00,sh07,sh[10-19]
#SBATCH --propagate=NONE

set -euo pipefail

echo "=== XAI ANALYSIS $SLURM_JOB_ID on $(hostname) ==="
echo "Start time: $(date)"
nvidia-smi

cd /usr/users/rl_for_wsi/dor_ali/projects/XAI
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m scripts.run_xai_analysis outputs/fuzzer_170260 --n-per-seed 2 --out-subdir xai_analysis_2

echo "=== XAI ANALYSIS $SLURM_JOB_ID finished ==="
echo "End time: $(date)"
