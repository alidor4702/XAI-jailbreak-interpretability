#!/bin/bash
#SBATCH --job-name=xai_harmbench
#SBATCH --output=outputs/harmbench_%j.out
#SBATCH --error=outputs/harmbench_%j.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_prod_long
#SBATCH --exclude=sh07,sh[10-19]
#SBATCH --propagate=NONE
#SBATCH --mail-user=ali.dor@student-cs.fr
#SBATCH --mail-type=END,FAIL

# HarmBench validation: judge all 600 fuzzer candidates
# Model: cais/HarmBench-Llama-2-13b-cls (13B, 8-bit ~13GB VRAM)
# Expected runtime: ~1-2h for 600 candidates

set -euo pipefail

PROJECT_DIR="/usr/users/rl_for_wsi/dor_ali/projects/XAI"
cd "$PROJECT_DIR"

echo "=== HARMBENCH VALIDATE $SLURM_JOB_ID on $(hostname) ==="
echo "Start time: $(date)"
nvidia-smi

export PATH="$PROJECT_DIR/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HOME="/usr/users/rl_for_wsi/dor_ali/.cache/huggingface"
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Validate the first fuzzer run
FUZZER_RUN="outputs/fuzzer_170260"

python3 -u -m scripts.harmbench_validate "$FUZZER_RUN"

echo "=== HARMBENCH VALIDATE $SLURM_JOB_ID finished ==="
echo "End time: $(date)"
