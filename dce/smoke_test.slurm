#!/bin/bash
#SBATCH --job-name=xai_smoke_test
#SBATCH --output=outputs/smoke_test_%j.out
#SBATCH --error=outputs/smoke_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_prod_long
#SBATCH --exclude=sh07,sh[10-19]
#SBATCH --propagate=NONE
#SBATCH --mail-user=ali.dor@student-cs.fr
#SBATCH --mail-type=END,FAIL

# Smoke test: model loading + single fuzzer eval + nnsight tracing
# Expected runtime: ~10-15 min (model loading ~5min + evals + tracing)

set -euo pipefail

PROJECT_DIR="/usr/users/rl_for_wsi/dor_ali/projects/XAI"
cd "$PROJECT_DIR"

mkdir -p outputs/

echo "=== SMOKE TEST $SLURM_JOB_ID on $(hostname) ==="
echo "Start time: $(date)"
nvidia-smi

export PATH="$PROJECT_DIR/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HOME="/usr/users/rl_for_wsi/dor_ali/.cache/huggingface"
export TORCHDYNAMO_DISABLE=1

python3 -u -m scripts.smoke_test 2>&1

echo "=== SMOKE TEST $SLURM_JOB_ID finished ==="
echo "End time: $(date)"
