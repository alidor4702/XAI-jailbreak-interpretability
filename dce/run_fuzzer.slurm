#!/bin/bash
#SBATCH --job-name=xai_fuzzer
#SBATCH --output=outputs/fuzzer_%j.out
#SBATCH --error=outputs/fuzzer_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_prod_long
#SBATCH --exclude=sh07,sh[10-19]
#SBATCH --propagate=NONE
#SBATCH --mail-user=ali.dor@student-cs.fr
#SBATCH --mail-type=END,FAIL

# Full fuzzer replication: 10 pop x 10 gen (same as Elora's experiment_02)
# Expected runtime: ~7h total (~3.5h per category)

set -euo pipefail

PROJECT_DIR="/usr/users/rl_for_wsi/dor_ali/projects/XAI"
cd "$PROJECT_DIR"

RUN_ID="fuzzer_${SLURM_JOB_ID}"
OUTPUT_DIR="outputs/${RUN_ID}"
mkdir -p "$OUTPUT_DIR"

echo "=== FUZZER $SLURM_JOB_ID on $(hostname) ==="
echo "Start time: $(date)"
nvidia-smi

export PATH="$PROJECT_DIR/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HOME="/usr/users/rl_for_wsi/dor_ali/.cache/huggingface"
export TORCHDYNAMO_DISABLE=1

python3 -u -m src.fuzzer.run \
    --output-dir "$OUTPUT_DIR" \
    --population-size 10 \
    --generations 10 \
    --mutation-rate 0.3 \
    --max-new-tokens 256

echo "=== FUZZER $SLURM_JOB_ID finished ==="
echo "End time: $(date)"
