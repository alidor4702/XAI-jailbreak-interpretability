#!/bin/bash
# Run smoke test directly via SSH (when SLURM nodes are allocated by others)
# Usage: bash dce/run_smoke_ssh.sh [node]  (default: sh22)

set -euo pipefail

NODE="${1:-sh22}"
PROJECT_DIR="/usr/users/rl_for_wsi/dor_ali/projects/XAI"

echo "Running smoke test on $NODE..."

ssh "$NODE" bash << 'REMOTE_SCRIPT'
set -euo pipefail
cd /usr/users/rl_for_wsi/dor_ali/projects/XAI
export PATH="/usr/users/rl_for_wsi/dor_ali/projects/XAI/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HOME="/usr/users/rl_for_wsi/dor_ali/.cache/huggingface"
export TORCHDYNAMO_DISABLE=1
python3 -u -m scripts.smoke_test
REMOTE_SCRIPT
