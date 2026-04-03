#!/bin/bash
# One-time setup for DCE: create venv and install dependencies.
# Run this from the project root: bash dce/setup_dce.sh
# (Already run — kept here for documentation/reproducibility)

set -euo pipefail

PROJECT_DIR="/usr/users/rl_for_wsi/dor_ali/projects/XAI"
cd "$PROJECT_DIR"

echo "=== Setting up XAI project on DCE ==="

# Need Python 3.11 from modules (system python3.10 lacks venv)
module load python/3.11.2/gcc-13.1.0

python3 -m venv .venv
.venv/bin/pip install --upgrade pip

# Unsloth pulls its own torch (cu128), compatible with DCE driver 550/CUDA 12.4
.venv/bin/pip install unsloth transformers accelerate bitsandbytes
.venv/bin/pip install nltk captum nnsight einops
.venv/bin/pip install datasets huggingface-hub
.venv/bin/pip install scipy scikit-learn
.venv/bin/pip install jupyter matplotlib seaborn pandas tqdm

# NLTK WordNet data (needed for synonym_swap mutation)
.venv/bin/python3 -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# HuggingFace cache
export HF_HOME="/usr/users/rl_for_wsi/dor_ali/.cache/huggingface"
mkdir -p "$HF_HOME"

echo ""
echo "=== Setup complete ==="
echo "Run smoke test: sbatch dce/smoke_test.slurm"
