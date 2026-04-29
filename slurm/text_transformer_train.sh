#!/bin/bash
#SBATCH --job-name=txt-train
#SBATCH --output=logs/txt-train-%j.out
#SBATCH --error=logs/txt-train-%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00

# Trains a single (CONFIG, SEED) pair.  Expects two env vars:
#   CONFIG  path to the YAML config  (default: bytes_enwik8)
#   SEED    integer seed             (default: 0)
#
# Examples:
#   CONFIG=configs/text_transformer/bytes_enwik8.yaml  SEED=0 sbatch slurm/text_transformer_train.sh
#   CONFIG=configs/text_transformer/tokens_enwik8.yaml SEED=1 sbatch slurm/text_transformer_train.sh
#
# Data must already be prepared (run text_transformer_prepare.sh first).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs/text_transformer

CONFIG="${CONFIG:-configs/text_transformer/bytes_enwik8.yaml}"
SEED="${SEED:-0}"

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Working directory: $(pwd)"
echo "CONFIG=${CONFIG}  SEED=${SEED}"
echo "Start time: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate trans-compress-gpu
export MKL_THREADING_LAYER=GNU
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Derive a log name from config stem + seed
CFG_STEM=$(basename "${CONFIG}" .yaml)
RUN_LOG="outputs/text_transformer/train-${CFG_STEM}-seed${SEED}-${SLURM_JOB_ID}.log"

srun python3 -u scripts/run_text_transformer.py \
    "${CONFIG}" \
    --seeds "${SEED}" \
    --skip-prepare \
    --no-plot \
    --device cuda \
    2>&1 | tee "${RUN_LOG}"

echo "Training finished at: $(date)"
