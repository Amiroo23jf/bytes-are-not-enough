#!/bin/bash
#SBATCH --job-name=txt-prepare
#SBATCH --output=logs/txt-prepare-%j.out
#SBATCH --error=logs/txt-prepare-%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

# Prepares data for both enwik8 configs (bytes + tokens).
# Bytes mode: trivially fast (reads file, writes npy).
# Tokens mode: trains ByteBPE on enwik8 (~10-15 min) and encodes the full file.
# Must complete before the seed training jobs are submitted.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs/text_transformer

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate trans-compress-gpu
export MKL_THREADING_LAYER=GNU
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_LOG="outputs/text_transformer/prepare-${SLURM_JOB_ID}.log"

srun python3 -u scripts/run_text_transformer.py \
    configs/text_transformer/bytes_enwik8.yaml \
    configs/text_transformer/tokens_enwik8.yaml \
    --prepare-only \
    2>&1 | tee "${RUN_LOG}"

echo "Prepare finished at: $(date)"
