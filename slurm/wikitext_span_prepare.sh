#!/bin/bash
#SBATCH --job-name=wts-prepare
#SBATCH --output=logs/wts-prepare-%j.out
#SBATCH --error=logs/wts-prepare-%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=02:00:00

# Trains (or loads a cached copy of) one tokenizer on the WikiText-103 train
# split, encodes train/validation/test with it, and persists everything
# under outputs/wikitext_span_transformer/data/<TOKENIZER>/. No GPU needed.
#
# Data is keyed by tokenizer name only (not by experiment), so this only
# needs to run once per tokenizer even though both exp1.yaml and exp2.yaml
# use it -- point CONFIG at either one, the tokenizer list is identical.
#
# Expects:
#   TOKENIZER   tokenizer name from the config's `tokenizers` list (required)
#   CONFIG      path to the YAML config (default: exp1.yaml)
#
# Example:
#   TOKENIZER=bpe_32k sbatch slurm/wikitext_span_prepare.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs/wikitext_span_transformer

CONFIG="${CONFIG:-configs/wikitext_span_transformer/exp1.yaml}"
: "${TOKENIZER:?Set TOKENIZER=<name> before submitting}"

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Working directory: $(pwd)"
echo "CONFIG=${CONFIG}  TOKENIZER=${TOKENIZER}"
echo "Start time: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bane
export MKL_THREADING_LAYER=GNU
export TOKENIZERS_PARALLELISM=false

RUN_LOG="outputs/wikitext_span_transformer/prepare-${TOKENIZER}-${SLURM_JOB_ID}.log"

srun python3 -u scripts/run_wikitext_span_transformer.py \
    "${CONFIG}" --stage prepare --tokenizer "${TOKENIZER}" \
    2>&1 | tee "${RUN_LOG}"

echo "Prepare finished at: $(date)"
