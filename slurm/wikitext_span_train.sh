#!/bin/bash
#SBATCH --job-name=wts-train
#SBATCH --output=logs/wts-train-%j.out
#SBATCH --error=logs/wts-train-%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=02:00:00

# Trains + evaluates one (CONFIG, TOKENIZER) pair: resolves the context
# window w per the config's `window` rule (fixed_w for Exp 1, fixed_w_s for
# Exp 2), trains a TransformerLM on the train split, evaluates held-out loss
# on the test split. Requires the tokenizer's prepare job to have completed
# first (see submit_wikitext_span_transformer.sh for the dependency graph).
#
# If this (TOKENIZER, resolved-w) pair was already trained under a
# different experiment (this always happens for `char`, whose 1-char/token
# ratio resolves to w=128 in both experiments), the cached result is reused
# and this job finishes immediately.
#
# Expects:
#   CONFIG      path to the YAML config (required)
#   TOKENIZER   tokenizer name from the config's `tokenizers` list (required)
#
# Example:
#   CONFIG=configs/wikitext_span_transformer/exp1.yaml TOKENIZER=bpe_32k \
#       sbatch slurm/wikitext_span_train.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs/wikitext_span_transformer

: "${CONFIG:?Set CONFIG=<path> before submitting}"
: "${TOKENIZER:?Set TOKENIZER=<name> before submitting}"

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Working directory: $(pwd)"
echo "CONFIG=${CONFIG}  TOKENIZER=${TOKENIZER}"
echo "Start time: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bane
export MKL_THREADING_LAYER=GNU
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

CFG_STEM=$(basename "${CONFIG}" .yaml)
RUN_LOG="outputs/wikitext_span_transformer/train-${CFG_STEM}-${TOKENIZER}-${SLURM_JOB_ID}.log"

srun python3 -u scripts/run_wikitext_span_transformer.py \
    "${CONFIG}" --stage train --tokenizer "${TOKENIZER}" --device cuda \
    2>&1 | tee "${RUN_LOG}"

echo "Train finished at: $(date)"
