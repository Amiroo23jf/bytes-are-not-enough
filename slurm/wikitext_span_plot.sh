#!/bin/bash
#SBATCH --job-name=wts-plot
#SBATCH --output=logs/wts-plot-%j.out
#SBATCH --error=logs/wts-plot-%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --time=00:15:00

# Aggregates all per-tokenizer results for one experiment config into its
# summary figure + JSON. Must run after all that config's train jobs have
# completed. No GPU needed.
#
# Expects:
#   CONFIG   path to the YAML config (required)
#
# Example:
#   CONFIG=configs/wikitext_span_transformer/exp1.yaml sbatch slurm/wikitext_span_plot.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs/wikitext_span_transformer

: "${CONFIG:?Set CONFIG=<path> before submitting}"

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Working directory: $(pwd)"
echo "CONFIG=${CONFIG}"
echo "Start time: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bane
export MKL_THREADING_LAYER=GNU

CFG_STEM=$(basename "${CONFIG}" .yaml)
RUN_LOG="outputs/wikitext_span_transformer/plot-${CFG_STEM}-${SLURM_JOB_ID}.log"

srun python3 -u scripts/run_wikitext_span_transformer.py \
    "${CONFIG}" --stage plot \
    2>&1 | tee "${RUN_LOG}"

echo "Plot finished at: $(date)"
echo "Figures saved under outputs/wikitext_span_transformer/figures/"
