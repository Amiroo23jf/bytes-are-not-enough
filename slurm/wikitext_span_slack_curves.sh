#!/bin/bash
#SBATCH --job-name=wts-slack
#SBATCH --output=logs/wts-slack-%j.out
#SBATCH --error=logs/wts-slack-%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

# Computes and plots the per-tokenizer slack curves (see
# scripts/plot_wikitext_span_slack_curves.py). No GPU needed -- reads the
# full char_lens_train.npy arrays already produced by the 'prepare' stage.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs/wikitext_span_transformer

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Start time: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bane
export MKL_THREADING_LAYER=GNU

RUN_LOG="outputs/wikitext_span_transformer/slack-curves-${SLURM_JOB_ID}.log"
EXTRA_ARGS="${EXTRA_ARGS:-}"

srun python3 -u scripts/plot_wikitext_span_slack_curves.py ${EXTRA_ARGS} 2>&1 | tee "${RUN_LOG}"

echo "Finished at: $(date)"
