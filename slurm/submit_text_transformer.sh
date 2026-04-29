#!/bin/bash
# Submit the full text_transformer enwik8 experiment as parallel SLURM jobs.
#
# Job graph:
#   [prepare]
#       ├── [bytes seed 0]  ──┐
#       ├── [bytes seed 1]    │
#       ├── [bytes seed 2]    ├── [plot]
#       ├── [tokens seed 0]   │
#       ├── [tokens seed 1]   │
#       └── [tokens seed 2] ──┘
#
# Usage:
#   cd /path/to/bytes-are-not-enough
#   bash slurm/submit_text_transformer.sh

set -euo pipefail

BYTES_CFG="configs/text_transformer/bytes_enwik8.yaml"
TOKENS_CFG="configs/text_transformer/tokens_enwik8.yaml"

mkdir -p logs

# ── Step 1: prepare both configs (runs once) ──────────────────────────────
PREP_JID=$(sbatch --parsable slurm/text_transformer_prepare.sh)
echo "Submitted prepare job: ${PREP_JID}"

# ── Step 2: train each (config, seed) in parallel ─────────────────────────
# All training jobs depend on prepare completing successfully.

B0=$(CONFIG="${BYTES_CFG}"  SEED=0 sbatch --parsable \
        --dependency=afterok:"${PREP_JID}" \
        --job-name=txt-bytes-s0 \
        slurm/text_transformer_train.sh)
echo "Submitted bytes seed=0: ${B0}"

B1=$(CONFIG="${BYTES_CFG}"  SEED=1 sbatch --parsable \
        --dependency=afterok:"${PREP_JID}" \
        --job-name=txt-bytes-s1 \
        slurm/text_transformer_train.sh)
echo "Submitted bytes seed=1: ${B1}"

B2=$(CONFIG="${BYTES_CFG}"  SEED=2 sbatch --parsable \
        --dependency=afterok:"${PREP_JID}" \
        --job-name=txt-bytes-s2 \
        slurm/text_transformer_train.sh)
echo "Submitted bytes seed=2: ${B2}"

T0=$(CONFIG="${TOKENS_CFG}" SEED=0 sbatch --parsable \
        --dependency=afterok:"${PREP_JID}" \
        --job-name=txt-tokens-s0 \
        slurm/text_transformer_train.sh)
echo "Submitted tokens seed=0: ${T0}"

T1=$(CONFIG="${TOKENS_CFG}" SEED=1 sbatch --parsable \
        --dependency=afterok:"${PREP_JID}" \
        --job-name=txt-tokens-s1 \
        slurm/text_transformer_train.sh)
echo "Submitted tokens seed=1: ${T1}"

T2=$(CONFIG="${TOKENS_CFG}" SEED=2 sbatch --parsable \
        --dependency=afterok:"${PREP_JID}" \
        --job-name=txt-tokens-s2 \
        slurm/text_transformer_train.sh)
echo "Submitted tokens seed=2: ${T2}"

# ── Step 3: plot after all training is done ───────────────────────────────
PLOT_JID=$(sbatch --parsable \
    --dependency=afterok:"${B0}:${B1}:${B2}:${T0}:${T1}:${T2}" \
    slurm/text_transformer_plot.sh)
echo "Submitted plot job:    ${PLOT_JID}"

echo ""
echo "All jobs submitted. Dependency chain:"
echo "  prepare (${PREP_JID})"
echo "    → bytes  s0 (${B0}), s1 (${B1}), s2 (${B2})"
echo "    → tokens s0 (${T0}), s1 (${T1}), s2 (${T2})"
echo "      → plot (${PLOT_JID})"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs under:    logs/  and  outputs/text_transformer/"
