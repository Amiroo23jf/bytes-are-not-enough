#!/bin/bash
# Submit the full wikitext_span_transformer experiment as parallel SLURM jobs.
#
# Job graph (N = number of tokenizers in the config, currently 9):
#   [prepare tok_1] ... [prepare tok_N]      (parallel, CPU only, run once --
#                                              shared by both experiments)
#         |                    |
#         v                    v
#   [train exp1 tok_1] .. [train exp1 tok_N] --> [plot exp1]
#   [train exp2 tok_1] .. [train exp2 tok_N] --> [plot exp2]
#
# Each train job depends only on its own tokenizer's prepare job
# (--dependency=afterok), so tokenizers finish independently and don't wait
# on each other. `char` under exp2 will resolve to the same w as under exp1
# and its train job will just reuse the cached result almost instantly.
#
# Usage:
#   cd /path/to/bytes-are-not-enough
#   bash slurm/submit_wikitext_span_transformer.sh

set -euo pipefail

EXP1_CFG="configs/wikitext_span_transformer/exp1.yaml"
EXP2_CFG="configs/wikitext_span_transformer/exp2.yaml"

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bane

# Tokenizer list is identical between exp1/exp2 (see exp2.yaml's comment);
# read it once from exp1.yaml.
mapfile -t TOKENIZERS < <(python3 -c "
import yaml
cfg = yaml.safe_load(open('${EXP1_CFG}'))
for t in cfg['tokenizers']:
    print(t['name'])
")
echo "Tokenizers: ${TOKENIZERS[*]}"

# ── Step 1: one prepare job per tokenizer (parallel, CPU only) ────────────
declare -A PREP_JID
for TOK in "${TOKENIZERS[@]}"; do
    JID=$(CONFIG="${EXP1_CFG}" TOKENIZER="${TOK}" sbatch --parsable \
            --job-name="wts-prep-${TOK}" \
            slurm/wikitext_span_prepare.sh)
    PREP_JID[$TOK]="${JID}"
    echo "Submitted prepare(${TOK}): ${JID}"
done

# ── Step 2: two train jobs per tokenizer (one per experiment), each ───────
#            depending only on that tokenizer's own prepare job.
declare -A EXP1_TRAIN_JIDS EXP2_TRAIN_JIDS
for TOK in "${TOKENIZERS[@]}"; do
    JID1=$(CONFIG="${EXP1_CFG}" TOKENIZER="${TOK}" sbatch --parsable \
            --dependency=afterok:"${PREP_JID[$TOK]}" \
            --job-name="wts-train-exp1-${TOK}" \
            slurm/wikitext_span_train.sh)
    EXP1_TRAIN_JIDS[$TOK]="${JID1}"
    echo "Submitted train(exp1, ${TOK}): ${JID1}"

    JID2=$(CONFIG="${EXP2_CFG}" TOKENIZER="${TOK}" sbatch --parsable \
            --dependency=afterok:"${PREP_JID[$TOK]}" \
            --job-name="wts-train-exp2-${TOK}" \
            slurm/wikitext_span_train.sh)
    EXP2_TRAIN_JIDS[$TOK]="${JID2}"
    echo "Submitted train(exp2, ${TOK}): ${JID2}"
done

# ── Step 3: one plot job per experiment, after all its train jobs ────────
join_by() { local IFS="$1"; shift; echo "$*"; }

EXP1_DEP=$(join_by : "${EXP1_TRAIN_JIDS[@]}")
EXP2_DEP=$(join_by : "${EXP2_TRAIN_JIDS[@]}")

PLOT1_JID=$(CONFIG="${EXP1_CFG}" sbatch --parsable \
        --dependency=afterok:"${EXP1_DEP}" \
        --job-name="wts-plot-exp1" \
        slurm/wikitext_span_plot.sh)
echo "Submitted plot(exp1): ${PLOT1_JID}"

PLOT2_JID=$(CONFIG="${EXP2_CFG}" sbatch --parsable \
        --dependency=afterok:"${EXP2_DEP}" \
        --job-name="wts-plot-exp2" \
        slurm/wikitext_span_plot.sh)
echo "Submitted plot(exp2): ${PLOT2_JID}"

echo ""
echo "All jobs submitted: $((${#TOKENIZERS[@]} * 3 + 2)) total"
echo "  ${#TOKENIZERS[@]} prepare + $((${#TOKENIZERS[@]} * 2)) train + 2 plot"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs under:   logs/  and  outputs/wikitext_span_transformer/"
