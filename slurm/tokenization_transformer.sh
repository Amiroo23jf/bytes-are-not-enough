#!/bin/bash
#SBATCH --job-name=tokenization-transformer
#SBATCH --output=logs/tokenization-transformer-%j.out
#SBATCH --error=logs/tokenization-transformer-%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=04:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs/tokenization_transformer

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bane
echo "Environment activated"

export MKL_THREADING_LAYER=GNU
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Optional:
#   export CONFIG_ARG="configs/tokenization_transformer/A_greedy.yaml"
#   export CONFIG_ARG="--smoke"
CONFIG_ARG="${CONFIG_ARG:-}"

RUN_LOG="outputs/tokenization_transformer/run-${SLURM_JOB_ID}.log"
echo "Writing tee log to ${RUN_LOG}"

if [ -n "${CONFIG_ARG}" ]; then
  srun python3 -u scripts/run_tokenization_transformer.py ${CONFIG_ARG} 2>&1 | tee "${RUN_LOG}"
else
  srun python3 -u scripts/run_tokenization_transformer.py 2>&1 | tee "${RUN_LOG}"
fi

echo "Job finished at: $(date)"
