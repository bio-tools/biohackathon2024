#!/bin/bash
# Submit with:
# sbatch scripts/run_optuna_260713.sh

#SBATCH -J optuna_260713
#SBATCH -p high
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-6
#SBATCH -o logs/slurm-%A_%a.out
#SBATCH -e logs/slurm-%A_%a.err

set -euo pipefail

echo "[INFO] Starting tool-disjoint Optuna benchmark on $(hostname) at $(date)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[INFO] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cd /home/afana01t/biohack2024

mkdir -p \
  data/IOB_260713 \
  logs/benchmark_260713 \
  logs/optuna_260713 \
  models/benchmark_260713 \
  models/optuna_260713 \
  results/optuna_260713 \
  wandb

poetry install --no-root

export PYTHONPATH=/home/afana01t/biohack2024/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/home/afana01t/biohack2024/wandb
export WANDB_PROJECT=biohack2024-optuna-260713

models=(
  bert
  biobert
  scibert
  pubmedbert
  deberta
  modernbert
  bioformer
)
model="${models[${SLURM_ARRAY_TASK_ID}]}"

for path in \
  data/IOB_260713/train_IOB.tsv \
  data/IOB_260713/val_IOB.tsv \
  data/IOB_260713/test_IOB.tsv; do
  if [[ ! -s "${path}" ]]; then
    echo "[ERROR] Missing prepared dataset file: ${path}" >&2
    exit 1
  fi
done

echo "[INFO] Running model=${model}"
poetry run python scripts/optuna_benchmark.py \
  --models "${model}" \
  --data-dir data/IOB_260713 \
  --run-id 260713 \
  --study-dir "results/optuna_260713/${model}"

echo "[INFO] Finished at $(date)"
