#!/bin/bash
# Submit like this:
# sbatch scripts/run_optuna_benchmark.sh
#
# Optional smoke run:
# sbatch scripts/run_optuna_benchmark.sh --models modernbert --trials 1 --final-seeds 42

#SBATCH -J optuna_260507
#SBATCH -p high
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

echo "[INFO] Starting Optuna benchmark on $(hostname) at $(date)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cd /home/afana01t/biohack2024

mkdir -p \
  logs \
  logs/benchmark_260507 \
  logs/optuna_260507 \
  models/benchmark_260507 \
  models/optuna_260507 \
  results/optuna_260507 \
  wandb

poetry install --no-root

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/afana01t/biohack2024/src
export WANDB_DIR=/home/afana01t/biohack2024/wandb
export WANDB_PROJECT=biohack2024-optuna-260507

poetry run python scripts/optuna_benchmark.py "$@"

echo "[INFO] Finished at $(date)"
