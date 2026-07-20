#!/bin/bash

#SBATCH -J collect_260713
#SBATCH -p high
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

set -euo pipefail

echo "[INFO] Starting result collection on $(hostname) at $(date)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID}"

cd /home/afana01t/biohack2024

poetry install --no-root

export PYTHONPATH=/home/afana01t/biohack2024/src

poetry run python scripts/optuna_benchmark.py \
  --models bert biobert scibert pubmedbert deberta modernbert bioformer \
  --study-dir results/optuna_260713 \
  --collect-test-results

echo "[INFO] Finished at $(date)"
