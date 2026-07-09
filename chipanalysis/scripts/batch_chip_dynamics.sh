#!/usr/bin/env bash
# =============================================================================
# batch_chip_dynamics.sh  –  SLURM job-array launcher for batch_chip_dynamics.py
# =============================================================================
#
# Each array task processes ONE CZI scene from the Excel manifest so that SLURM
# can balance multi-scene CZI files across jobs.
#
# Quick-start
# -----------
# Step 1 – find how many scene tasks are in your manifest (sets the array upper bound):
#
#   N=$(python /path/to/batch_chip_dynamics.py \
#         --excel   /path/to/manifest.xlsx   \
#         --czi-root /scratch/bisot/ZeisData \
#         --config  /path/to/config.yaml     \
#         --output  /tmp                     \
#         --count-tasks)
#
# Step 2 – submit (limit concurrency with  %K, e.g. --array=0-${N}%4):
#
#   sbatch --array=0-${N} batch_chip_dynamics.sh \
#       --excel  /path/to/manifest.xlsx       \
#       --czi-root /scratch/bisot/ZeisData    \
#       --config /path/to/config.yaml         \
#       --output /path/to/results
#
# Skip video export (faster, data-only):
#
#   sbatch --array=0-${N} batch_chip_dynamics.sh \
#       --excel  manifest.xlsx --config config.yaml \
#       --czi-root /scratch/bisot/ZeisData \
#       --output results --skip-video
#
# Default SLURM resource requests
# --------------------------------
# Override any of these on the sbatch command line, e.g.:
#   sbatch --mem=64G --time=24:00:00 --array=... batch_chip_dynamics.sh ...
#
#SBATCH --job-name=chip_dynamics
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/chip_dynamics_%A_%a.out
#SBATCH --error=logs/chip_dynamics_%A_%a.err

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
# Adjust module and venv path for your cluster setup.
module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null || true
module load FFmpeg/7.0.2-GCCcore-13.3.0

source /home/bisot/Documents/chipanalysis/.venv/bin/activate

# ── Script location (resolved relative to this .sh file) ─────────────────────
PYTHON_SCRIPT="${SLURM_SUBMIT_DIR}/chipanalysis/scripts/batch_chip_dynamics.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "[ERROR] batch_chip_dynamics.py not found at ${PYTHON_SCRIPT}" >&2
    exit 1
fi

# ── Create log directory ──────────────────────────────────────────────────────
mkdir -p logs

# ── Run ──────────────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')]  Array task ${SLURM_ARRAY_TASK_ID} starting on $(hostname)"

python "${PYTHON_SCRIPT}" \
    --task "${SLURM_ARRAY_TASK_ID}" \
    "$@"

echo "[$(date '+%Y-%m-%d %H:%M:%S')]  Array task ${SLURM_ARRAY_TASK_ID} done."
