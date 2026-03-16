#!/usr/bin/env bash
# =============================================================================
# batch_make_videos.sh  –  SLURM parallel worker (single job, N CPUs)
# -----------------------------------------------------------------------------
# Convert every .czi file in a folder to annotated MP4s.
# All files are processed in parallel using --cpus-per-task worker processes.
# No need to count files or pass --array.
#
# Submission
# ----------
#   sbatch batch_make_videos.sh /path/to/folder
#
#   # Override CPUs (= max parallelism):
#   sbatch --cpus-per-task=16 batch_make_videos.sh /path/to/folder
#
#   # Forward extra options to make_video_from_czi.py:
#   sbatch batch_make_videos.sh /path/to/folder --fps 15 --resize 2048
#
# Default SLURM resources (override on the sbatch command line)
# --------------------------------------------------------------
#SBATCH --job-name=make_videos
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


set -euo pipefail

# ---------------------------------------------------------------------------
# Environment activation (edit as needed, or export before calling sbatch)
# ---------------------------------------------------------------------------
module load Python/3.12.8-GCCcore-12.2.0
source /home/bisot/Documents/fungal_growth_model/.venv/bin/activate


CZI_FOLDER="${1%/}"
shift
EXTRA_ARGS="$*"   # remaining args forwarded verbatim to make_video_from_czi.py

if [[ ! -d "${CZI_FOLDER}" ]]; then
    echo "ERROR: '${CZI_FOLDER}' is not a directory." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Locate make_video_from_czi.py (same directory as this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/make_video_from_czi.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: make_video_from_czi.py not found at '${PYTHON_SCRIPT}'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build sorted file list
# ---------------------------------------------------------------------------
mapfile -d '' ALL_FILES < <(find "${CZI_FOLDER}" -maxdepth 1 -name "*.czi" -print0 | sort -z)
N_FILES=${#ALL_FILES[@]}

if [[ ${N_FILES} -eq 0 ]]; then
    echo "No .czi files found in '${CZI_FOLDER}'." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Parallelism: up to $SLURM_CPUS_PER_TASK concurrent workers
# ---------------------------------------------------------------------------
N_WORKERS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Processing ${N_FILES} file(s) with ${N_WORKERS} parallel worker(s)."

job_count=0
for CZI in "${ALL_FILES[@]}"; do
    echo "[$(date +%T)] Starting: ${CZI}"
    # shellcheck disable=SC2086
    python "${PYTHON_SCRIPT}" "${CZI}" ${EXTRA_ARGS} &

    (( job_count++ ))
    # Once we have N_WORKERS jobs running, wait for one to finish before launching more
    if (( job_count >= N_WORKERS )); then
        wait -n 2>/dev/null || wait   # 'wait -n' needs bash ≥ 4.3
        (( job_count-- ))
    fi
done

# Wait for all remaining workers to finish
wait
echo "All ${N_FILES} file(s) complete."
