#!/usr/bin/env bash
# =============================================================================
# batch_region_fluorescence.sh  –  SLURM parallel worker (single job, N CPUs)
# -----------------------------------------------------------------------------
# Extract fluorescence metrics from every .czi file in a folder.
# All files are processed in parallel using --cpus-per-task worker processes.
#
# Submission
# ----------
#   sbatch batch_region_fluorescence.sh /path/to/folder
#
#   # Override CPUs (= max parallelism):
#   sbatch --cpus-per-task=16 batch_region_fluorescence.sh /path/to/folder
#
#   # Forward extra options to analyze_region_fluorescence.py:
#   sbatch batch_region_fluorescence.sh /path/to/folder --metric both --channels 0,1
#
# Default SLURM resources (override on the sbatch command line)
# --------------------------------------------------------------
#SBATCH --job-name=analyze_fluorescence
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fluorescence_%j.out
#SBATCH --error=logs/fluorescence_%j.err


set -uo pipefail  # Remove 'e' so failed background jobs don't kill the script

# ---------------------------------------------------------------------------
# Environment activation (edit as needed, or export before calling sbatch)
# ---------------------------------------------------------------------------
module load Python/3.12.3-GCCcore-13.3.0
source /home/bisot/Documents/chipanalysis/.venv/bin/activate


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
PYTHON_SCRIPT="/home/bisot/Documents/chipanalysis/chipanalysis/scripts/analyze_region_fluorescence.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: analyze_region_fluorescence.py not found at '${PYTHON_SCRIPT}'" >&2
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
# Two-level parallelism:
#   FILE_WORKERS  – how many CZI files run concurrently (bash level)
#   T_WORKERS     – how many timestep threads each Python process spawns
#
# Rule: FILE_WORKERS × T_WORKERS ≤ N_WORKERS
#   • Many files  → favour file-level concurrency, 1 thread per process
#   • Few files   → each process gets more threads for timestep parallelism
# ---------------------------------------------------------------------------
N_WORKERS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

if (( N_FILES >= N_WORKERS )); then
    # More files than CPUs: saturate at the file level, 1 thread per process
    FILE_WORKERS=$N_WORKERS
    T_WORKERS=1
else
    # Fewer files than CPUs: each process gets a share of the threads
    FILE_WORKERS=$N_FILES
    T_WORKERS=$(( N_WORKERS / N_FILES ))
fi

echo "Processing ${N_FILES} file(s) — ${FILE_WORKERS} file(s) in parallel, ${T_WORKERS} timestep thread(s) each."

job_count=0
for CZI in "${ALL_FILES[@]}"; do
    echo "[$(date +%T)] Starting: ${CZI}"
    # shellcheck disable=SC2086
    python "${PYTHON_SCRIPT}" "${CZI}" --workers "${T_WORKERS}" ${EXTRA_ARGS} &

    (( job_count++ ))
    # Once FILE_WORKERS jobs are running, wait for one to finish before launching more
    if (( job_count >= FILE_WORKERS )); then
        wait -n 2>/dev/null || wait   # 'wait -n' needs bash ≥ 4.3
        (( job_count-- ))
    fi
done

# Wait for all remaining workers to finish
wait
echo "All ${N_FILES} file(s) complete."
