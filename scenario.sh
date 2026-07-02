#!/bin/bash

# ----- SLURM OPTIONS ----------------------------------------
#SBATCH --partition=cpu
#SBATCH --job-name=sim_2
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # one program...
#SBATCH --cpus-per-task=250         # ...with this many cores -> this is what the ProcessPool gets
#SBATCH --mem=500GB
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --mail-user=djad.benguerra@univ-lr.fr
#SBATCH --mail-type=END,FAIL

# ============================================================
#  0. EDIT THESE
# ============================================================
PROJECT_DIR="$HOME/ntn_tn_optim/"     # where your code + configs live on the SHARED filesystem
ENV_NAME="ntn_env"             # conda environment name
PKG_NAME="hybrid_ntn_optimizer"  # top-level importable package

# ============================================================
#  1. CONDA ENVIRONMENT
# ============================================================

# ============================================================
#  3. THREADING — keep BLAS/NumPy single-threaded so they
#     don't fight the ProcessPool for cores.
# ============================================================
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh


if ! conda activate "$ENV_NAME" 2>/dev/null; then
    echo "Environment '$ENV_NAME' not found — creating it..."
    conda create -n "$ENV_NAME" python=3.11 -y
    conda activate "$ENV_NAME"
fi

cd "$PROJECT_DIR"

# ============================================================
#  2. DEPENDENCIES + MAKE THE PACKAGE IMPORTABLE
# ============================================================
if [[ -f requirement.txt ]]; then
    pip install --no-cache-dir --progress-bar=off -r requirement.txt
elif [[ -f requirements.txt ]]; then
    pip install --no-cache-dir --progress-bar=off -r requirements.txt
fi

# Prefer a proper editable install; fall back to PYTHONPATH (the "send it to src" trick).
if [[ -f "$PROJECT_DIR/pyproject.toml" || -f "$PROJECT_DIR/setup.py" ]]; then
    echo "Installing $PKG_NAME in editable mode..."
    pip install --no-cache-dir -e "$PROJECT_DIR"
else
    PKG_DIR="$(find "$PROJECT_DIR" -maxdepth 3 -type d -name "$PKG_NAME" | head -n 1)"
    echo "Found package directory: $PKG_DIR"
    if [[ -z "$PKG_DIR" ]]; then
        echo "ERROR: could not locate package '$PKG_NAME' under $PROJECT_DIR" >&2
        exit 1
    fi
    export PYTHONPATH="$(dirname "$PKG_DIR"):${PYTHONPATH:-}"
    echo "PYTHONPATH set to: $(dirname "$PKG_DIR")"
fi



# ============================================================
#  4. OUTPUT STAGING — write on node-local scratch, copy back.
# ============================================================
SCRATCH="${SLURM_TMPDIR:-/tmp/${SLURM_JOB_ID}}"
export SIM_OUTPUT_DIR="$SCRATCH/ntn_outputs"     # scenario.py reads this env var
mkdir -p "$SIM_OUTPUT_DIR"

RESULTS_DIR="$PROJECT_DIR/results/job-${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

# Copy results back even if the run crashes or is killed (scratch is wiped at job end).
copy_back() {
    cp -a "$SIM_OUTPUT_DIR/." "$RESULTS_DIR/" 2>/dev/null || true
    echo "Results staged to: $RESULTS_DIR"
}
trap copy_back EXIT

# ============================================================
#  5. RUN
# ============================================================
echo "Launching simulation on ${SLURM_CPUS_PER_TASK} cores..."
export PYTHONUNBUFFERED=1
python scenario.py