#!/bin/bash
# ============================================================
#  Reconstruct Hour-20 dropped-users map from Final_Animation.html
#  + detailed_drop_log.csv → hour20_drops.html
#
#  Usage:
#    sbatch run_visualize.sh
#
#  Requires:
#    - Final_Animation.html   in $RESULTS_DIR
#    - detailed_drop_log.csv  in $RESULTS_DIR
# ============================================================
#SBATCH --partition=cpu
#SBATCH --job-name=ntn_viz
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --error=viz-%j.err
#SBATCH --output=viz-%j.out
#SBATCH --mail-user=djad.benguerra@univ-lr.fr
#SBATCH --mail-type=END,FAIL

set -euo pipefail
unset PYTHONHOME
unset PYTHONPATH

# ── paths ────────────────────────────────────────────────────
PROJECT_DIR="/Utilisateurs/dbenguer/ntn_tn_optim"
ENV_NAME="ntn_env"
RESULTS_DIR="${PROJECT_DIR}"   # adjust to latest job dir if needed
SCRIPT="${PROJECT_DIR}/reconstruct_hour20.py"

# ── environment ──────────────────────────────────────────────
module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── find latest results dir automatically if RESULTS_DIR not specific ────────
if [[ ! -d "${RESULTS_DIR}" ]]; then
    RESULTS_DIR=$(ls -td "${PROJECT_DIR}/results/job-"* 2>/dev/null | head -1)
    echo "Using results dir: ${RESULTS_DIR}"
fi

cd "${RESULTS_DIR}"

echo "=================================================="
echo "  NTN-TN Hour-20 Drop Map Reconstruction"
echo "  Results dir : ${RESULTS_DIR}"
echo "  Script      : ${SCRIPT}"
echo "=================================================="


python "${SCRIPT}" \
    --animation  "${RESULTS_DIR}/Final_Animation.html" \
    --log        "${RESULTS_DIR}/detailed_drop_log.csv" \
    --out        "${RESULTS_DIR}/hour20_drops.html" \
    --hour       20.0 \
    --max-users  300000 \

echo ""
echo "Done. Output: ${RESULTS_DIR}/hour20_drops.html"