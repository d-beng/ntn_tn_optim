#!/bin/bash
# ============================================================
#  Analyze detailed_drop_log.csv (where demand goes + why drops).
#  Submit with:  sbatch run_analyze_drops.sh
# ============================================================
#SBATCH --partition=cpu
#SBATCH --job-name=drop_analysis
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --error=analysis-%j.err
#SBATCH --output=analysis-%j.out
#SBATCH --mail-user=djad.benguerra@univ-lr.fr
#SBATCH --mail-type=FAIL

set -euo pipefail

# Clean inherited Python env so conda's Python works.
unset PYTHONHOME
unset PYTHONPATH

# ----- EDIT THESE -------------------------------------------
PROJECT_DIR="$HOME/ntn_tn_optim"  
ENV_NAME="ntn_env"
CSV_PATH="$PROJECT_DIR/detailed_drop_log.csv"   
HOUR="20"                                        # which Hour to analyze
# ------------------------------------------------------------

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

# This script does not unpickle User objects, so it needs no package import.
python -u analyze_drops.py --csv "$CSV_PATH" --hour "$HOUR"

echo "Done. See the printed breakdown above and drop_analysis.png."