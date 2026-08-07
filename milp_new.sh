#!/bin/bash
#SBATCH --job-name=milp_smoke_tile
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=200
#SBATCH --mem=420G
#SBATCH --output=milp_smoke_tile-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=djad.benguerra@univ-lr.fr
set -uo pipefail          # NOT -e: conda activate returns odd codes

# ---------------------------------------------------------------------------
# The guard now checks CAPABILITY, not which env we are in.
# What actually matters is: numpy 2.x (users.pkl was written with it) and a
# clean import of the stack. Every successful run so far has been base
# Anaconda + ~/.local packages, so that combination is ACCEPTED -- but it is
# verified before the 8-minute pickle load instead of assumed.
# ---------------------------------------------------------------------------

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)" 2>/dev/null || true

ENV_NAME="ntn_env"
if conda activate "$ENV_NAME" 2>/dev/null; then
    echo "conda activate $ENV_NAME -> ok"
else
    echo "WARN: could not activate $ENV_NAME"
fi

echo "python      : $(which python)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}"
conda env list 2>/dev/null || true

# ~/.local/lib/python3.11/site-packages sits AHEAD of the env on sys.path,
# so `pip install --user numpy==1.26.4` was shadowing the env's own numpy.
# This disables user-site entirely -> the env's packages win. (Anything you
# installed ONLY with --user, e.g. torch, becomes invisible; install it into
# the env instead if you need it.)
export PYTHONNOUSERSITE=1

export PYTHONPATH=/Utilisateurs/dbenguer/ntn_tn_optim/src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=65 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# --- capability check: fails in 2 s instead of after an 8-minute load ------
python - <<'PYCHK' || exit 1
import sys, numpy as np
print(f"numpy {np.__version__}")
print(f"  from {np.__file__}")
if int(np.__version__.split('.')[0]) < 2:
    sys.exit("FATAL: users.pkl was written with numpy 2.x; it cannot be read "
             "under numpy 1.x.\n"
             "  The env is active but numpy came from the path printed above.\n"
             "  If that path is ~/.local, PYTHONNOUSERSITE=1 should have hidden\n"
             "  it -- check it is exported. If it is the ENV path, install a\n"
             "  newer numpy INTO the env (no --user):\n"
             "      pip install 'numpy>=2.0'")
if "/.local/" in np.__file__:
    print("  WARNING: numpy is coming from ~/.local, not the env "
          "(PYTHONNOUSERSITE not in effect)")
import pandas, h3, highspy, scipy
print(f"pandas {pandas.__version__} | h3 {h3.__version__} | stack OK")
PYCHK

cd /Utilisateurs/dbenguer/ntn_tn_optim/milp

# --- stale-file guards ----------------------------------------------------
grep -q "drop_by_dem"         real_sim_oracle.py || { echo "STALE real_sim_oracle.py"; exit 1; }
grep -q "skip_first_sim=True" smoke_real_sim.py  || { echo "STALE smoke_real_sim.py (would simulate TWICE)"; exit 1; }
grep -q "tn_reason census"    smoke_real_sim.py  || { echo "smoke_real_sim.py missing drop-reason census"; exit 1; }
grep -q "require_coverage"    hex_milp.py        || { echo "STALE hex_milp.py"; exit 1; }


# python smoke_real_sim.py workers=200 +solver_threads=64 \
#     +solve_time_limit=21600 \
#     +dens_uma=999999 +dens_umi=0 +k_elig=12 \
#     +agg_res=10 +agg_safety=1.0 +rho_dep=0.65 +rho_cand=0.75 \
#     +cross_tier_m=legacy +min_outage=true +calibrate=true

# # archive immediately — base_stations.csv is overwritten by every run
# mkdir -p results/runA
# cp base_stations.csv site_utilisation.csv bs_sector_utilisation.csv \
#    cell_se_sinr.csv ntn_cell_demand.csv drops_852b9bd7* results/runA/

# python plot_deployment.py --bs results/runA/base_stations.csv \
#     --util results/runA/site_utilisation.csv --prefix results/runA/deploy

python province_solver.py \
    --workers 200 --min-users 1 \
    --agg-res 10 --agg-safety 1.0 \
    --rho-dep 0.65 --rho-cand 0.75 --k-elig 12 \
    --dens-uma 999999 --dens-umi 0 \
    --min-outage --gap 0.02 --time-limit 1800 \
    --out province_placement.csv

python milp_to_bs_csv.py --placement province_placement.csv \
    --config-dir /Utilisateurs/dbenguer/ntn_tn_optim/configs \
    --config-name base --out milp_bs.csv

python run_province_sim.py workers=200 +bs_csv=milp_bs.csv