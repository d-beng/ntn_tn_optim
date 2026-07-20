# MILP Stack — Cluster Integration Guide

## 0. What goes where

Two INDEPENDENT groups of files. Don't mix them.

### Group A — simulator files (REPLACE existing ones in your package)
| file | destination |
|---|---|
| `coverage.py` | `src/hybrid_ntn_optimizer/terrestrial/coverage.py` |
| `full_pipeline.py` | `src/hybrid_ntn_optimizer/simulation/full_pipeline.py` |
| `sinr.py` | `src/hybrid_ntn_optimizer/link_budget/sinr.py` |

These are the simulator fixes (gap fill, mmWave overlay, array-snapshot
workers, gc.freeze, tuple interferers). **The MILP stack imports sinr.py, so
Group A must be deployed BEFORE the MILP stack runs** — `run_real_tile.py`
and `sim_oracle.py` call `calculate_tn_sinr_capacity` with the NEW
raw-tuple interferer format. Old sinr.py = crash.

### Group B — MILP stack (NEW directory, nothing overwritten)
```
mkdir -p /home/db3n/Documents/Ph.D./Courses/ntn_tn_test/ntn_tn_optim-master_2/milp
```
| file | role |
|---|---|
| `candidate_generator.py` | builds Instance: lattices, demand pts, eligibility+sectors, conflicts, ext_sites |
| `hex_milp.py` | formulation (8) in HiGHS + sector_usage_mhz helper |
| `run_tile.py` | synthetic 10K smoke test (no cluster data needed) |
| `run_real_tile.py` | ONE real hex from users.pkl + real link budget → placement CSV |
| `baselines.py` | greedy + GA on the same Instance |
| `refine.py` | coarse→fine local repositioning |
| `iterative_milp.py` | the novel simulator-corrected loop (proxy oracle built in) |
| `sim_oracle.py` | REAL oracle: SE vs actually-opened interferers (needs Group A) |
| `experiments.py` | all 5 methods on one instance → results.csv + figures |
| `province_solver.py` | multi-hex pass1/pass2/stitch driver |

All ten go in `milp/`, flat, side by side (they import each other by name).

## 1. Environment (one time)
```bash
conda activate ntn_env
pip install highspy pyyaml matplotlib     # h3, numpy, scipy already there
```

## 2. Verification ladder — run IN THIS ORDER, each gates the next

### Step 1 — synthetic smoke test (no cluster data, ~1 min)
```bash
cd /home/db3n/Documents/Ph.D./Courses/ntn_tn_test/ntn_tn_optim-master_2/milp
python run_tile.py
```
PASS = `status: Optimal`, `served %: 100`, solve < 30 s.
Proves: highspy installed, generator+MILP consistent.

### Step 2 — all methods on synthetic (~5 min)
```bash
python experiments.py
```
PASS = summary table with milp_refined best, results.csv + 2 PNGs written.
Proves: baselines, refinement, iterative loop all wired.

### Step 3 — ONE REAL HEX  ← the critical run
```bash
export PYTHONPATH=/home/db3n/Documents/Ph.D./Courses/ntn_tn_test/ntn_tn_optim-master_2/src
python run_real_tile.py                     # densest hex auto-selected
# knobs if too slow: --agg-res 8   --k-elig 4   --time-limit 3600
```
PASS = gap ≤ 0.02 within the limit, sane tier mix (UMi in the core),
`milp_placement_<hex>.csv` written.
Proves: users.pkl extraction, YAML tier loading, REAL sinr.py SE calls.
**Paste this output before going further — agg-res/runtime tuning and
everything downstream depends on it.**

### Step 4 — real-hex experiments (paper table)
Edit `experiments.py::main` — replace the synthetic block:
```python
from run_real_tile import load_tiers, make_real_se_fn, extract_hex
tiers = load_tiers(CFG, 1600, 4000, 15000)
se_fn = make_real_se_fn(CFG)
hex_id, lat, lon, mbps = extract_hex(USERS, None, 2.847)
inst = build_instance(lat, lon, mbps, hex_id, tiers=tiers, se_fn=se_fn,
                      rho_cand=1.0, agg_res=9, K_elig=6)
```
and pass the real oracle to the iterative run:
```python
from sim_oracle import make_sim_oracle
oracle = make_sim_oracle(inst, CFG)
```

### Step 5 — small province run, then full
```bash
python province_solver.py --max-hexes 5 --workers 8      # validate loop
python province_solver.py --workers 64                   # all dense hexes
```
PASS = pass-2 `(±d)` deltas small/negative at borders, province_placement.csv.

### Step 6 — simulator validation (closes the loop)
Feed `milp_placement_<hex>.csv` / `province_placement.csv` into the simulator
in place of its own placement (coverage.py bypass), run hour 20, compare
served demand vs the MILP's claim. (Harness for this: next deliverable.)

## 3. Pitfalls
- `ModuleNotFoundError: hybrid_ntn_optimizer` → PYTHONPATH not exported.
- `KeyError: UMI_MMW` → fine; tier skipped if not in YAML (add block to use it).
- users.pkl load takes minutes and ~30-60 GB — run steps 3+ on a compute node.
- Numbers like `100e6` in YAML load as strings — handled (float() everywhere).
- province_solver workers each solve with 1 HiGHS thread by design; parallelism
  is ACROSS hexes. Don't set --workers × threads > cores.
- Steps 3+ need Group A deployed (tuple-interferer sinr.py). Verify with:
  `grep -c "d_intf_m, scen_j" src/hybrid_ntn_optimizer/link_budget/sinr.py`
  → must print ≥ 1.
