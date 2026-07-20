#!/usr/bin/env python3
"""
sim_oracle.py — the REAL oracle for the iterative loop on the cluster:
given a placement y, recompute each eligible (demand point, candidate)
spectral efficiency with your actual TR 38.901 link budget and the ACTUAL
set of opened interferers (instead of the nominal 6-ring of Assumption 1).

This is the planning-granularity version of "run the simulator": it uses
your calculate_tn_sinr_capacity with the true deployed interferer geometry.
The full-simulator validation (mobility, PF scheduling, per-user association)
remains the final a-posteriori check via milp_placement_<hex>.csv.

Usage in run_real_tile.py / iterative run:
    from sim_oracle import make_sim_oracle
    oracle = make_sim_oracle(inst, cfg_path)
    res, hist = solve_iterative(inst, oracle, ...)
"""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial import cKDTree
import yaml


def make_sim_oracle(inst, cfg_path: str,
                    g_rx_ue_dbi: float = 0.0,
                    serving_bf_db: float = 12.0,
                    intf_suppression_db: float = 12.0,
                    body_loss_db: float = 3.0,
                    nf_db: float = 7.0,
                    nf_fr2_db: float = 10.0,
                    impl_loss: float = 0.65,
                    ue_h_m: float = 1.5):
    from hybrid_ntn_optimizer.link_budget.sinr import calculate_tn_sinr_capacity
    from hybrid_ntn_optimizer.models.base_station import DeploymentScenario

    with open(cfg_path) as f:
        scen_cfg = yaml.safe_load(f)["scenarios"]

    tiers = inst.tiers
    scen_of = {}
    for ti, t in enumerate(tiers):
        enum_key = "UMI" if t.name == "UMI_MMW" else t.name
        scen_of[ti] = DeploymentScenario[enum_key]

    def oracle(y: np.ndarray):
        open_idx = np.where(y)[0]
        realized = []
        if len(open_idx) == 0:
            return [list(se) for se in inst.elig_se]
        tree = cKDTree(inst.cand_xy[open_idx])
        cutoff_km = 5.0     # interference horizon (matches sim cutoffs)

        for u in range(len(inst.dem_mbps)):
            du = inst.dem_xy[u]
            near_loc = tree.query_ball_point(du, cutoff_km)
            near = [int(open_idx[l]) for l in near_loc]
            row = []
            for j, se0 in zip(inst.elig_j[u], inst.elig_se[u]):
                t = tiers[inst.cand_tier[j]]
                s = scen_cfg[t.name]
                # interferers = OPENED same-band sites within cutoff, minus j
                interferers = []
                for k in near:
                    if k == j:
                        continue
                    tk = tiers[inst.cand_tier[k]]
                    same_band = (abs(tk.freq_hz - t.freq_hz) < 1e6) or \
                                (0.5e9 < tk.freq_hz < 7e9 and
                                 0.5e9 < t.freq_hz < 7e9)
                    if not same_band:
                        continue
                    dk = float(np.hypot(*(inst.cand_xy[k] - du))) * 1000.0
                    sk = scen_cfg[tk.name]
                    interferers.append((
                        max(dk, float(sk["min_user_dist_m"])),
                        scen_of[inst.cand_tier[k]],
                        float(sk["carrier_freq_hz"]),
                        float(sk["default_h_bs"]),
                        0.0, 0.0,
                        float(sk["p_tx_dbm"]), float(sk["g_tx_dbi"]),
                        0.0, 0.0, None,
                    ))
                d_m = max(float(np.hypot(*(inst.cand_xy[j] - du))) * 1000.0,
                          float(s["min_user_dist_m"]))
                try:
                    _, _, se, _ = calculate_tn_sinr_capacity(
                        bs_height_m=float(s["default_h_bs"]),
                        dist_to_serving_m=d_m,
                        interferers=interferers,
                        shadow_sigma_los_db=0.0, shadow_sigma_nlos_db=0.0,
                        scenario=scen_of[inst.cand_tier[j]],
                        p_tx_dbm=float(s["p_tx_dbm"]),
                        g_tx_dbi=float(s["g_tx_dbi"]),
                        g_rx_ue_dbi=g_rx_ue_dbi,
                        serving_beamforming_gain_db=serving_bf_db,
                        interferer_beamforming_suppression_db=intf_suppression_db,
                        carrier_freq_hz=float(s["carrier_freq_hz"]),
                        bandwidth_hz=float(s["bandwidth_hz"]),
                        body_loss_db=body_loss_db,
                        noise_figure_db=(nf_fr2_db
                                         if float(s["carrier_freq_hz"]) > 24e9
                                         else nf_db),
                        implementation_loss_factor=impl_loss,
                        ue_height_m=ue_h_m,
                    )
                except Exception:
                    se = 0.0
                row.append(float(max(se, 0.0)))
            realized.append(row)
        return realized

    return oracle
