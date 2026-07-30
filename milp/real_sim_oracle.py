#!/usr/bin/env python3
"""
real_sim_oracle.py — THE SIMULATOR INSIDE THE LOOP.

Replaces the analytic planning oracle: after each MILP solve, the ACTUAL
3GPP system-level simulator is run on this hex's users and the MILP's towers
for ONE busy hour, and its measured outcome is fed back as corrections to the
optimisation model. Validation is no longer a separate downstream step -- the
thing that judges is the thing that corrects.

WHAT THE SIMULATOR CORRECTS (two distinct feedbacks):

 (1) eta_{u,j}  -- realised spectral efficiency per (demand point, site).
     Measured from the users that actually attached to j, WITH shadowing
     draws, sector patterns and the true opened-interferer geometry.
     Corrects the frozen-SE assumption (Prop. 3).

 (2) rho_j      -- per-site REALISED CAPACITY EFFICIENCY:
            rho_j = (demand the simulator actually served through j)
                  / (demand the MILP assigned to j)
     One number per site, in [0,1]. It absorbs everything the MILP's
     idealised association cannot represent: greedy per-user attachment
     instead of genie fractional assignment, admission control, PF
     scheduling, mobility, and cell-edge losses. Fed back as a derating of
     the per-sector bandwidth budget:
            sum_u (d_u/eta_uj) x_uj  <=  W_t * rho_j * y_j
     so the next solve builds against capacity the SIMULATOR agrees exists.

The loop therefore converges to a deployment whose planned service the full
simulator reproduces -- there is no residual "planning vs reality" gap to
report afterwards, because the gap is what the loop is minimising.

COST: one single-hour simulation per iteration per hex. The single-hour hook
is cfg.simulation.duration_s = 72000 with time_step_s = 3600, which makes
full_pipeline evaluate exactly one step at hour 20.
"""
from __future__ import annotations
import copy
import math
from typing import Dict, List, Tuple

import numpy as np

try:
    import h3
except Exception:
    h3 = None

AZIMUTHS = (30.0, 150.0, 270.0)
ENUM_OF = {"RMA": "RMA", "UMA": "UMA", "UMI": "UMI", "UMI_MMW": "UMI"}


# ----------------------------------------------------------------------
def build_base_stations(inst, y, scen_cfg, h3_resolution: int = 5):
    """MILP placement -> BaseStation objects, 3 sectors per site sharing a
    site_id, RF from the same YAML the MILP used. Mirrors coverage.py's
    construction exactly so the simulator cannot tell the difference."""
    from hybrid_ntn_optimizer.models.base_station import (BaseStation,
                                                          DeploymentScenario)
    from candidate_generator import unproject

    bss, bs_id = [], 0
    open_idx = np.where(y)[0]
    for site_id, j in enumerate(open_idx):
        tier = inst.tiers[inst.cand_tier[j]].name
        sc = scen_cfg[tier]
        la, lo = unproject(inst.cand_xy[j, 0:1], inst.cand_xy[j, 1:2], inst.lat0)
        lat, lon = float(la[0]), float(lo[0])
        for az in AZIMUTHS:
            bs = BaseStation(
                bs_id=bs_id, lat=lat, lon=lon,
                scenario=DeploymentScenario[ENUM_OF[tier]],
                p_tx_dbm=float(sc["p_tx_dbm"]), g_tx_dbi=float(sc["g_tx_dbi"]),
                carrier_freq_hz=float(sc["carrier_freq_hz"]),
                total_bandwidth_hz=float(sc["bandwidth_hz"]),
                capacity_mbps=float(sc.get("bs_capacity_mbps", 0.0)),
                bs_height_m=float(sc["default_h_bs"]),
                shadow_sigma_los_db=float(sc["shadow_sigma_los_db"]),
                shadow_sigma_nlos_db=float(sc["shadow_sigma_nlos_db"]),
                interference_cutoff_m=float(sc["interference_cutoff_m"]),
                coverage_radius_km=float(sc["coverage_radius_km"]),
                min_user_dist_m=float(sc["min_user_dist_m"]),
                use_physical_radius=True)
            bs.site_id = int(site_id)
            bs.sector_azimuth_deg = float(az)
            bs.num_sectors = 3
            bs.milp_cand_index = int(j)          # back-reference for feedback
            bs.assigned_user_count = 0
            bs.cluster_density = 0.0
            bs.raw_cluster_radius_km = float(sc["coverage_radius_km"])
            bs.discovery_radius_km = float(sc["coverage_radius_km"])
            bs.discovery_cluster_size = 0
            bs.area_class = "TN-Service-Area"
            bs.set_resolution(h3_resolution)
            bss.append(bs)
            bs_id += 1
    return bss, {int(b.bs_id): int(b.milp_cand_index) for b in bss}


# ----------------------------------------------------------------------
def make_real_simulation_oracle(inst, cfg, hex_users, leos, region,
                                scen_cfg, agg_res: int = 9,
                                workers: int = 1,
                                se_percentile: float = 20.0,
                                calibrate: bool = True,   # SEE NOTE BELOW
                                skip_first_sim: bool = False,
                                log: bool = False):
    """se_percentile: which quantile of the REALISED per-user SE distribution
    the MILP should plan against. The mean/median plans for the typical user,
    so ~half the users in a demand point are worse than assumed and drop under
    shadowing. Planning at p20 builds the margin that drives realised drops
    toward zero -- it is a deterministic surrogate for a chance constraint
    P(user served) >= 1 - eps. Raise the percentile for cheaper networks,
    lower it for fewer drops.

    skip_first_sim: reuse a simulation already run by the caller (the smoke
    test runs one in [C]); avoids paying 24 min twice."""
    """Return an oracle(y, x_assign) -> (elig_se_realised, rho_per_cand).

    hex_users : the User objects of this hex+halo (same ones the MILP's
                demand points were aggregated from).
    leos, region : passed straight through to the simulator (built once by
                the driver exactly as scenario.py does).
    """
    from omegaconf import OmegaConf
    from full_pipeline_hooks import run_single_hour   # thin wrapper, below

    # Map every user to its demand point ONCE, by nearest demand-point
    # centroid in the instance's own projected frame. Robust: does not depend
    # on the Instance storing res-9 keys, and it is exactly the aggregation
    # the MILP used (demand points ARE the res-9 centroids).
    from scipy.spatial import cKDTree
    from candidate_generator import project_km
    _ux, _uy = project_km(
        np.array([float(getattr(u, "current_lat", None)
                        if getattr(u, "current_lat", None) is not None
                        else u.home_lat) for u in hex_users]),
        np.array([float(getattr(u, "current_lon", None)
                        if getattr(u, "current_lon", None) is not None
                        else u.home_lon) for u in hex_users]),
        inst.lat0)
    _tree = cKDTree(inst.dem_xy)
    _, u_dem = _tree.query(np.column_stack([_ux, _uy]))   # user -> demand pt

    _first_call = True
    # Pristine NOMINAL spectral efficiencies. Ratios must always be taken
    # against these, never against an already-corrected value, or the
    # correction compounds across iterations.
    _se_nominal = [list(r) for r in inst.elig_se]

    def oracle(y, x_assign: Dict[Tuple[int, int], float] | None = None):
        bss, bsid_to_cand = build_base_stations(inst, y, scen_cfg)
        if not bss:
            return [list(se) for se in inst.elig_se], {}

        # ---- RUN THE REAL SIMULATOR FOR ONE BUSY HOUR --------------------
        nonlocal _first_call
        users_copy = hex_users        # simulator mutates user state in place
        if _first_call and skip_first_sim:
            if log:
                print("      [sim] reusing the caller's simulation "
                      "(skip_first_sim)", flush=True)
        else:
            sim_cfg = copy.deepcopy(cfg)
            try:
                from omegaconf import open_dict
                with open_dict(sim_cfg):
                    sim_cfg.simulation.duration_s = 72000
                    sim_cfg.simulation.time_step_s = 3600
                    sim_cfg.simulation.num_workers = int(workers)
            except Exception:
                pass
            run_single_hour(sim_cfg, users_copy, bss, leos, region)
        _first_call = False

        # ---- HARVEST per-user realised state ----------------------------
        # served_by[(dem_pt, cand)] -> [se samples]; served/assigned per site
        from full_pipeline_hooks import user_result_fields
        se_samples: Dict[Tuple[int, int], List[float]] = {}
        served_by_cand: Dict[int, float] = {}
        attached_by_cand: Dict[int, float] = {}
        n_attached = 0
        n_dropped = 0
        dropped_mbps = 0.0
        drop_by_dem: Dict[int, float] = {}
        for u, dp in zip(users_copy, u_dem):
            bsid, se, got, dem_u, dropped = user_result_fields(u)
            if dropped:
                n_dropped += 1
                short = max(dem_u - got, 0.0)
                dropped_mbps += short
                drop_by_dem[int(dp)] = drop_by_dem.get(int(dp), 0.0) + short
            if bsid is None or bsid not in bsid_to_cand:
                continue
            j = bsid_to_cand[bsid]
            n_attached += 1
            served_by_cand[j] = served_by_cand.get(j, 0.0) + got
            attached_by_cand[j] = attached_by_cand.get(j, 0.0) + dem_u
            if se > 0:
                se_samples.setdefault((int(dp), j), []).append(se)
        if log:
            print(f"      [sim] {n_attached:,} users attached to MILP cells; "
                  f"{len(se_samples):,} (demand-pt, site) SE samples; "
                  f"{n_dropped:,} users short by {dropped_mbps/1e3:.1f} Gbps",
                  flush=True)

        # ---- FEEDBACK 1: realised eta per (demand point, candidate) -----
        # ---- FEEDBACK 1: realised eta ------------------------------------
        # NAIVE VERSION (what this used to do): correct ONLY the pairs the
        # simulator exercised, leave the rest at their nominal value. On the
        # real hex only ~12% of eligible pairs are ever exercised, so the
        # model ends up believing the sites it TESTED are bad and the sites it
        # NEVER TESTED are good -- it then rebuilds on untested candidates,
        # which underperform in turn. Measured effect: the "corrected" solve
        # was consistently ~15 Gbps WORSE than the uncorrected one
        # (25.2->43.2 and 28.2->42.8 Gbps on hex 852b9bd7).
        #
        # CALIBRATED VERSION (calibrate=True): fit a correction factor
        #     kappa_t(d) = realised_SE / nominal_SE
        # per tier and distance bin from every sample, then apply it to ALL
        # eligible pairs. Exercised and unexercised pairs are treated
        # identically, so the selection bias disappears.
        realised = []
        spread = []
        ratios: Dict[Tuple[int, int], List[float]] = {}   # (tier, dbin)->ratios
        NB, DMAX = 8, 3.0                                  # bins, km

        def _dbin(dkm):
            return min(NB - 1, int(NB * min(dkm, DMAX) / DMAX))

        for uidx in range(len(inst.dem_mbps)):
            for j, se_nom in zip(inst.elig_j[uidx], _se_nominal[uidx]):
                smp = se_samples.get((uidx, j))
                if not smp or se_nom <= 0:
                    continue
                meas = float(np.percentile(smp, se_percentile))
                d = float(np.hypot(*(inst.cand_xy[j] - inst.dem_xy[uidx])))
                ratios.setdefault((int(inst.cand_tier[j]), _dbin(d)),
                                  []).append(meas / se_nom)
                if len(smp) >= 5:
                    spread.append(float(np.percentile(smp, 80)
                                        - np.percentile(smp, 20)))

        kappa = {k: float(np.median(v)) for k, v in ratios.items()
                 if len(v) >= 5}
        kappa_tier = {}
        for (t, _b), v in ratios.items():
            kappa_tier.setdefault(t, []).extend(v)
        kappa_tier = {t: float(np.median(v)) for t, v in kappa_tier.items()}

        for uidx in range(len(inst.dem_mbps)):
            row = []
            for j, se_nom in zip(inst.elig_j[uidx], _se_nominal[uidx]):
                if not calibrate:
                    smp = se_samples.get((uidx, j))
                    row.append(float(np.percentile(smp, se_percentile))
                               if smp else se_nom)
                    continue
                t = int(inst.cand_tier[j])
                d = float(np.hypot(*(inst.cand_xy[j] - inst.dem_xy[uidx])))
                k = kappa.get((t, _dbin(d)), kappa_tier.get(t, 1.0))
                row.append(max(0.1, se_nom * k))
            realised.append(row)

        if log and kappa_tier:
            names = {i: t.name for i, t in enumerate(inst.tiers)}
            txt = "  ".join(f"{names.get(t,t)}={k:.2f}"
                            for t, k in sorted(kappa_tier.items()))
            n_cal = sum(len(v) for v in ratios.values())
            print(f"      [cal] kappa = realised/nominal SE from {n_cal:,} "
                  f"samples: {txt}", flush=True)
            print(f"      [cal] applied to ALL {sum(len(r) for r in realised):,}"
                  f" eligible pairs ({len(kappa)} tier x distance bins fitted)"
                  f" -- no selection bias", flush=True)
        if log and spread:
            print(f"      [sim] planning at p{se_percentile:.0f} of realised SE; "
                  f"within-demand-point SE spread (p80-p20) median "
                  f"{np.median(spread):.2f} bps/Hz", flush=True)

        # ---- FEEDBACK 2: rho_j = simulator-served / MILP-assigned -------
        # rho_j must measure DELIVERY EFFICIENCY, not association divergence.
        #   delivered / ATTACHED  -> what the site could actually push to the
        #                            users that really showed up (the derating
        #                            the MILP needs)
        #   attached / ASSIGNED   -> how far greedy attachment strayed from the
        #                            plan (reported, NOT fed back: re-routing is
        #                            not a capacity fault of site j)
        rho: Dict[int, float] = {}
        divergence: Dict[int, float] = {}
        if x_assign:
            assigned: Dict[int, float] = {}
            for (uidx, j), frac in x_assign.items():
                if frac > 1e-9:
                    assigned[j] = assigned.get(j, 0.0) + \
                        frac * float(inst.dem_mbps[uidx])
            for j in set(list(assigned) + list(attached_by_cand)):
                att = attached_by_cand.get(j, 0.0)
                if att > 1e-6:
                    rho[j] = min(1.0, max(0.05, served_by_cand.get(j, 0.0) / att))
                a = assigned.get(j, 0.0)
                if a > 1e-6:
                    divergence[j] = att / a
        if log and divergence:
            dv = np.array(list(divergence.values()))
            print(f"      [sim] association divergence attached/assigned: "
                  f"mean={dv.mean():.2f} p10={np.percentile(dv,10):.2f} "
                  f"p90={np.percentile(dv,90):.2f}  (reported, not fed back)",
                  flush=True)
        if log:
            if rho:
                v = np.array(list(rho.values()))
                print(f"      [sim] rho: mean={v.mean():.3f} "
                      f"p10={np.percentile(v,10):.3f} min={v.min():.3f} "
                      f"over {len(rho)} sites", flush=True)
        oracle.last_divergence = divergence
        oracle.last_drop_by_dem = drop_by_dem
        return realised, rho, drop_by_dem

    return oracle


# ---------------------------------------------------------------------------
def coordination_gap(hex_users, bss, log: bool = True):
    """UPPER BOUND on what perfect association steering could recover, using
    ONLY data the simulator already produced. No patch to full_pipeline.

    Measured on hex 852b9bd7: the deployed network carries ~2.5x the capacity
    of the demand, mean cell utilisation is 37%, yet 5.4% of demand drops.
    That is only possible if dropped users sit within reach of cells that have
    spare spectrum but were not chosen by greedy max-SINR attachment.

    Method:
      used_hz[j]  = sum of tn_eval_hz over users the simulator attached to j
      spare_hz[j] = total_bandwidth_hz - used_hz[j]     (per sector cell)
      for each short user, in descending shortfall:
          find cells whose coverage radius contains the user
          fill from the one with the most spare, at the user's measured SE
    The result is an UPPER bound: a user steered to a different cell would see
    lower SINR than the one it actually measured, so real recovery is smaller.
    If even this bound is small, the shortfall is genuine capacity. If it is
    large, the shortfall is coordination and no extra CAPEX will remove it.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from full_pipeline_hooks import user_result_fields

    cells = {int(b.bs_id): b for b in bss}
    used = {}
    for u in hex_users:
        bsid, se, got, dem, dropped = user_result_fields(u)
        if bsid is not None and bsid in cells and se > 0:
            # Hz ACTUALLY consumed = delivered Mbps / spectral efficiency.
            # (Do NOT use tn_eval_hz: that is the bandwidth the scheduler
            # evaluated/offered for the link, ~10,000x larger than what the
            # user consumed. Summing it made every cell look 100% full and
            # produced a spurious 0% coordination gap.)
            used[bsid] = used.get(bsid, 0.0) + got * 1e6 / se
    spare = {j: max(float(b.total_bandwidth_hz) - used.get(j, 0.0), 0.0)
             for j, b in cells.items()}
    # SANITY CHECK: our reconstructed utilisation must match the simulator's
    # own reported mean cell utilisation. If it does not, the Hz accounting
    # is wrong and the gap number is meaningless.
    utils = [used.get(j, 0.0) / max(float(b.total_bandwidth_hz), 1.0)
             for j, b in cells.items()]
    if log and utils:
        import numpy as _np
        print(f"      [coord] reconstructed mean cell utilisation "
              f"{100*_np.mean(utils):.1f}% (compare with the simulator's own "
              f"[util] line -- they MUST agree, else the Hz accounting is off)",
              flush=True)

    ids = list(cells)
    xy = np.array([[cells[j].lat, cells[j].lon] for j in ids])
    tree = cKDTree(xy)
    KM_DEG = 111.0

    short = []
    for u in hex_users:
        bsid, se, got, dem, dropped = user_result_fields(u)
        if dem - got > 1e-9:
            short.append((dem - got, float(getattr(u, "current_lat", 0.0)),
                          float(getattr(u, "current_lon", 0.0)),
                          se if se > 0 else 1.0))
    short.sort(reverse=True)

    total_short = sum(s[0] for s in short)
    recovered = 0.0
    for need_mbps, la, lo, se in short:
        r_km = 3.0
        cand = tree.query_ball_point([la, lo], r_km / KM_DEG)
        best = sorted(((spare[ids[i]], ids[i]) for i in cand
                       if _in_range(cells[ids[i]], la, lo)), reverse=True)
        for sp, j in best:
            if need_mbps <= 1e-9:
                break
            hz_needed = need_mbps * 1e6 / max(se, 1e-6)
            take = min(sp, hz_needed)
            if take <= 0:
                continue
            got_mbps = take * se / 1e6
            recovered += got_mbps
            need_mbps -= got_mbps
            spare[j] -= take

    if log:
        print(f"      [coord] shortfall {total_short/1e3:,.1f} Gbps | "
              f"absorbable by in-range cells with spare spectrum: "
              f"{recovered/1e3:,.1f} Gbps "
              f"({100*recovered/max(total_short,1e-9):.0f}%)", flush=True)
        print(f"      [coord] -> that fraction is a COORDINATION gap "
              f"(steering/CIO recovers it, CAPEX does not); the remainder is "
              f"genuine capacity or coverage shortage.", flush=True)
    return {"shortfall_mbps": total_short, "absorbable_mbps": recovered,
            "fraction": recovered / max(total_short, 1e-9)}


def _in_range(bs, lat, lon):
    import math
    dlat = (lat - bs.lat) * 111.0
    dlon = (lon - bs.lon) * 111.0 * math.cos(math.radians(bs.lat))
    return math.hypot(dlat, dlon) <= float(bs.coverage_radius_km)