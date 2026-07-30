#!/usr/bin/env python3
"""
full_pipeline_hooks.py — thin, honest wrapper around the REAL simulator.

run_single_hour() calls hybrid_ntn_optimizer's own
run_daily_mobility_simulation with a config that evaluates EXACTLY ONE time
step at hour 20 (duration_s=72000, time_step_s=3600 -> the pipeline's own
`range(20*3600, duration_s+step, step)` yields [72000]).

Nothing about the physics is reimplemented here: attachment, admission
control, PF scheduling, NTN beam allocation and drop accounting are the
simulator's own code paths. This module only exists so the MILP loop has one
stable call site and ONE place where the User field names live.

FIELD NAMES verified from a live run (smoke test [D], 2026-07):
    tn_eval_bs           str   "BS_1337"   <- attached cell, STRING not int
    spectral_efficiency  float 4.9197      <- realised SE [bps/Hz]
    served_mbps          float 0.2428      <- delivered throughput
    current_demand       float 0.2428      <- offered demand at this hour
    tn_reason            str   "Fully Served" / drop reason
    coverage_type        str   "TN" | "NTN" | ...
    tn_sinr_db, tn_I_dbm, tn_N_dbm, tn_num_interferers, tn_eval_hz
"""
from __future__ import annotations
import re

import numpy as np

_BS_NUM = re.compile(r"(\d+)")


def run_single_hour(cfg, users, base_stations, leos, region):
    from hybrid_ntn_optimizer.simulation.full_pipeline import (
        run_daily_mobility_simulation)
    return run_daily_mobility_simulation(cfg=cfg, users=users,
                                         base_stations=base_stations,
                                         leos=leos, region=region)


def parse_bs_id(raw):
    """'BS_1337' -> 1337 ; 1337 -> 1337 ; None/junk -> None.
    The simulator labels cells as strings; BaseStation.bs_id is an int, so
    the loop MUST parse or every lookup misses and rho comes back empty."""
    if raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    m = _BS_NUM.search(str(raw))
    return int(m.group(1)) if m else None


def user_result_fields(u):
    """Per-user realised outcome after a simulated hour.S
    Returns (bs_id:int|None, spec_eff, served_mbps, demand_mbps, dropped:bool).
    """
    bsid = parse_bs_id(getattr(u, "tn_eval_bs", None))
    se = float(getattr(u, "spectral_efficiency", 0.0) or 0.0)
    served = float(getattr(u, "served_mbps", 0.0) or 0.0)
    demand = float(getattr(u, "current_demand", 0.0) or 0.0)
    reason = str(getattr(u, "tn_reason", "") or "")
    dropped = (served + 1e-9) < demand or reason.lower().startswith("drop")
    return bsid, se, served, demand, dropped