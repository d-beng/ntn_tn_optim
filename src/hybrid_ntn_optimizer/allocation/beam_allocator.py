import math
import h3
from typing import Dict, List, Any, Set
from omegaconf import DictConfig

from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.models.beam import Beam
from hybrid_ntn_optimizer.link_budget.sinr import calculate_ntn_sinr_capacity
from hybrid_ntn_optimizer.core.utils import haversine_distance
from hybrid_ntn_optimizer.core.types import GeoPoint
from hybrid_ntn_optimizer.constellation.propagator import build_earth_satellite
from hybrid_ntn_optimizer.constellation.visibility import visible_satellites


def allocate_ntn_beams(
    cfg: DictConfig,
    leos: List[LEOConstellation],
    unmet_demand_ledger: Dict[str, List[Dict[str, Any]]],
    dt_s: float,
) -> List[Beam]:
    """
    3GPP TR 38.821 NTN scheduler — coverage-first, quality-weighted satellite pick.

    Policy (strict one-beam-per-cell, one-beam-per-(sat,cell)):
      - PASS 1 (COVERAGE): every demanding hex gets exactly ONE beam.
        Hexes are processed MOST-CONSTRAINED FIRST (fewest visible satellites),
        so cells with poor visibility claim a satellite before the shared ones
        fill. Each hex's satellite is chosen by a quality-weighted score that
        blends link quality (elevation) with load spreading (free beams), and
        ALWAYS falls back to any free visible satellite so coverage is never
        sacrificed for quality.
      - PASS 2 (CAPACITY): leftover beams go to hexes still carrying unmet
        demand, highest-unmet first. Still one beam per (sat, hex): a hex may
        receive an ADDITIONAL beam only from a DIFFERENT satellite that has not
        already beamed it this step.

    quality_weight (cfg.constellation.beam_quality_weight, default 0.3):
        1.0 = pure elevation (best SINR, more contention)
        0.0 = pure load spreading (best coverage robustness)
        0.3 = coverage-safe with a small quality bias (recommended).
    """
    # ── 1. RF / hardware parameters ──────────────────────────────────────────
    max_spot_beams = cfg.constellation.get("max_spot_beams", 15)
    min_elevation  = cfg.constellation.get("min_elevation_deg", 25.0)
    base_eirp_dbw  = cfg.constellation.get("eirp_dbw", 40.0)
    g_t_db         = cfg.constellation.get("g_t_db", 10.0)
    f_ntn          = cfg.constellation.get("freq_ghz", 2.2)
    bw_ntn         = cfg.constellation.get("bandwidth_hz", 40e6)
    sinr_min_ntn   = cfg.constellation.get("sinr_min_db", 0.0)
    theta_3db      = cfg.constellation.get("theta_3db_deg", 2.5)
    sll            = cfg.constellation.get("sll_db", 25.0)
    quality_weight = float(cfg.constellation.get("beam_quality_weight", 0.3))

    # ── 2. Snapshot orbital positions from ALL shells ────────────────────────
    sat_states = []
    earth_sats = []
    for leo in leos:
        sat_states.extend(leo.snapshot(dt_s=dt_s))
        earth_sats.extend(build_earth_satellite(d, leo.epoch_utc) for d in leo.descriptors)

    for sat in sat_states:
        sat.active_beams.clear()

    # O(1) satellite lookup (replaces the per-hex linear scan).
    sat_by_id = {s.satellite_id: s for s in sat_states}

    # ── 3. Build demanding-hex list ──────────────────────────────────────────
    hex_needs: List[Dict[str, Any]] = []
    for hex_id, user_list in unmet_demand_ledger.items():
        total_need = sum(item["unmet_mbps"] for item in user_list if item["unmet_mbps"] > 0.1)
        if total_need > 0.1:
            hex_needs.append({"hex_id": hex_id, "total_need": total_need, "users": user_list})

    if not hex_needs:
        return []

    # ── 4. Per-hex visibility, computed ONCE ─────────────────────────────────
    # hex_id -> list of (sat, record) for visible satellites (elevation-sorted).
    hex_vis: Dict[str, List] = {}
    hex_latlon: Dict[str, Any] = {}
    for needy in hex_needs:
        hid = needy["hex_id"]
        lat, lon = h3.cell_to_latlng(hid)
        hex_latlon[hid] = (lat, lon)
        recs = visible_satellites(
            states=sat_states, ground=GeoPoint(lat_deg=lat, lon_deg=lon),
            min_elevation_deg=min_elevation, earth_sats=earth_sats,
        )
        pairs = []
        for rec in recs:
            sat = sat_by_id.get(rec.satellite_id)
            if sat is not None:
                pairs.append((sat, rec))
        hex_vis[hid] = pairs

    def _free(sat) -> int:
        return max_spot_beams - len(sat.active_beams)

    def _pick_sat(hid, exclude_ids: Set[str] = frozenset()):
        """Quality-weighted choice among visible satellites that (a) have a free
        beam and (b) are not already beaming this hex. Falls back to ANY free
        visible satellite (coverage guaranteed). Returns (sat, rec) or (None, None)."""
        cands = [(sat, rec) for (sat, rec) in hex_vis[hid]
                 if _free(sat) > 0 and sat.satellite_id not in exclude_ids]
        if not cands:
            return None, None

        els  = [rec.elevation_deg for _, rec in cands]
        lds  = [_free(sat) for sat, _ in cands]
        el_lo, el_hi = min(els), max(els)
        ld_lo, ld_hi = min(lds), max(lds)

        def _norm(v, lo, hi):
            return 1.0 if hi == lo else (v - lo) / (hi - lo)

        w = quality_weight
        return max(
            cands,
            key=lambda sr: w * _norm(sr[1].elevation_deg, el_lo, el_hi)
                           + (1.0 - w) * _norm(_free(sr[0]), ld_lo, ld_hi),
        )

    all_active_beams: List[Beam] = []

    # ── 5. Serve one hex with one beam on a chosen satellite ─────────────────
    def _serve_hex(needy, sat, rec) -> Beam:
        """Per-user physics + proportional-fair scheduling on a single beam.
        (Logic preserved from the original sections 5–6.)"""
        hid = needy["hex_id"]
        hex_lat, hex_lon = hex_latlon[hid]
        slant_range_km = rec.slant_range_km
        elevation_deg  = rec.elevation_deg

        # Interference from THIS satellite's already-active beams.
        off_axis_angles_interferers: List[float] = []
        for existing_beam in sat.active_beams:
            adj_lat, adj_lon = h3.cell_to_latlng(existing_beam.target_cell_id)
            surface_dist_km = haversine_distance(hex_lat, hex_lon, adj_lat, adj_lon) / 1000.0
            off_axis_angles_interferers.append(
                math.degrees(math.atan2(surface_dist_km, slant_range_km)))

        eligible = [e for e in needy["users"] if e["unmet_mbps"] > 0.1]
        new_beam = Beam(
            satellite_id=sat.satellite_id, target_cell_id=hid,
            elevation_deg=elevation_deg, slant_range_km=slant_range_km, is_active=True)
        if not eligible:
            return new_beam

        # Per-user SINR & PF scoring.
        for entry in eligible:
            u = entry["user"]
            dist_from_center_km = haversine_distance(
                u.current_lat, u.current_lon, hex_lat, hex_lon) / 1000.0
            user_theta_deg = math.degrees(math.atan2(dist_from_center_km, slant_range_km))
            roll_off_db = min(12.0 * (user_theta_deg / theta_3db) ** 2, sll)
            effective_eirp_dbw = base_eirp_dbw - roll_off_db

            sinr_ntn_db, capacity_mbps, spec_eff = calculate_ntn_sinr_capacity(
                slant_range_km=slant_range_km,
                off_axis_angles_deg=off_axis_angles_interferers,
                eirp_dbw=effective_eirp_dbw, g_t_db=g_t_db, freq_ghz=f_ntn,
                bandwidth_hz=bw_ntn, theta_3db_deg=theta_3db, sll_db=sll)
            u.ntn_sinr_db = sinr_ntn_db  
            u.spectral_efficiency = spec_eff
            u.achievable_rate_mbps = (bw_ntn * spec_eff) / 1e6
            if sinr_ntn_db < sinr_min_ntn or spec_eff <= 0.0:
                u.pf_score = -1.0
                u.ntn_reason = f"NTN SINR too low ({sinr_ntn_db:.1f} dB)"
                u.ntn_eval_beam = f"Sat_{sat.satellite_id}"
            else:
                u.pf_score = u.achievable_rate_mbps / max(0.1, getattr(u, 'historical_avg_mbps', 0.1))

        eligible.sort(key=lambda x: x["user"].pf_score, reverse=True)

        # Bandwidth exhaustion on this one beam.
        remaining_beam_hz = bw_ntn
        for entry in eligible:
            u = entry["user"]
            if u.pf_score < 0:
                u.current_state = "DROPPED"
                continue

            u.ntn_eval_beam = f"Sat_{sat.satellite_id}"
            u.ntn_eval_hz = remaining_beam_hz

            if remaining_beam_hz <= 0:
                u.ntn_reason = "NTN Beam Congested (Empty)"
                u.current_state = "DROPPED"
                continue

            demand_mbps = entry["unmet_mbps"]
            required_hz = (demand_mbps * 1e6) / u.spectral_efficiency
            min_qos_hz  = (u.qos_min_mbps * 1e6) / u.spectral_efficiency

            if remaining_beam_hz >= min_qos_hz:
                allocated_hz = min(required_hz, remaining_beam_hz)
                remaining_beam_hz -= allocated_hz
                served = (allocated_hz * u.spectral_efficiency) / 1e6
                entry["unmet_mbps"] -= served
                u.served_mbps += served
                new_beam.allocated_mbps += served
                new_beam.active_users += 1
                if entry["unmet_mbps"] <= 0.1:
                    u.current_state = "LEO"
                    u.ntn_reason = "Fully Served"
                else:
                    u.current_state = "DROPPED"
                    u.ntn_reason = "Partially Served (Congested)"
            else:
                u.current_state = "DROPPED"
                u.ntn_reason = f"NTN Bandwidth too low for QoS (Req: {min_qos_hz/1e6:.3f} MHz)"

            u.historical_avg_mbps = (0.8 * getattr(u, 'historical_avg_mbps', 0.1)) + (0.2 * u.served_mbps)

        return new_beam

    # ── 6. PASS 1 — COVERAGE: one beam per hex, MOST-CONSTRAINED FIRST ───────
    # hexes that fail to attach this hex's served satellite (because nothing is
    # visible at all) get the correct, distinct reason labels.
    served_sat_for_hex: Dict[str, Set[str]] = {}
    coverage_order = sorted(hex_needs, key=lambda n: len(hex_vis[n["hex_id"]]))

    for needy in coverage_order:
        hid = needy["hex_id"]
        if not hex_vis[hid]:
            for entry in needy["users"]:
                entry["user"].ntn_reason = "No Satellite Overhead"
            continue
        sat, rec = _pick_sat(hid)
        if sat is None:
            # Visible satellites exist but all are beam-saturated.
            for entry in needy["users"]:
                if entry["unmet_mbps"] > 0.1:
                    entry["user"].ntn_reason = "All Visible Satellites Beam-Saturated"
            continue
        beam = _serve_hex(needy, sat, rec)
        if beam.active_users > 0:
            sat.active_beams.append(beam)
            all_active_beams.append(beam)
            served_sat_for_hex.setdefault(hid, set()).add(sat.satellite_id)

    # ── 7. PASS 2 — CAPACITY: spare beams to still-unmet hexes ───────────────
    # One beam per (sat, hex): a hex may get a SECOND beam only from a DIFFERENT
    # satellite it has not already used. Highest remaining unmet first.
    def _remaining_unmet(n):
        return sum(e["unmet_mbps"] for e in n["users"] if e["unmet_mbps"] > 0.1)

    still = sorted(
        (n for n in hex_needs if _remaining_unmet(n) > 0.1),
        key=_remaining_unmet, reverse=True)

    for needy in still:
        hid = needy["hex_id"]
        used = served_sat_for_hex.get(hid, set())
        sat, rec = _pick_sat(hid, exclude_ids=used)
        if sat is None:
            continue
        beam = _serve_hex(needy, sat, rec)
        if beam.active_users > 0:
            sat.active_beams.append(beam)
            all_active_beams.append(beam)
            served_sat_for_hex.setdefault(hid, set()).add(sat.satellite_id)

    return all_active_beams