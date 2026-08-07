#!/usr/bin/env python3
"""
plot_drops.py — WHERE the simulator lost users, plotted against the network
that was actually built.

Dropped users only exist in memory, after run_single_hour() has mutated the
User objects. So this is called from INSIDE smoke_real_sim.py, right after the
tn_reason census in [D]:

    from plot_drops import export_and_plot_drops
    export_and_plot_drops(hex_users, bss, prefix=f"drops_{hex_id[:8]}")

It writes  <prefix>_users.csv  (one row per SHORT user) and
           <prefix>.png       (4 panels), and can be re-run standalone later:

    python plot_drops.py --csv drops_852b9bd7_users.csv \
                         --bs base_stations.csv --prefix drops_852b9bd7

WHY PANEL D IS THE POINT
------------------------
It plots, per drop reason, the distance from the user to the NEAREST deployed
site. The census labels are only as good as the simulator's own bookkeeping;
this checks them against geometry:

  * COVERAGE drops ("No 5G Tower in Geographic Range") must sit BEYOND the
    serving tier's radius. If they do not, the label is wrong and the real
    cause is admission control, not geometry.
  * CONGESTION drops ("Tower Empty") must sit well INSIDE a cell. If they do,
    the user had a tower in range and still got nothing -- that is the
    association gap, and no extra site fixes it.

That one histogram separates "build more" from "steer better", which is the
question the whole residual turns on.
"""
from __future__ import annotations
import argparse
import math

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KM_DEG_LAT = 111.32
TIER_COLOR = {"RMA": "#C1666B", "UMA": "#E8A33D", "UMI": "#3D7EA6",
              "UMI_MMW": "#6B4E9B"}
REASON_COLOR = {
    "COVERAGE":   "#B23A48",   # no tower in range    -> geometry
    "CONGESTION": "#E8A33D",   # tower empty / PF     -> association
    "QOS":        "#4C956C",   # bandwidth < QoS min  -> admission granularity
    "SINR":       "#6B4E9B",   # link too weak        -> interference
    "NTN":        "#2C6E9B",   # satellite side
    "OTHER":      "#888888",
}


def km_per_deg_lon(lat):
    return 111.32 * math.cos(math.radians(lat))


def classify(reason: str) -> str:
    r = str(reason or "").lower()
    if "no 5g tower" in r or "geographic range" in r:
        return "COVERAGE"
    if "congestion" in r or "tower empty" in r or "partially served" in r:
        return "CONGESTION"
    if "bandwidth too low" in r or "qos" in r:
        return "QOS"
    if "sinr" in r:
        return "SINR"
    if r.startswith("ntn") or "beam" in r:
        return "NTN"
    return "OTHER"


# ---------------------------------------------------------------------------
def collect_drops(hex_users) -> pd.DataFrame:
    """One row per user who did NOT receive its full demand."""
    from full_pipeline_hooks import user_result_fields
    lat, lon, short, reason, dem, got = [], [], [], [], [], []
    for u in hex_users:
        _bs, _se, g, d, _dropped = user_result_fields(u)
        if g + 1e-9 >= d:
            continue
        lat.append(float(getattr(u, "current_lat", 0.0)))
        lon.append(float(getattr(u, "current_lon", 0.0)))
        short.append(d - g)
        dem.append(d)
        got.append(g)
        rtn = str(getattr(u, "tn_reason", "") or "")
        rntn = str(getattr(u, "ntn_reason", "") or "")
        reason.append(rtn if rtn and rtn != "N/A" else rntn)
    df = pd.DataFrame({"lat": lat, "lon": lon, "demand_mbps": dem,
                       "served_mbps": got, "short_mbps": short,
                       "reason": reason})
    df["cause"] = df["reason"].map(classify)
    return df


def sites_from_bss(bss) -> pd.DataFrame:
    """BaseStation objects -> one row per SITE (they arrive in 3-sector groups)."""
    rows = {}
    for b in bss:
        sid = int(getattr(b, "site_id", b.bs_id))
        if sid in rows:
            continue
        sc = getattr(b, "scenario", None)
        rows[sid] = {
            "site_id": sid, "lat": float(b.lat), "lon": float(b.lon),
            "tier": getattr(sc, "name", str(sc)),
            "radius_km": float(getattr(b, "coverage_radius_km", 0.5)),
        }
    return pd.DataFrame(list(rows.values()))


def sites_from_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    tier = low.get("scenario") or low.get("tier")
    if "site_id" in low:
        g = df.groupby(low["site_id"])
        out = pd.DataFrame({
            "site_id": list(g.groups.keys()),
            "lat": g[low["lat"]].first().values,
            "lon": g[low["lon"]].first().values,
            "tier": g[tier].first().values,
        })
        if "coverage_radius_km" in low:
            out["radius_km"] = g[low["coverage_radius_km"]].first().values
    else:
        out = pd.DataFrame({"site_id": np.arange(len(df)),
                            "lat": df[low["lat"]].values,
                            "lon": df[low["lon"]].values,
                            "tier": df[tier].values})
    if "radius_km" not in out:
        out["radius_km"] = out["tier"].map(
            {"RMA": 2.847, "UMA": 0.474, "UMI": 0.311}).fillna(0.5)
    return out


# ---------------------------------------------------------------------------
def plot_drops(drops: pd.DataFrame, sites: pd.DataFrame, prefix: str,
               max_scatter: int = 60000, title: str | None = None):
    if len(drops) == 0:
        print("  no dropped users -- nothing to plot")
        return None
    lat0 = float(sites["lat"].mean())
    aspect = KM_DEG_LAT / km_per_deg_lon(lat0)
    kx, ky = km_per_deg_lon(lat0), KM_DEG_LAT

    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([sites.lon * kx, sites.lat * ky]))
    d_km, nn = tree.query(np.column_stack([drops.lon * kx, drops.lat * ky]))
    drops = drops.assign(dist_km=d_km, near_tier=sites.tier.values[nn])

    by = (drops.groupby("cause")["short_mbps"].agg(["sum", "count"])
          .sort_values("sum", ascending=False))
    order = list(by.index)

    pad = 0.03
    ext = (sites.lon.min() - pad, sites.lon.max() + pad,
           sites.lat.min() - pad, sites.lat.max() + pad)

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.24, wspace=0.18)

    # ---- A: sites + dropped users coloured by cause ----
    axA = fig.add_subplot(gs[0, 0])
    axA.scatter(sites.lon, sites.lat, s=1.6, c="#BBBBBB", marker=".",
                edgecolors="none", zorder=1, label=f"sites ({len(sites):,})")
    d = drops.sample(min(max_scatter, len(drops)), random_state=0)
    for c in order:
        s = d[d.cause == c]
        if len(s) == 0:
            continue
        axA.scatter(s.lon, s.lat, s=2.2, c=REASON_COLOR.get(c, "#888"),
                    edgecolors="none", alpha=0.45, zorder=3,
                    label=f"{c} ({by.loc[c, 'count']:,} u, "
                          f"{by.loc[c, 'sum']/1e3:.1f} Gbps)")
    axA.legend(loc="upper right", fontsize=7, framealpha=0.92, markerscale=4)
    axA.set_title("A  dropped users over the deployed network",
                  fontsize=11, loc="left")

    # ---- B: heat map of LOST Mbps ----
    axB = fig.add_subplot(gs[0, 1])
    hb = axB.hexbin(drops.lon, drops.lat, C=drops.short_mbps,
                    reduce_C_function=np.sum, gridsize=100, cmap="inferno_r",
                    mincnt=1, bins="log", linewidths=0.0)
    cb = fig.colorbar(hb, ax=axB, fraction=0.045, pad=0.02)
    cb.set_label("lost throughput [Mbps per cell, log]", fontsize=8)
    axB.scatter(sites.lon, sites.lat, s=0.8, c="#3D7EA6", marker=".",
                edgecolors="none", alpha=0.5)
    axB.set_title(f"B  where the {drops.short_mbps.sum()/1e3:,.1f} Gbps is lost",
                  fontsize=11, loc="left")

    # ---- C: zoom on the worst hexbin cell ----
    axC = fig.add_subplot(gs[1, 0])
    offs, vals = hb.get_offsets(), np.asarray(hb.get_array())
    zlon, zlat = offs[int(np.argmax(vals))]
    hla, hlo = 0.75 / ky, 0.75 / kx                  # 1.5 km x 1.5 km
    zext = (zlon - hlo, zlon + hlo, zlat - hla, zlat + hla)
    zs = sites[sites.lat.between(zext[2], zext[3]) &
               sites.lon.between(zext[0], zext[1])]
    zd = drops[drops.lat.between(zext[2], zext[3]) &
               drops.lon.between(zext[0], zext[1])]
    for t, g in zs.groupby("tier"):
        axC.scatter(g.lon, g.lat, s=90, marker="^",
                    c=TIER_COLOR.get(t, "#666"), edgecolors="white",
                    linewidths=0.8, zorder=5, label=f"{t} site")
    for c in order:
        s = zd[zd.cause == c]
        if len(s) == 0:
            continue
        axC.scatter(s.lon, s.lat, s=7, c=REASON_COLOR.get(c, "#888"),
                    edgecolors="none", alpha=0.6, zorder=3, label=c)
    axC.set_xlim(zext[0], zext[1]); axC.set_ylim(zext[2], zext[3])
    axC.legend(loc="upper right", fontsize=7, framealpha=0.92, markerscale=1.6)
    axC.set_title(f"C  worst 1.5x1.5 km  ({len(zd):,} dropped users, "
                  f"{zd.short_mbps.sum()/1e3:.2f} Gbps, {len(zs)} sites)",
                  fontsize=11, loc="left")

    # ---- D: distance to nearest site, per cause (the decisive panel) ----
    axD = fig.add_subplot(gs[1, 1])
    for c in order:
        s = drops[drops.cause == c]
        if len(s) < 20:
            continue
        axD.hist(s.dist_km * 1000, bins=np.logspace(0.7, 4.2, 70),
                 alpha=0.6, color=REASON_COLOR.get(c, "#888"),
                 label=f"{c}  median {1000*s.dist_km.median():.0f} m")
    ymax = axD.get_ylim()[1]
    for t, r in (("UMI", 0.311), ("UMA", 0.474), ("RMA", 2.847)):
        if (sites.tier == t).any():
            axD.axvline(1000 * r, ls="--", lw=0.9, color=TIER_COLOR[t])
            axD.text(1000 * r, ymax * 0.95, f" {t} R", fontsize=7,
                     color=TIER_COLOR[t], rotation=90, va="top")
    axD.set_xscale("log")
    axD.set_xlabel("distance from dropped user to NEAREST deployed site [m]",
                   fontsize=9)
    axD.set_ylabel("users", fontsize=9)
    axD.legend(fontsize=8)
    axD.set_title("D  COVERAGE drops must fall BEYOND the tier radius;\n"
                  "     CONGESTION drops falling INSIDE it are an association\n"
                  "     gap, not a CAPEX gap", fontsize=10, loc="left")

    for ax in (axA, axB):
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    for ax in (axA, axB, axC):
        ax.set_aspect(aspect)
        ax.set_xlabel("longitude", fontsize=8)
        ax.set_ylabel("latitude", fontsize=8)
        ax.tick_params(labelsize=7)

    tot = drops.short_mbps.sum()
    fig.suptitle(title or (f"{len(drops):,} short users, {tot/1e3:,.1f} Gbps "
                           f"lost, {len(sites):,} sites"), fontsize=13, y=0.965)
    out = f"{prefix}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")

    print("  cause | Gbps lost | users | median distance to nearest site")
    for c in order:
        s = drops[drops.cause == c]
        print(f"      {c:<11} {s.short_mbps.sum()/1e3:>7.2f} | "
              f"{len(s):>8,} | {1000*s.dist_km.median():>7.0f} m")
    return out


# ---------------------------------------------------------------------------
def export_and_plot_drops(hex_users, bss, prefix="drops", title=None):
    """Call from smoke_real_sim.py right after the tn_reason census."""
    drops = collect_drops(hex_users)
    sites = sites_from_bss(bss)
    csv = f"{prefix}_users.csv"
    drops.to_csv(csv, index=False)
    print(f"  -> {csv}  ({len(drops):,} short users)")
    try:
        plot_drops(drops, sites, prefix, title=title)
    except Exception as e:
        print(f"  (drop plot skipped: {e})")
    return drops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="<prefix>_users.csv")
    ap.add_argument("--bs", default="base_stations.csv")
    ap.add_argument("--prefix", default="drops")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    drops = pd.read_csv(a.csv)
    if "cause" not in drops.columns:
        drops["cause"] = drops["reason"].map(classify)
    sites = sites_from_csv(a.bs)
    plot_drops(drops, sites, a.prefix, title=a.title)


if __name__ == "__main__":
    main()