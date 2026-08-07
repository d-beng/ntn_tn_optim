#!/usr/bin/env python3
"""
plot_deployment.py — deployment figures from the artefacts the smoke test
already writes. No re-run required.

INPUTS (all optional except --bs):
  --bs      base_stations.csv      written by the simulator every run
                                   (bs_id, site_id, lat, lon, scenario,
                                    sector_azimuth_deg, coverage_radius_km, ...)
                                   OR a milp_placement_<hex>.csv (lat, lon, tier)
  --util    site_utilisation.csv   per-site load, for the utilisation panel
  --users   users.pkl              for the demand heat map (slow: ~8 min load)
  --hour    20.0                   which hour to evaluate demand at
  --sample  200000                 users to keep for plotting (3.5M is not
                                   plottable; the heat map is computed on the
                                   SAMPLE and rescaled, the counts printed are
                                   full-population)

OUTPUT
  <prefix>_deployment.png   4 panels:
     A  demand heat map + sites coloured by tier
     B  sites coloured by realised utilisation
     C  zoom on the densest square + coverage discs
     D  tier composition, utilisation CDF, site-spacing histogram

  <prefix>_sites.csv        one row per SITE (not sector) with tier, lat, lon,
                            utilisation -- convenient for the paper's tables.

USAGE
  python plot_deployment.py --bs base_stations.csv --util site_utilisation.csv \
      --users /Utilisateurs/dbenguer/ntn_tn_optim/data/users.pkl \
      --prefix run_res10_safety1

  # fast version, no users
  python plot_deployment.py --bs base_stations.csv --util site_utilisation.csv
"""
from __future__ import annotations
import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

TIER_COLOR = {"RMA": "#C1666B", "UMA": "#E8A33D", "UMI": "#3D7EA6",
              "UMI_MMW": "#6B4E9B"}
TIER_ORDER = ["RMA", "UMA", "UMI", "UMI_MMW"]
KM_PER_DEG_LAT = 111.32


def km_per_deg_lon(lat):
    return 111.32 * math.cos(math.radians(lat))


# ---------------------------------------------------------------------------
def load_sites(path):
    """Collapse a sector-level base_stations.csv to one row per SITE.
    Also accepts a milp_placement CSV (lat, lon, tier)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    tier_col = cols.get("scenario") or cols.get("tier")
    if tier_col is None:
        raise SystemExit(f"{path}: no 'scenario' or 'tier' column "
                         f"(found {list(df.columns)})")

    if "site_id" in cols:
        g = df.groupby(cols["site_id"])
        sites = pd.DataFrame({
            "site_id": list(g.groups.keys()),
            "lat": g[cols["lat"]].first().values,
            "lon": g[cols["lon"]].first().values,
            "tier": g[tier_col].first().values,
            "n_sectors": g.size().values,
        })
        if "coverage_radius_km" in cols:
            sites["radius_km"] = g[cols["coverage_radius_km"]].first().values
    else:
        sites = pd.DataFrame({
            "site_id": np.arange(len(df)),
            "lat": df[cols["lat"]].values,
            "lon": df[cols["lon"]].values,
            "tier": df[tier_col].values,
            "n_sectors": 3,
        })
    if "radius_km" not in sites:
        default_r = {"RMA": 2.847, "UMA": 0.474, "UMI": 0.311, "UMI_MMW": 0.150}
        sites["radius_km"] = sites["tier"].map(default_r).fillna(0.5)
    return sites


def load_utilisation(path, sites):
    """Attach per-site utilisation. Column names vary, so search for anything
    that looks like a utilisation fraction."""
    if not path or not os.path.exists(path):
        return sites
    u = pd.read_csv(path)
    low = {c.lower(): c for c in u.columns}
    id_col = next((low[k] for k in ("site_id", "bs_id", "id") if k in low), None)
    util_col = next((low[k] for k in low
                     if "util" in k or "load" in k or "occupanc" in k), None)
    if id_col is None or util_col is None:
        print(f"  (utilisation: could not find id/util columns in "
              f"{list(u.columns)} -- skipping)")
        return sites
    vals = pd.to_numeric(u[util_col], errors="coerce")
    if vals.max() is not np.nan and vals.max() > 1.5:      # percent, not frac
        vals = vals / 100.0
    m = dict(zip(u[id_col].values, vals.values))
    sites["util"] = sites["site_id"].map(m)
    got = sites["util"].notna().sum()
    print(f"  utilisation matched on {got:,}/{len(sites):,} sites "
          f"(column '{util_col}')")
    return sites


def load_users(path, hour, sample, seed=0):
    print(f"  loading {path} (this is the slow part) ...", flush=True)
    t0 = time.time()
    with open(path, "rb") as f:
        users = pickle.load(f)
    print(f"    {len(users):,} users in {time.time()-t0:.0f}s", flush=True)
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(users), min(sample, len(users)), replace=False)
           if sample and sample < len(users) else np.arange(len(users)))
    lat, lon, dem = [], [], []
    for i in idx:
        u = users[i]
        try:
            u.move(hour, 5)
        except Exception:
            pass
        lat.append(float(getattr(u, "current_lat", u.home_lat)))
        lon.append(float(getattr(u, "current_lon", u.home_lon)))
        try:
            dem.append(float(u.get_demand_at_time(hour)))
        except Exception:
            dem.append(1.0)
    scale = len(users) / len(idx)
    del users
    return (np.array(lat), np.array(lon), np.array(dem), scale)


# ---------------------------------------------------------------------------
def make_figure(sites, users, prefix, title):
    have_users = users is not None
    if have_users:
        ulat, ulon, udem, uscale = users

    lat0 = float(sites["lat"].mean())
    aspect = KM_PER_DEG_LAT / km_per_deg_lon(lat0)

    # plotting extent from the sites (pad 5%)
    pad_la = 0.05 * (sites["lat"].max() - sites["lat"].min() + 1e-6)
    pad_lo = 0.05 * (sites["lon"].max() - sites["lon"].min() + 1e-6)
    ext = (sites["lon"].min() - pad_lo, sites["lon"].max() + pad_lo,
           sites["lat"].min() - pad_la, sites["lat"].max() + pad_la)

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.18)

    # ---------- A: demand heat map + sites by tier ----------
    axA = fig.add_subplot(gs[0, 0])
    if have_users:
        m = ((ulon >= ext[0]) & (ulon <= ext[1]) &
             (ulat >= ext[2]) & (ulat <= ext[3]))
        hb = axA.hexbin(ulon[m], ulat[m], C=udem[m], reduce_C_function=np.sum,
                        gridsize=110, cmap="YlOrRd", mincnt=1, bins="log",
                        linewidths=0.0, zorder=1, alpha=0.95)
        cb = fig.colorbar(hb, ax=axA, fraction=0.045, pad=0.02)
        cb.set_label(f"offered demand [Mbps/cell, log scale, "
                     f"{uscale:.0f}x sample]", fontsize=8)
    for t in TIER_ORDER:
        s = sites[sites.tier == t]
        if len(s) == 0:
            continue
        axA.scatter(s.lon, s.lat, s=2.0 if t == "UMI" else 22,
                    c="#111111" if t == "UMI" else TIER_COLOR[t],
                    marker="." if t == "UMI" else "^",
                    label=f"{t} ({len(s):,})", edgecolors="none",
                    alpha=0.55 if t == "UMI" else 0.95, zorder=3)
    axA.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axA.set_title("A  offered demand and deployed sites", fontsize=11, loc="left")

    # ---------- B: sites by utilisation ----------
    axB = fig.add_subplot(gs[0, 1])
    if "util" in sites and sites["util"].notna().any():
        s = sites.dropna(subset=["util"])
        sc = axB.scatter(s.lon, s.lat, c=s["util"], s=9, cmap="RdYlGn_r",
                         vmin=0, vmax=1, edgecolors="none")
        cb = fig.colorbar(sc, ax=axB, fraction=0.045, pad=0.02)
        cb.set_label("realised site utilisation", fontsize=8)
        hot = (s["util"] >= 0.9).sum()
        axB.set_title(f"B  utilisation  (mean {s['util'].mean():.1%}, "
                      f"{hot:,} sites >=90%)", fontsize=11, loc="left")
    else:
        axB.scatter(sites.lon, sites.lat, s=6, c="#3D7EA6", edgecolors="none")
        axB.set_title("B  sites (no utilisation file supplied)",
                      fontsize=11, loc="left")

    # ---------- C: zoom on the densest square, with coverage discs ----------
    axC = fig.add_subplot(gs[1, 0])
    # densest 2 km x 2 km window, by site count
    half_la = 1.0 / KM_PER_DEG_LAT
    half_lo = 1.0 / km_per_deg_lon(lat0)
    best, bc = None, -1
    for _, r in sites.sample(min(400, len(sites)), random_state=0).iterrows():
        n = ((sites.lat.between(r.lat - half_la, r.lat + half_la)) &
             (sites.lon.between(r.lon - half_lo, r.lon + half_lo))).sum()
        if n > bc:
            bc, best = n, (r.lat, r.lon)
    zlat, zlon = best
    zext = (zlon - half_lo, zlon + half_lo, zlat - half_la, zlat + half_la)
    if have_users:
        m = ((ulon >= zext[0]) & (ulon <= zext[1]) &
             (ulat >= zext[2]) & (ulat <= zext[3]))
        axC.scatter(ulon[m], ulat[m], s=1.0, c="#666666", alpha=0.22,
                    edgecolors="none", zorder=1,
                    label=f"users ({m.sum()*uscale:,.0f} full-scale)")
    zs = sites[sites.lat.between(zext[2], zext[3]) &
               sites.lon.between(zext[0], zext[1])]
    patches, colors = [], []
    for _, r in zs.iterrows():
        patches.append(Circle((r.lon, r.lat),
                              r.radius_km / km_per_deg_lon(lat0)))
        colors.append(TIER_COLOR.get(r.tier, "#666666"))
    if patches:
        axC.add_collection(PatchCollection(patches, facecolors="none",
                                           edgecolors=colors, alpha=0.45,
                                           linewidths=0.7, zorder=2))
    for t in TIER_ORDER:
        s = zs[zs.tier == t]
        if len(s):
            axC.scatter(s.lon, s.lat, s=34, c=TIER_COLOR[t],
                        marker="^", edgecolors="white", linewidths=0.5,
                        zorder=4, label=f"{t}")
    axC.set_xlim(zext[0], zext[1]); axC.set_ylim(zext[2], zext[3])
    axC.legend(loc="upper right", fontsize=7, framealpha=0.9)
    axC.set_title(f"C  densest 2x2 km  ({len(zs)} sites, coverage discs to "
                  f"scale)", fontsize=11, loc="left")

    # ---------- D: distributions ----------
    axD = fig.add_subplot(gs[1, 1])
    axD.axis("off")
    sub = axD.inset_axes([0.0, 0.55, 0.42, 0.42])
    counts = [len(sites[sites.tier == t]) for t in TIER_ORDER]
    keep = [(t, c) for t, c in zip(TIER_ORDER, counts) if c]
    sub.bar(np.arange(len(keep)), [c for _, c in keep],
            color=[TIER_COLOR[t] for t, _ in keep])
    sub.set_xticks(np.arange(len(keep)))
    sub.set_xticklabels([t for t, _ in keep])
    sub.set_title("sites per tier", fontsize=8)
    sub.tick_params(labelsize=7)
    for i, (_, c) in enumerate(keep):
        sub.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=7)

    if "util" in sites and sites["util"].notna().any():
        sub2 = axD.inset_axes([0.56, 0.55, 0.42, 0.42])
        v = np.sort(sites["util"].dropna().values)
        sub2.plot(v, np.linspace(0, 1, len(v)), color="#3D7EA6")
        sub2.axvline(0.9, ls="--", lw=0.8, color="#C1666B")
        sub2.set_title("utilisation CDF", fontsize=8)
        sub2.set_xlabel("utilisation", fontsize=7)
        sub2.tick_params(labelsize=7)

    # nearest-neighbour spacing, per tier
    sub3 = axD.inset_axes([0.0, 0.02, 0.98, 0.42])
    from scipy.spatial import cKDTree
    for t in TIER_ORDER:
        s = sites[sites.tier == t]
        if len(s) < 3:
            continue
        xy = np.column_stack([s.lon * km_per_deg_lon(lat0),
                              s.lat * KM_PER_DEG_LAT])
        d, _ = cKDTree(xy).query(xy, k=2)
        v = d[:, 1] * 1000
        sub3.hist(v, bins=np.logspace(np.log10(max(v.min(), 5)),
                                      np.log10(v.max() + 1), 55),
                  alpha=0.65, color=TIER_COLOR[t],
                  label=f"{t} (median {np.median(v):.0f} m)")
    sub3.set_xscale("log")
    sub3.set_xlabel("nearest-neighbour site spacing [m], log scale", fontsize=8)
    sub3.legend(fontsize=7)
    sub3.tick_params(labelsize=7)
    sub3.set_title("deployed inter-site distance", fontsize=8)

    for ax in (axA, axB, axC):
        ax.set_aspect(aspect)
        ax.set_xlabel("longitude", fontsize=8)
        ax.set_ylabel("latitude", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in (axA, axB):
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])

    fig.suptitle(title, fontsize=13, y=0.965)
    out = f"{prefix}_deployment.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"-> {out}")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", default="base_stations.csv")
    ap.add_argument("--util", default="site_utilisation.csv")
    ap.add_argument("--users", default=None)
    ap.add_argument("--hour", type=float, default=20.0)
    ap.add_argument("--sample", type=int, default=200000)
    ap.add_argument("--prefix", default="deployment")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    print(f"[1] sites from {args.bs}")
    sites = load_sites(args.bs)
    mix = sites.tier.value_counts().to_dict()
    print(f"    {len(sites):,} sites: {mix}")

    print(f"[2] utilisation from {args.util}")
    sites = load_utilisation(args.util, sites)

    users = None
    if args.users:
        print(f"[3] users from {args.users}")
        users = load_users(args.users, args.hour, args.sample)
    else:
        print("[3] no --users given: demand heat map skipped")

    title = args.title or (f"{len(sites):,} sites  |  " +
                           "  ".join(f"{k} {v:,}" for k, v in mix.items()))
    make_figure(sites, users, args.prefix, title)

    out_csv = f"{args.prefix}_sites.csv"
    sites.to_csv(out_csv, index=False)
    print(f"-> {out_csv}")


if __name__ == "__main__":
    main()