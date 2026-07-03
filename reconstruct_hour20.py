"""
Reconstructs the Hour 20 view of Final_Animation.html as a lightweight
interactive map showing ONLY dropped users + deployed base stations.

Sources:
  Final_Animation.html  -> extracts the 64,976 base station positions & tiers
  detailed_drop_log.csv -> provides dropped user lat/lon & TN reason at hour 20

Output: hour20_drops.html  (~5-15 MB, opens instantly in any browser)

Usage:
    python reconstruct_hour20.py
    python reconstruct_hour20.py --animation Final_Animation.html \
                                  --log detailed_drop_log.csv \
                                  --out hour20_drops.html \
                                  --max-users 300000
"""
import argparse, re, json, os, sys
import numpy as np
import pandas as pd

try:
    import folium
    from folium.plugins import FastMarkerCluster
except ImportError:
    sys.exit("folium required:  pip install folium")


# ── tier colours ─────────────────────────────────────────────────────────────
BS_COLOUR  = {"UMI": "#ef4444", "UMA": "#3b82f6", "RMA": "#22c55e"}
DROP_COLOUR = {
    "No 5G Tower in Geographic Range": "#dc2626",
    "5G Bandwidth too low for QoS":    "#f97316",
    "5G Congestion (Tower Empty)":      "#f59e0b",
    "5G SINR too low":                  "#a855f7",
}
DEFAULT_DROP = "#fb923c"


def _tier(r):
    if r is None or np.isnan(float(r)):  return "UMI"
    r = float(r)
    if r < 0.4:  return "UMI"
    if r < 1.0:  return "UMA"
    return "RMA"


def _drop_col(reason: str) -> str:
    for k, c in DROP_COLOUR.items():
        if str(reason).startswith(k):
            return c
    return DEFAULT_DROP


# ── extract towers from animation HTML ───────────────────────────────────────
def extract_towers(html_path: str) -> pd.DataFrame:
    print(f"  Extracting tower positions from {html_path} …", flush=True)
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Tower positions: first vertex of each coverage polygon in GeoJSON layers
    pos_pat  = r'"id":"(BS_\d+)","geometry":\{"type":"Polygon","coordinates":\[\[\[([+-]?\d+\.\d+),([+-]?\d+\.\d+)'
    pos_m    = re.findall(pos_pat, content)

    # Tower radius (for tier inference): from the scatter trace text labels
    txt_pat  = r'"Tower (\d+)\\u003cbr\\u003eRadius: ([0-9.]+) km"'
    txt_m    = re.findall(txt_pat, content)
    radius_map = {int(n): float(r) for n, r in txt_m}

    print(f"  Found {len(pos_m)} tower polygons, {len(radius_map)} radius labels.", flush=True)

    rows = []
    for bs_id, lon_s, lat_s in pos_m:
        n = int(bs_id.replace("BS_", ""))
        r = radius_map.get(n)
        rows.append({"bs_id": bs_id, "lat": float(lat_s), "lon": float(lon_s),
                     "radius_km": r, "tier": _tier(r)})

    df = pd.DataFrame(rows)
    # De-duplicate co-located sectors (same lat/lon → one dot per site)
    df = df.drop_duplicates(subset=["lat", "lon"])
    print(f"  → {len(df):,} unique tower sites (sectors merged).", flush=True)
    return df


# ── load dropped users from log ───────────────────────────────────────────────
def load_drops(log_path: str, hour: float, max_users: int) -> pd.DataFrame:
    print(f"  Loading {log_path} …", flush=True)
    cols = ["Hour", "Lat", "Lon", "TN_Reason", "NTN_Reason",
            "Final_State", "Demand_Mbps"]
    avail = pd.read_csv(log_path, nrows=0).columns.tolist()
    use   = [c for c in cols if c in avail]
    df    = pd.read_csv(log_path, usecols=use)

    df = df[df["Hour"] == hour]
    df = df[df["Final_State"] == "DROPPED"]
    print(f"  → {len(df):,} dropped users at hour {hour}.", flush=True)

    if len(df) > max_users:
        df = df.sample(n=max_users, random_state=42)
        print(f"  Sub-sampled to {max_users:,} for rendering.", flush=True)
    return df.reset_index(drop=True)


# ── build the map ─────────────────────────────────────────────────────────────
def build_map(towers: pd.DataFrame, drops: pd.DataFrame, out: str,
              max_tower_dots: int) -> None:

    clat = 45.5; clon = -81.0
    if len(drops):
        clat = float(drops["Lat"].mean())
        clon = float(drops["Lon"].mean())

    m = folium.Map(location=[clat, clon], zoom_start=6,
                   tiles="CartoDB dark_matter",
                   prefer_canvas=True)

    # ── BASE STATIONS ─────────────────────────────────────────────────────────
    print(f"  Plotting {len(towers):,} tower sites …", flush=True)
    if len(towers) > max_tower_dots:
        towers_plot = towers.sample(n=max_tower_dots, random_state=0)
        print(f"  Sub-sampled towers to {max_tower_dots:,}.", flush=True)
    else:
        towers_plot = towers

    for tier, col, r_px in [("UMI","#ef4444",3), ("UMA","#3b82f6",4), ("RMA","#22c55e",5)]:
        sub = towers_plot[towers_plot["tier"] == tier]
        if len(sub) == 0: continue
        fg = folium.FeatureGroup(name=f"🔵 BS {tier} ({len(sub):,})", show=True)
        for _, row in sub.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=r_px, color=col, weight=0,
                fill=True, fill_color=col, fill_opacity=0.8,
                tooltip=f"{tier} – {row.get('radius_km', '?')} km"
            ).add_to(fg)
        fg.add_to(m)

    # ── DROPPED USERS ─────────────────────────────────────────────────────────
    print(f"  Plotting {len(drops):,} dropped users …", flush=True)

    cats = {
        "🔴 No Coverage (No TN Tower)":
            drops[drops["TN_Reason"].str.startswith("No 5G Tower", na=False)],
        "🟠 TN Bandwidth Exhausted":
            drops[drops["TN_Reason"].str.startswith("5G Bandwidth", na=False)],
        "🟡 TN Tower Fully Empty":
            drops[drops["TN_Reason"].str.startswith("5G Congestion", na=False)],
        "🟣 TN SINR Too Low":
            drops[drops["TN_Reason"].str.startswith("5G SINR", na=False)],
        "⚫ Other Drop":
            drops[~drops["TN_Reason"].str.startswith(
                ("No 5G Tower","5G Bandwidth","5G Congestion","5G SINR"), na=False)],
    }
    cat_cols = {
        "🔴 No Coverage (No TN Tower)":  "#dc2626",
        "🟠 TN Bandwidth Exhausted":      "#f97316",
        "🟡 TN Tower Fully Empty":         "#f59e0b",
        "🟣 TN SINR Too Low":             "#a855f7",
        "⚫ Other Drop":                   "#6b7280",
    }

    for label, sub in cats.items():
        if len(sub) == 0: continue
        col = cat_cols[label]
        fg  = folium.FeatureGroup(name=f"{label} ({len(sub):,})", show=True)
        coords = sub[["Lat","Lon"]].values.tolist()
        if len(coords) > 5_000:
            cb = (
                f"function(row){{"
                f"return L.circleMarker(new L.LatLng(row[0],row[1]),"
                f"{{radius:3,color:'{col}',fillColor:'{col}',"
                f"fillOpacity:0.75,weight:0}});}}"
            )
            FastMarkerCluster(coords, callback=cb).add_to(fg)
        else:
            for lat, lon in coords:
                folium.CircleMarker(
                    [lat, lon], radius=3, color=col, weight=0,
                    fill=True, fill_color=col, fill_opacity=0.75
                ).add_to(fg)
        fg.add_to(m)

    # ── STATS BOX ─────────────────────────────────────────────────────────────
    n_no_tower  = int(drops["TN_Reason"].str.startswith("No 5G Tower", na=False).sum())
    n_congested = int(drops["TN_Reason"].str.startswith("5G Bandwidth", na=False).sum())
    n_total     = len(drops)

    stats_html = f"""
    <div style="position:fixed;top:20px;right:20px;z-index:9999;
                background:rgba(0,0,0,0.82);color:#f1f5f9;
                padding:14px 18px;border-radius:10px;
                font-family:monospace;font-size:12px;line-height:1.8;
                border:1px solid #475569">
      <b style="font-size:14px">Hour 20 — Dropped Users</b><br>
      Total dropped: <b>{n_total:,}</b><br>
      No TN tower:   <b style="color:#f87171">{n_no_tower:,}</b>
                     ({100*n_no_tower/max(n_total,1):.1f}%)<br>
      TN congested:  <b style="color:#fb923c">{n_congested:,}</b>
                     ({100*n_congested/max(n_total,1):.1f}%)<br>
      <hr style="border-color:#475569;margin:6px 0">
      <b>Base Stations</b><br>
      <span style="color:#ef4444">●</span> UMi {len(towers[towers.tier=='UMI']):,} &nbsp;
      <span style="color:#3b82f6">●</span> UMa {len(towers[towers.tier=='UMA']):,} &nbsp;
      <span style="color:#22c55e">●</span> RMa {len(towers[towers.tier=='RMA']):,}
    </div>"""
    m.get_root().html.add_child(folium.Element(stats_html))

    # ── LEGEND ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed;bottom:30px;left:20px;z-index:9999;
                background:rgba(0,0,0,0.82);color:#f1f5f9;
                padding:12px 16px;border-radius:10px;font-size:12px;
                font-family:sans-serif;line-height:1.9;
                border:1px solid #475569">
      <b>Dropped Users — TN Failure Reason</b><br>
      <span style="color:#dc2626;font-size:16px">●</span> No TN coverage<br>
      <span style="color:#f97316;font-size:16px">●</span> TN bandwidth full<br>
      <span style="color:#f59e0b;font-size:16px">●</span> TN tower empty<br>
      <span style="color:#a855f7;font-size:16px">●</span> TN SINR too low<br>
      <hr style="border-color:#475569;margin:6px 0">
      <b>Base Stations</b><br>
      <span style="color:#ef4444;font-size:16px">●</span> UMi small cell<br>
      <span style="color:#3b82f6;font-size:16px">●</span> UMa macro<br>
      <span style="color:#22c55e;font-size:16px">●</span> RMa rural macro
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    print(f"  Saving {out} …", flush=True)
    m.save(out)
    size_mb = os.path.getsize(out) / 1e6
    print(f"  ✅  Saved {out}  ({size_mb:.1f} MB)", flush=True)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animation", default="Final_Animation.html")
    ap.add_argument("--log",       default="detailed_drop_log.csv")
    ap.add_argument("--out",       default="hour20_drops.html")
    ap.add_argument("--hour",      type=float, default=20.0)
    ap.add_argument("--max-users", type=int,   default=300_000)
    ap.add_argument("--max-towers",type=int,   default=40_000)
    args = ap.parse_args()

    print("\n🗺  Hour 20 Drop Map Reconstruction")
    print("=" * 45)

    if not os.path.exists(args.animation):
        sys.exit(f"ERROR: {args.animation} not found.")
    if not os.path.exists(args.log):
        sys.exit(f"ERROR: {args.log} not found.")

    towers = extract_towers(args.animation)
    drops  = load_drops(args.log, args.hour, args.max_users)

    if len(drops) == 0:
        sys.exit("No dropped users found — check the log and --hour value.")

    build_map(towers, drops, args.out, args.max_towers)
    print(f"\n  Open {args.out} in a browser.\n")


if __name__ == "__main__":
    main()