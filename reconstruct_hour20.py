#!/usr/bin/env python3
"""
reconstruct_hour20.py  (v2 — with coverage radius circles)
===========================================================
Reconstructs the Hour-20 snapshot of the hybrid NTN-TN simulation as a
Plotly-based interactive HTML map — same library and dark style as
Final_Animation.html, but:
  • STATIC  (hour 20 only — no 21-frame animation overhead)
  • Dropped users shown as coloured dots by TN failure reason
  • Base-station coverage RADIUS CIRCLES drawn as GeoJSON layers
  • Result: ~30-50 MB, opens instantly in any browser

Data sources (must be in the same directory or passed via --* flags):
  Final_Animation.html   -> 64,976 tower positions + radii extracted directly
  detailed_drop_log.csv  -> dropped user lat/lon + TN/NTN reasons at hour 20

Usage:
  python reconstruct_hour20.py
  python reconstruct_hour20.py --animation path/to/Final_Animation.html \
                                --log path/to/detailed_drop_log.csv \
                                --out hour20_drops.html \
                                --max-users 300000
"""
import argparse, re, os, sys, math, json
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    sys.exit("plotly required:  pip install plotly")


MAPBOX_STYLE = "carto-darkmatter"

TIER_COL = {
    "UMI": "rgba(255,  80,  80, 0.90)",
    "UMA": "rgba( 80, 140, 255, 0.90)",
    "RMA": "rgba( 60, 200,  90, 0.90)",
}
TIER_FILL = {
    "UMI": "rgba(255, 100, 100, 0.18)",
    "UMA": "rgba( 80, 140, 255, 0.15)",
    "RMA": "rgba( 60, 200,  90, 0.12)",
}
DROP_COL = {
    "No 5G Tower in Geographic Range": "#f87171",
    "5G Bandwidth too low for QoS":    "#fb923c",
    "5G Congestion (Tower Empty)":      "#fbbf24",
    "5G SINR too low":                  "#c084fc",
    "Other":                            "#94a3b8",
}


def _circle_lonlat(lat, lon, r_km, n=20):
    cos_lat = math.cos(math.radians(lat))
    pts = []
    for i in range(n + 1):
        a = 2 * math.pi * i / n
        pts.append([lon + (r_km / (111.0 * cos_lat)) * math.cos(a),
                    lat + (r_km / 111.0) * math.sin(a)])
    return pts


def _tier(r):
    if r is None: return "UMI"
    r = float(r)
    if r < 0.4:   return "UMI"
    if r < 1.0:   return "UMA"
    return "RMA"


def extract_towers(html_path):
    print(f"  Extracting tower data from {html_path} ...", flush=True)
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    pos_pat = (r'"id":"(BS_\d+)","geometry":\{"type":"Polygon",'
               r'"coordinates":\[\[\[([+-]?\d+\.\d+),([+-]?\d+\.\d+)')
    pos_m   = re.findall(pos_pat, content)
    txt_pat = r'"Tower (\d+)\\u003cbr\\u003eRadius: ([0-9.]+) km"'
    txt_m   = re.findall(txt_pat, content)
    radius_map = {int(n): float(r) for n, r in txt_m}
    rows = []
    for bs_id, lon_s, lat_s in pos_m:
        n = int(bs_id.replace("BS_", ""))
        r = radius_map.get(n)
        rows.append({"bs_id": bs_id, "lat": float(lat_s),
                     "lon": float(lon_s), "radius_km": r, "tier": _tier(r)})
    df = pd.DataFrame(rows).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    print(f"  -> {len(df):,} unique sites  "
          f"(UMi {(df.tier=='UMI').sum():,}  "
          f"UMa {(df.tier=='UMA').sum():,}  "
          f"RMa {(df.tier=='RMA').sum():,})", flush=True)
    return df


def load_drops(log_path, hour, max_users):
    print(f"  Loading {log_path} ...", flush=True)
    cols  = ["Hour", "Lat", "Lon", "TN_Reason", "NTN_Reason",
             "Final_State", "Demand_Mbps"]
    avail = pd.read_csv(log_path, nrows=0).columns.tolist()
    use   = [c for c in cols if c in avail]
    df    = pd.read_csv(log_path, usecols=use)
    df    = df[(df["Hour"] == hour) & (df["Final_State"] == "DROPPED")]
    print(f"  -> {len(df):,} dropped users at hour {hour}.", flush=True)
    if len(df) > max_users:
        df = df.sample(n=max_users, random_state=42)
        print(f"  Sub-sampled to {max_users:,}.", flush=True)
    return df.reset_index(drop=True)


def classify_drop(reason):
    r = str(reason)
    if r.startswith("No 5G Tower"):  return "No 5G Tower in Geographic Range"
    if r.startswith("5G Bandwidth"): return "5G Bandwidth too low for QoS"
    if r.startswith("5G Congestion"):return "5G Congestion (Tower Empty)"
    if r.startswith("5G SINR"):      return "5G SINR too low"
    return "Other"


def build_geojson_layers(towers, max_circles):
    print(f"  Building coverage-circle GeoJSON layers ...", flush=True)
    layers = []
    for tier in ["UMI", "UMA", "RMA"]:
        sub = towers[towers["tier"] == tier].reset_index(drop=True)
        if len(sub) == 0:
            continue
        n_pts = 12 if tier == "UMI" else 16 if tier == "UMA" else 20
        if len(sub) > max_circles:
            sub = sub.sample(n=max_circles, random_state=0).reset_index(drop=True)
            print(f"    {tier}: sampled to {max_circles:,}", flush=True)
        features = []
        for _, row in sub.iterrows():
            r = float(row["radius_km"]) if pd.notna(row["radius_km"]) else 0.3
            coords = _circle_lonlat(row["lat"], row["lon"], r, n_pts)
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {}
            })
        geojson = {"type": "FeatureCollection", "features": features}
        layers.append({"sourcetype": "geojson", "source": geojson,
                       "type": "fill", "color": TIER_FILL[tier], "opacity": 1.0})
        layers.append({"sourcetype": "geojson", "source": geojson,
                       "type": "line", "color": TIER_COL[tier], "opacity": 0.7})
        print(f"    {tier}: {len(features):,} circles ({n_pts}-pt polygons)", flush=True)
    return layers


def build_figure(towers, drops, max_circles):
    traces = []

    # Tower centre dots
    for tier in ["UMI", "UMA", "RMA"]:
        sub = towers[towers["tier"] == tier]
        if len(sub) == 0: continue
        r_label = {"UMI": "0.31 km", "UMA": "0.47 km", "RMA": "2.85 km"}[tier]
        traces.append(go.Scattermapbox(
            lat=sub["lat"].tolist(), lon=sub["lon"].tolist(),
            mode="markers",
            marker=dict(size=4 if tier == "UMI" else 5 if tier == "UMA" else 6,
                        color=TIER_COL[tier], opacity=0.95),
            name=f"BS {tier}  (r={r_label}, {len(sub):,} sites)",
            hovertemplate=f"<b>{tier}</b><br>Lat: %{{lat:.4f}}<br>Lon: %{{lon:.4f}}<extra></extra>",
        ))

    # Dropped users
    if "TN_Reason" in drops.columns:
        drops = drops.copy()
        drops["_cat"] = drops["TN_Reason"].apply(classify_drop)
    else:
        drops = drops.copy(); drops["_cat"] = "Other"

    cat_order = ["No 5G Tower in Geographic Range", "5G Bandwidth too low for QoS",
                 "5G Congestion (Tower Empty)", "5G SINR too low", "Other"]
    cat_label = {
        "No 5G Tower in Geographic Range": "No TN Coverage",
        "5G Bandwidth too low for QoS":    "TN Bandwidth Full",
        "5G Congestion (Tower Empty)":      "TN Tower Empty",
        "5G SINR too low":                 "TN SINR Too Low",
        "Other":                            "Other Drop",
    }

    for cat in cat_order:
        sub = drops[drops["_cat"] == cat]
        if len(sub) == 0: continue
        col = DROP_COL.get(cat, "#94a3b8")
        cd  = sub["Demand_Mbps"].tolist() if "Demand_Mbps" in sub.columns else None
        htpl = (f"<b>{cat_label[cat]}</b><br>"
                + ("Demand: %{customdata:.2f} Mbps<br>" if cd else "")
                + "Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>")
        traces.append(go.Scattermapbox(
            lat=sub["Lat"].tolist(), lon=sub["Lon"].tolist(),
            mode="markers",
            marker=dict(size=4, color=col, opacity=0.75),
            name=f"{cat_label[cat]}  ({len(sub):,})",
            hovertemplate=htpl, customdata=cd,
        ))

    clat = float(drops["Lat"].mean()) if len(drops) else 46.0
    clon = float(drops["Lon"].mean()) if len(drops) else -80.0

    circle_layers = build_geojson_layers(towers, max_circles)

    n_total  = len(drops)
    n_no_twr = int((drops["_cat"] == "No 5G Tower in Geographic Range").sum())
    n_cong   = int((drops["_cat"] == "5G Bandwidth too low for QoS").sum())
    anno = (f"<b>Hour 20 — Dropped Users</b><br>"
            f"Total: {n_total:,}<br>"
            f"No tower: {n_no_twr:,} ({100*n_no_twr/max(n_total,1):.1f}%)<br>"
            f"Congested: {n_cong:,} ({100*n_cong/max(n_total,1):.1f}%)<br>"
            f"<br><b>Towers (unique sites)</b><br>"
            f"UMi {(towers.tier=='UMI').sum():,}  "
            f"UMa {(towers.tier=='UMA').sum():,}  "
            f"RMa {(towers.tier=='RMA').sum():,}")

    layout = go.Layout(
        title=dict(text="Hybrid NTN-TN — Hour 20 · Dropped Users + Coverage Radius",
                   font=dict(color="#f1f5f9", size=15), x=0.5),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#f1f5f9"),
        mapbox=dict(style=MAPBOX_STYLE,
                    center=dict(lat=clat, lon=clon),
                    zoom=5.5, layers=circle_layers),
        legend=dict(bgcolor="rgba(15,23,42,0.85)", bordercolor="#475569",
                    borderwidth=1, font=dict(color="#f1f5f9", size=11),
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
        margin=dict(l=0, r=0, t=40, b=0),
        annotations=[dict(
            x=0.99, y=0.01, xref="paper", yref="paper",
            xanchor="right", yanchor="bottom",
            text=anno, align="left",
            bgcolor="rgba(15,23,42,0.85)", bordercolor="#475569",
            borderwidth=1, font=dict(color="#f1f5f9", size=11),
            showarrow=False)],
        height=800,
    )
    return go.Figure(data=traces, layout=layout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animation",   default="Final_Animation.html")
    ap.add_argument("--log",         default="detailed_drop_log.csv")
    ap.add_argument("--out",         default="hour20_drops.html")
    ap.add_argument("--hour",        type=float, default=20.0)
    ap.add_argument("--max-users",   type=int,   default=900_000)
    ap.add_argument("--max-circles", type=int,   default=30_000,
                    help="Max circles per tier (cap for file size)")
    args = ap.parse_args()

    print("\n   Hour-20 Drop Map  (Plotly + coverage radius circles)")
    print("=" * 55)

    for p in [args.animation, args.log]:
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found.")

    towers = extract_towers(args.animation)
    drops  = load_drops(args.log, args.hour, args.max_users)

    if len(drops) == 0:
        sys.exit("No dropped users found.")

    print(f"\n  Building Plotly figure ...", flush=True)
    fig = build_figure(towers, drops, args.max_circles)

    print(f"  Saving {args.out} ...", flush=True)
    fig.write_html(
        args.out,
        full_html=True,
        include_plotlyjs="cdn",   # CDN link (~100 KB) instead of 3 MB inline
        config={"scrollZoom": True, "displayModeBar": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\n  Saved {args.out}  ({size_mb:.1f} MB)")
    print(f"  Requires internet to load Plotly from CDN.")
    print(f"  For offline: change include_plotlyjs='cdn' -> True\n")


if __name__ == "__main__":
    main()