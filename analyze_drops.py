#!/usr/bin/env python
"""
Analyze detailed_drop_log.csv: where demand goes and WHY users drop.

Reads the log (optionally one hour), and reports:
  - Final_State split (counts + demand Mbps + % of demand)
  - For DROPPED users: the TN_Reason x NTN_Reason breakdown (the root cause)
  - TN and NTN reason tallies overall
  - Per-tower load: busiest base stations and their served demand
  - A map of dropped users (lat/lon scatter) saved as PNG

Usage:
    python analyze_drops.py
    python analyze_drops.py --csv detailed_drop_log.csv --hour 20
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pct(x, total):
    return f"{100.0 * x / total:5.1f}%" if total else "  n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="detailed_drop_log.csv")
    ap.add_argument("--hour", type=float, default=None, help="filter to one Hour value")
    ap.add_argument("--top", type=int, default=15, help="how many busiest towers to list")
    ap.add_argument("--out", default="drop_analysis.png")
    ap.add_argument("--tn-floor", type=float, default=-3.0, help="TN min SINR dB")
    ap.add_argument("--ntn-floor", type=float, default=0.0, help="NTN min SINR dB")
    ap.add_argument("--split-lat", type=float, default=46.0,
                    help="latitude dividing 'south/dense' (<lat) from 'north/sparse' (>=lat)")
    ap.add_argument("--region", choices=["all", "south", "north"], default="all",
                    help="restrict the WHOLE analysis to one region")
    args = ap.parse_args()

    # Read only the columns we need, with compact dtypes (the file is huge).
    cols = ["Hour", "User_ID", "Lat", "Lon", "Demand_Mbps",
            "TN_Eval_BS", "TN_Reason", "NTN_Reason", "Final_State",
            "TN_SINR_dB", "NTN_SINR_dB",
            "TN_S_dBm", "TN_I_dBm", "TN_N_dBm", "TN_NumIntf", "TN_IoverN_dB"]
    # Read only columns that actually exist (SINR cols may be absent in old logs).
    available = pd.read_csv(args.csv, nrows=0).columns.tolist()
    use = [c for c in cols if c in available]
    df = pd.read_csv(args.csv, usecols=use)
    for c in ("TN_SINR_dB", "NTN_SINR_dB",
              "TN_S_dBm", "TN_I_dBm", "TN_N_dBm", "TN_NumIntf", "TN_IoverN_dB"):
        if c not in df.columns:
            df[c] = float("nan")
    print(f"Loaded {len(df):,} rows.")

    if args.hour is not None:
        df = df[df["Hour"] == args.hour]
        print(f"Filtered to hour {args.hour}: {len(df):,} rows.")

    if args.region == "south":
        df = df[df["Lat"] < args.split_lat]
        print(f"Region = SOUTH (Lat < {args.split_lat}): {len(df):,} rows.")
    elif args.region == "north":
        df = df[df["Lat"] >= args.split_lat]
        print(f"Region = NORTH (Lat >= {args.split_lat}): {len(df):,} rows.")

    cfg_tn_floor = args.tn_floor
    cfg_ntn_floor = args.ntn_floor
    total_demand = df["Demand_Mbps"].sum()
    print(f"\nTotal demand in log: {total_demand/1e6:.3f} Tbps over {len(df):,} active users\n")

    # ---- 1. Final state split ----
    print("=== FINAL STATE ===")
    g = df.groupby("Final_State")["Demand_Mbps"].agg(["count", "sum"])
    for state, row in g.sort_values("sum", ascending=False).iterrows():
        print(f"  {state:10s}  users={int(row['count']):>10,}  "
              f"demand={row['sum']/1e6:7.3f} Tbps  ({pct(row['sum'], total_demand)} of demand)")

    # ---- 2. Why did DROPPED users drop? (TN x NTN reason pairs) ----
    dropped = df[df["Final_State"] == "DROPPED"]
    print(f"\n=== DROP ROOT CAUSE  ({len(dropped):,} dropped users, "
          f"{dropped['Demand_Mbps'].sum()/1e6:.3f} Tbps) ===")
    if len(dropped):
        pair = (dropped.groupby(["TN_Reason", "NTN_Reason"])["Demand_Mbps"]
                .agg(["count", "sum"]).sort_values("sum", ascending=False))
        d_tot = dropped["Demand_Mbps"].sum()
        for (tn_r, ntn_r), row in pair.head(12).iterrows():
            print(f"  [{pct(row['sum'], d_tot)}] {int(row['count']):>9,} users  "
                  f"demand={row['sum']/1e6:6.3f} Tbps")
            print(f"            TN : {tn_r}")
            print(f"            NTN: {ntn_r}")


    # ---- 2b. GEOGRAPHIC drop breakdown: south/dense vs north/sparse ----
    if args.region == "all":
        print(f"\n=== DROPS BY REGION (split at Lat {args.split_lat}) ===")
        for label, sub in (("SOUTH (Lat <  %.1f)" % args.split_lat, df[df["Lat"] < args.split_lat]),
                           ("NORTH (Lat >= %.1f)" % args.split_lat, df[df["Lat"] >= args.split_lat])):
            d = sub[sub["Final_State"] == "DROPPED"]
            tot_dem = sub["Demand_Mbps"].sum()
            d_dem = d["Demand_Mbps"].sum()
            served = sub[sub["Final_State"].isin(["TN", "LEO"])]["Demand_Mbps"].sum()
            print(f"\n  {label}: {len(sub):,} users  "
                  f"demand={tot_dem/1e6:.3f} Tbps  "
                  f"served={served/1e6:.3f}  dropped={d_dem/1e6:.3f} Tbps "
                  f"({pct(d_dem, tot_dem)} dropped)")
            if len(d):
                # TN reason split among dropped users in this region.
                tn = d.groupby("TN_Reason")["Demand_Mbps"].agg(["count", "sum"]).sort_values("sum", ascending=False)
                print(f"    TN reason among dropped:")
                for r, row in tn.head(4).iterrows():
                    print(f"      {pct(row['sum'], d_dem)}  {int(row['count']):>9,}  {r}")
                ntn = d.groupby("NTN_Reason")["Demand_Mbps"].agg(["count", "sum"]).sort_values("sum", ascending=False)
                print(f"    NTN reason among dropped:")
                for r, row in ntn.head(4).iterrows():
                    print(f"      {pct(row['sum'], d_dem)}  {int(row['count']):>9,}  {r}")

    # ---- 3. Overall reason tallies ----
    print("\n=== TN_REASON tally (all users) ===")
    for r, c in df["TN_Reason"].value_counts().head(10).items():
        print(f"  {c:>10,}  {r}")
    print("\n=== NTN_REASON tally (all users) ===")
    for r, c in df["NTN_Reason"].value_counts().head(10).items():
        print(f"  {c:>10,}  {r}")


    # ---- 3b. SINR distributions (TN vs NTN) ----
    import numpy as np
    tn_floor  = float(cfg_tn_floor)
    ntn_floor = float(cfg_ntn_floor)

    def sinr_stats(series, label, floor):
        a = pd.to_numeric(series, errors="coerce").to_numpy()
        a = a[np.isfinite(a)]
        if len(a) == 0:
            print(f"  {label}: no SINR values logged"); return
        qs = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
        below = (a < floor).sum()
        print(f"  {label}: n={len(a):,}  mean={a.mean():6.2f} dB  "
              f"median={qs[3]:6.2f}  p5={qs[1]:6.2f}  p95={qs[5]:6.2f}  "
              f"min={a.min():6.2f}  max={a.max():6.2f}")
        print(f"      below floor ({floor:+.1f} dB): {below:,} "
              f"({100.0*below/len(a):.2f}%)   "
              f"p1={qs[0]:.2f}  p99={qs[6]:.2f}")

    print("\n=== SINR DISTRIBUTIONS (served + evaluated users) ===")
    # All users that had a TN SINR evaluated (attached to a tower), regardless of final state.
    sinr_stats(df["TN_SINR_dB"], "TN  SINR (all evaluated)", tn_floor)
    # Only users actually served by TN.
    sinr_stats(df.loc[df["Final_State"] == "TN", "TN_SINR_dB"], "TN  SINR (served only) ", tn_floor)
    # All users that had an NTN SINR evaluated (a beam looked at them).
    sinr_stats(df["NTN_SINR_dB"], "NTN SINR (all evaluated)", ntn_floor)
    # Only users actually served by NTN/LEO.
    sinr_stats(df.loc[df["Final_State"] == "LEO", "NTN_SINR_dB"], "NTN SINR (served only) ", ntn_floor)

    # Low-SINR breakdown by final state (who are the weak-link users?)
    print("\n=== LOW-SINR USERS by final state ===")
    tn_low  = pd.to_numeric(df["TN_SINR_dB"], errors="coerce") < tn_floor
    ntn_low = pd.to_numeric(df["NTN_SINR_dB"], errors="coerce") < ntn_floor
    print(f"  TN  SINR < {tn_floor:+.1f} dB : {int(tn_low.sum()):,} users")
    if tn_low.any():
        for st, c in df.loc[tn_low, "Final_State"].value_counts().items():
            print(f"        {st:8s}: {c:,}")
    print(f"  NTN SINR < {ntn_floor:+.1f} dB : {int(ntn_low.sum()):,} users")
    if ntn_low.any():
        for st, c in df.loc[ntn_low, "Final_State"].value_counts().items():
            print(f"        {st:8s}: {c:,}")

    # SINR histograms saved alongside the drop map.
    try:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        tn_vals  = pd.to_numeric(df["TN_SINR_dB"], errors="coerce").dropna()
        ntn_vals = pd.to_numeric(df["NTN_SINR_dB"], errors="coerce").dropna()
        if len(tn_vals):
            ax[0].hist(tn_vals, bins=80, color="#2563eb")
            ax[0].axvline(tn_floor, c="r", ls="--", label=f"floor {tn_floor:+.1f} dB")
            ax[0].set_title("TN SINR (dB)"); ax[0].set_xlabel("SINR (dB)"); ax[0].legend()
        if len(ntn_vals):
            ax[1].hist(ntn_vals, bins=80, color="#16a34a")
            ax[1].axvline(ntn_floor, c="r", ls="--", label=f"floor {ntn_floor:+.1f} dB")
            ax[1].set_title("NTN SINR (dB)"); ax[1].set_xlabel("SINR (dB)"); ax[1].legend()
        plt.tight_layout(); plt.savefig("sinr_distributions.png", dpi=120)
        print("\nSaved SINR histograms to sinr_distributions.png")
    except Exception as e:
        print(f"(SINR plot skipped: {e})")


    # ---- 3c. TN SINR COMPONENT DIAGNOSTIC (why is SINR high/low?) ----
    # S/I/N in dBm + interferer count. The key signal is I/N:
    #   I/N > 0 dB  -> interference-limited (realistic dense network)
    #   I/N < 0 dB  -> noise-limited (SINR will read high; few/weak interferers)
    if "TN_IoverN_dB" in df.columns and df["TN_IoverN_dB"].notna().any():
        print("\n=== TN SINR COMPONENTS (served TN users) ===")
        tnu = df[df["Final_State"] == "TN"]
        if len(tnu):
            S   = pd.to_numeric(tnu["TN_S_dBm"], errors="coerce")
            Ico = pd.to_numeric(tnu["TN_I_dBm"], errors="coerce")
            N   = pd.to_numeric(tnu["TN_N_dBm"], errors="coerce")
            ni  = pd.to_numeric(tnu["TN_NumIntf"], errors="coerce")
            ion = pd.to_numeric(tnu["TN_IoverN_dB"], errors="coerce")

            def line(name, a, unit="dBm"):
                a = a.replace([np.inf, -np.inf], np.nan).dropna()
                if len(a) == 0:
                    print(f"  {name}: (none)"); return
                q = np.percentile(a, [5, 50, 95])
                print(f"  {name:14s}: median={q[1]:8.2f} {unit}  "
                      f"p5={q[0]:8.2f}  p95={q[2]:8.2f}  mean={a.mean():8.2f}")

            line("Signal  S", S)
            line("Interf. I", Ico)
            line("Noise   N", N)
            line("I/N ratio", ion, unit="dB")
            # Interferer-count distribution.
            ni_clean = ni.dropna()
            if len(ni_clean):
                print(f"  interferers   : median={int(np.median(ni_clean))}  "
                      f"mean={ni_clean.mean():.1f}  "
                      f"max={int(ni_clean.max())}  "
                      f"zero={int((ni_clean==0).sum()):,} users")

            # The verdict line: noise- vs interference-limited.
            ion_clean = ion.replace([np.inf, -np.inf], np.nan).dropna()
            if len(ion_clean):
                frac_noise = float((ion_clean < 0).mean()) * 100.0
                med_ion = float(np.median(ion_clean))
                regime = "NOISE-LIMITED" if med_ion < 0 else "INTERFERENCE-LIMITED"
                print(f"  --> {regime}: median I/N = {med_ion:+.1f} dB, "
                      f"{frac_noise:.0f}% of users have I < N.")
                if med_ion < 0:
                    print("      High SINR is EXPECTED here: interference sits below noise,")
                    print("      so the link is noise-limited. To make it interference-limited,")
                    print("      towers must be closer / interference_cutoff wider / more reuse.")
                else:
                    print("      Interference dominates noise, yet check if SINR is still high:")
                    print("      if so, the serving-signal term (power+gains) is too strong.")

    # ---- 4. Busiest towers (TN served demand by BS) ----
    tn = df[df["Final_State"] == "TN"]
    if len(tn):
        print(f"\n=== TOP {args.top} BUSIEST TOWERS (by served demand) ===")
        bs = tn.groupby("TN_Eval_BS")["Demand_Mbps"].agg(["count", "sum"]).sort_values("sum", ascending=False)
        for name, row in bs.head(args.top).iterrows():
            print(f"  {name:>12}  users={int(row['count']):>8,}  served={row['sum']:9.1f} Mbps")

    # ---- 5. Map of dropped users ----
    if len(dropped):
        plt.figure(figsize=(9, 9))
        served = df[df["Final_State"].isin(["TN", "LEO"])]
        plt.scatter(served["Lon"], served["Lat"], s=0.2, c="#cbd5e1", label="served", rasterized=True)
        # A true coverage hole = NEITHER tier could reach the user:
        #   TN has no tower in range AND NTN has no satellite overhead.
        # Everything else is a congestion/capacity drop (infrastructure reached
        # the user but was full).
        no_tn  = dropped["TN_Reason"].str.contains("No 5G Tower", na=False)
        no_sat = dropped["NTN_Reason"].str.contains("No Satellite", na=False)
        hole = dropped[no_tn & no_sat]
        cong = dropped[~(no_tn & no_sat)]
        print(f"  map: {len(hole):,} true coverage-hole drops, "
              f"{len(cong):,} congestion drops")
        plt.scatter(cong["Lon"], cong["Lat"], s=0.6, c="#f59e0b", label="dropped: congestion", rasterized=True)
        plt.scatter(hole["Lon"], hole["Lat"], s=0.6, c="#dc2626", label="dropped: no coverage", rasterized=True)
        plt.xlabel("Lon"); plt.ylabel("Lat")
        plt.title(f"Dropped users  (hour {args.hour})")
        plt.legend(markerscale=20, loc="upper right")
        plt.tight_layout()
        plt.savefig(args.out, dpi=120)
        print(f"\nSaved map to {args.out}")


if __name__ == "__main__":
    main()