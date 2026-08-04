#!/usr/bin/env python3
"""
milp_to_bs_csv.py — validation harness, step 1 of 2.

Converts a MILP placement (province_placement.csv / milp_placement_<hex>.csv:
columns hex,lat,lon,tier  or  lat,lon,tier) into the simulator's NEW-format
base-station CSV, so the EXISTING reload path runs it through the full 3GPP
pipeline untouched:

    step 2 (nothing new to write -- the pipeline already supports it):
        set  terrestrial.load_bs_csv: /path/to/milp_bs.csv  in the config
        and run the normal simulation sbatch. Placement is skipped; mobility,
        shadowing, capacity-aware attachment, PF scheduling and NTN beam
        allocation all run for real.

FIX IN THIS REVISION
--------------------
This used `yaml.safe_load(open(root_config))` and then looked for a
`terrestrial.scenarios` block. The root config is a Hydra `defaults:` list --
yaml.safe_load does NOT resolve it, so the lookup found nothing and the
script exited with "no scenarios block found in the YAML" no matter what.
It now composes the config with Hydra (and still accepts a plain group file
via --scenarios-yaml for offline use).

Faithful expansion of the MILP's own assumptions:
  - every site -> 3 sector cells at azimuths 30/150/270 (TR 38.901
    calibration), each with the FULL channel bandwidth (reuse-1), shared
    site_id == exactly the per-sector W_t budget the MILP enforced;
  - UMI_MMW sites are exported with scenario "UMI" (the enum the simulator
    uses for 28 GHz street canyon) but carry their own RF columns;
  - all RF columns read from the SAME config the MILP used, so the simulator
    judges the placement under identical physics parameters.

Usage:
    python milp_to_bs_csv.py --placement province_placement.csv \
        --config-dir /Utilisateurs/dbenguer/ntn_tn_optim/configs \
        --config-name base --out milp_bs.csv
"""
from __future__ import annotations
import argparse
import pandas as pd

AZIMUTHS = (30.0, 150.0, 270.0)          # TR 38.901 3-sector calibration
ENUM_OF = {"RMA": "RMA", "UMA": "UMA", "UMI": "UMI", "UMI_MMW": "UMI"}

RF_MAP = {                                # csv column   <- config key
    "p_tx_dbm": "p_tx_dbm",
    "g_tx_dbi": "g_tx_dbi",
    "carrier_freq_hz": "carrier_freq_hz",
    "bandwidth_hz": "bandwidth_hz",
    "bs_height_m": "default_h_bs",
    "shadow_sigma_los_db": "shadow_sigma_los_db",
    "shadow_sigma_nlos_db": "shadow_sigma_nlos_db",
    "interference_cutoff_m": "interference_cutoff_m",
    "min_user_dist_m": "min_user_dist_m",
}


def _load_scenarios(args):
    if args.scenarios_yaml:
        import yaml
        with open(args.scenarios_yaml) as f:
            doc = yaml.safe_load(f)
        return (doc.get("terrestrial", {}) or {}).get("scenarios") \
            or doc.get("scenarios") or doc
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=args.config_dir, version_base=None):
        cfg = compose(config_name=args.config_name)
    return cfg.terrestrial.scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", required=True)
    ap.add_argument("--config-dir",
                    default="/Utilisateurs/dbenguer/ntn_tn_optim/configs")
    ap.add_argument("--config-name", default="base")
    ap.add_argument("--scenarios-yaml", default=None,
                    help="bypass Hydra: a YAML that already contains the "
                         "resolved scenarios block")
    ap.add_argument("--out", default="milp_bs.csv")
    args = ap.parse_args()

    scen_cfg = _load_scenarios(args)
    if not scen_cfg:
        raise SystemExit("no scenarios block found")

    df = pd.read_csv(args.placement)
    tier_col = "tier" if "tier" in df.columns else "scenario"
    rows, bs_id = [], 0
    skipped = {}
    for site_id, r in df.iterrows():
        tier = str(r[tier_col])
        if tier not in scen_cfg:
            skipped[tier] = skipped.get(tier, 0) + 1
            continue
        sc = scen_cfg[tier]
        for az in AZIMUTHS:
            row = {
                "bs_id": bs_id, "site_id": site_id,
                "lat": float(r["lat"]), "lon": float(r["lon"]),
                "scenario": ENUM_OF[tier],
                "sector_azimuth_deg": az, "num_sectors": 3,
                "coverage_radius_km": float(sc["coverage_radius_km"]),
            }
            for col, key in RF_MAP.items():
                row[col] = float(sc[key])
            rows.append(row)
            bs_id += 1

    if not rows:
        raise SystemExit(f"no rows written; tiers in the placement "
                         f"({sorted(set(df[tier_col].astype(str)))}) do not "
                         f"match the config ({sorted(scen_cfg.keys())})")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    mix = out[out.sector_azimuth_deg == 30.0].groupby("scenario").size().to_dict()
    n_mmw = int((df[tier_col].astype(str) == "UMI_MMW").sum())
    print(f"wrote {args.out}: {len(out):,} sector cells "
          f"({len(out)//3:,} sites, mix by enum: {mix}, "
          f"of which {n_mmw} mmWave sites exported as UMI+400MHz RF)")
    if skipped:
        print(f"WARNING skipped tiers not in config: {skipped}")
    print("\nnext: set  terrestrial.load_bs_csv: "
          f"{args.out}  in the config and run the normal simulation. "
          "Placement will be skipped; the full pipeline (mobility, shadowing, "
          "capacity-aware attachment, PF, NTN beams) judges this deployment.")


if __name__ == "__main__":
    main()