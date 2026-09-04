"""Stage 02c -- harmonized.nc -> flat one-row-per-ocean-cell table.

    python scripts/02c_export_table.py [--stride N] [--limit-days N]

Columns (25 by default):
    date, lat, lon, sst, sss, sla, ucur, vcur, uwind, vwind, t_0 ... t_1000

This is the hand-off / baseline format: anyone can open it in pandas without
knowing anything about xarray. It is NOT what the CNN trains on -- the CNN needs
the N x N patches from dataset.npz.

Vectorised and chunked by day: each day builds one DataFrame from a boolean mask
of valid ocean cells. A per-cell Python loop over the full domain would OOM.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from oceanembed import ensure_dirs, load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="spatial subsampling stride")
    ap.add_argument("--limit-days", type=int, default=None, help="only export the first N days")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    src = resolve(cfg["paths"]["harmonized"])
    out = Path(args.out) if args.out else resolve(cfg["paths"]["samples_table"])
    if not src.exists():
        raise SystemExit(f"missing {src} -- run scripts/01_harmonize.py first")

    ds = xr.open_dataset(src)
    svars = list(cfg["surface_vars"])
    depths = np.asarray(ds["depth"].values, dtype=np.float32)
    tcols = [f"t_{int(z)}" for z in depths]

    lat = np.asarray(ds["lat"].values, dtype=np.float32)[:: args.stride]
    lon = np.asarray(ds["lon"].values, dtype=np.float32)[:: args.stride]
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    LA, LO = LA.ravel(), LO.ravel()
    land = np.asarray(ds["land_mask"].values)[:: args.stride, :: args.stride].ravel() > 0.5

    times = ds["time"].values
    if args.limit_days:
        times = times[: args.limit_days]

    frames = []
    for it in range(times.size):
        day = ds.isel(time=it)
        surf = {v: np.asarray(day[v].values, np.float32)[:: args.stride, :: args.stride].ravel() for v in svars}
        tgt = np.asarray(day["temp"].values, np.float32)[:, :: args.stride, :: args.stride]
        tgt = tgt.reshape(depths.size, -1)

        ok = ~land
        for v in svars:
            ok &= np.isfinite(surf[v])
        ok &= np.isfinite(tgt).all(axis=0)
        if not ok.any():
            continue

        data = {"date": np.datetime64(times[it], "D"), "lat": LA[ok], "lon": LO[ok]}
        data.update({v: surf[v][ok] for v in svars})
        data.update({c: tgt[k][ok] for k, c in enumerate(tcols)})
        frames.append(pd.DataFrame(data))

        if (it + 1) % 20 == 0 or it == times.size - 1:
            print(f"  day {it + 1}/{times.size}  rows so far {sum(len(f) for f in frames):,}")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    try:
        df.to_parquet(out, index=False)
    except Exception as exc:                       # no pyarrow -> CSV fallback
        out = out.with_suffix(".csv.gz")
        print(f"parquet unavailable ({type(exc).__name__}), writing {out}")
        df.to_csv(out, index=False, compression="gzip")

    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"rows    : {len(df):,}")
    print(f"columns ({len(df.columns)}): {list(df.columns)}")
    print("\nhead:")
    print(df.head(3).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\ndescribe (subset):")
    print(df[["sst", "sla", "t_100", "t_1000"]].describe().to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
