"""Stage 02 -- harmonized.nc (Contract B) -> dataset.npz (Contract C).

    python scripts/02_build_dataset.py

Works identically on the synthetic and the real harmonized cube.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.dataset import build, channel_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-day", type=int, default=None,
                    help="ocean cells sampled per day (default: dataset.max_samples_per_day "
                         "in config). Raise it when you have few DAYS but many cells.")
    args = ap.parse_args()

    cfg = load_config()
    if args.max_per_day:
        cfg["dataset"]["max_samples_per_day"] = args.max_per_day
        print(f"[dataset] sampling up to {args.max_per_day} cells per day")
    ensure_dirs(cfg)
    src = resolve(cfg["paths"]["harmonized"])
    out = resolve(cfg["paths"]["dataset"])
    if not src.exists():
        raise SystemExit(f"missing {src} -- run scripts/01_harmonize.py first")

    ds = xr.open_dataset(src)
    payload = build(ds, cfg)
    ds.close()

    np.savez(out, **payload)

    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("channels (%d):" % len(channel_names(cfg)), channel_names(cfg))
    for split in ("tr", "va", "ar"):
        X, Y = payload[f"{split}X"], payload[f"{split}Y"]
        print(f"  {split}: X{X.shape}  Y{Y.shape}")
    print("\ntrain days :", payload["train_days"].min(), "..", payload["train_days"].max())
    print("val   days :", payload["val_days"].min(), "..", payload["val_days"].max())
    print("argo  days :", payload["argo_days"].min(), "..", payload["argo_days"].max(), "(never trained on)")
    print("\nper-depth target mean/std from TRAIN split:")
    for z, m, s in zip(payload["depths"], payload["ym"], payload["ys"]):
        print(f"  {z:6.0f} m : mean {m:6.2f}  std {s:5.2f}")
    print("\nNaNs in trX/trY:", int(np.isnan(payload["trX"]).sum()), int(np.isnan(payload["trY"]).sum()))


if __name__ == "__main__":
    main()
