"""Stage 00 -- generate the synthetic raw cube.

    python scripts/00_make_synthetic.py

Writes data/raw/synthetic_cube.nc. Nothing downstream reads this file directly;
01_harmonize.py turns it into the harmonized cube (Contract B).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.synthetic import make_cube


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    out = resolve(cfg["paths"]["synthetic_cube"])

    ds = make_cube(cfg)

    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out, encoding=enc)

    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(ds)
    land_frac = float(ds["land_mask"].mean())
    print(f"\nland fraction : {land_frac:.1%}")
    print("sst  range    : %.2f .. %.2f degC" % (float(ds.sst.min()), float(ds.sst.max())))
    print("sla  range    : %.3f .. %.3f m" % (float(ds.sla.min()), float(ds.sla.max())))
    print("temp range    : %.2f .. %.2f degC" % (float(ds.temp.min()), float(ds.temp.max())))
    prof = ds.temp.isel(time=0).mean(dim=("lat", "lon"))
    print("\nbasin-mean profile at t=0:")
    for z, t in zip(np.asarray(ds.depth), np.asarray(prof)):
        print(f"  {z:6.0f} m : {t:6.2f} degC")


if __name__ == "__main__":
    main()
