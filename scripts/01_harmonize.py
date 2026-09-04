"""Stage 01 -- synthetic raw cube  ->  harmonized.nc (Contract B).

    python scripts/01_harmonize.py

Same output schema as 01_harmonize_real.py, so swapping in real data later is a
file swap, not a code change.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xarray as xr

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.harmonize import harmonize_from_cube, missing_report


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    src = resolve(cfg["paths"]["synthetic_cube"])
    out = resolve(cfg["paths"]["harmonized"])
    if not src.exists():
        raise SystemExit(f"missing {src} -- run scripts/00_make_synthetic.py first")

    cube = xr.open_dataset(src)
    ds = harmonize_from_cube(cube, cfg)
    ds = ds.load()
    cube.close()

    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    if out.exists():
        out.unlink()
    ds.to_netcdf(out, encoding=enc)

    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(ds)
    print("\nocean fraction : %.1f%%" % (100.0 * float((ds.land_mask < 0.5).mean())))
    print("missing (%% of ocean cells):")
    for k, v in missing_report(ds, cfg).items():
        print(f"  {k:6s} {v:6.2f}%")


if __name__ == "__main__":
    main()
