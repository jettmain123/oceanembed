"""Stage 01c -- merge harmonized cubes from several laptops into one.

    python scripts/01c_merge_harmonized.py data/incoming/*.nc

Each collecting laptop downloads its OWN months, runs 01_harmonize_real.py, and
uploads the resulting cube. This concatenates them along time into the single
harmonized.nc that everything downstream reads (Contract B).

Time splitting works because every cube already covers the full domain on its own
days -- there is nothing to stitch spatially, and a missing laptop costs you
months rather than corrupting every day.

Checks performed:
  - all cubes share the same lat/lon grid and depth levels
  - duplicate dates are dropped (keeping the first)
  - land_mask is reconciled across files
  - the merged profile still decreases with depth
"""
from __future__ import annotations

import argparse
import glob as _glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr

from oceanembed import ensure_dirs, load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="harmonized .nc files or globs")
    ap.add_argument("--out", default=None, help="default: the configured harmonized.nc")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    out = Path(args.out) if args.out else resolve(cfg["paths"]["harmonized"])

    files: list[str] = []
    for pat in args.inputs:
        files.extend(sorted(_glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    files = [f for f in files if Path(f).resolve() != out.resolve()]
    if not files:
        raise SystemExit("no input files found")

    print(f"merging {len(files)} cubes:")
    cubes = []
    ref_lat = ref_lon = ref_depth = None
    for f in files:
        ds = xr.open_dataset(f).load()
        t = np.asarray(ds["time"].values, dtype="datetime64[D]")
        print(f"  {Path(f).name:44s} {t.min()} .. {t.max()}  ({ds.sizes['time']} days)")

        lat = np.asarray(ds["lat"].values, dtype=np.float32)
        lon = np.asarray(ds["lon"].values, dtype=np.float32)
        dep = np.asarray(ds["depth"].values, dtype=np.float32)
        if ref_lat is None:
            ref_lat, ref_lon, ref_depth = lat, lon, dep
        else:
            for name, a, b in (("lat", lat, ref_lat), ("lon", lon, ref_lon), ("depth", dep, ref_depth)):
                if a.shape != b.shape or not np.allclose(a, b, atol=1e-4):
                    raise SystemExit(
                        f"{Path(f).name}: {name} grid differs from the first cube. "
                        f"Every laptop must use the same configs/config.yaml."
                    )
        cubes.append(ds)

    # land_mask lives outside the time axis; reconcile it, then concatenate
    masks = np.stack([np.asarray(c["land_mask"].values) > 0.5 for c in cubes])
    # land only where EVERY cube agrees -- a short cube can mark a cell land
    # simply because it happened to be missing there for its few days
    land = masks.all(axis=0).astype("float32")
    disagree = int((masks.any(axis=0) & ~masks.all(axis=0)).sum())
    if disagree:
        print(f"\n  {disagree} cells disagreed about land; treated as ocean")

    merged = xr.concat([c.drop_vars("land_mask") for c in cubes], dim="time")
    merged = merged.sortby("time")

    # drop duplicate dates (overlapping month assignments)
    days = np.asarray(merged["time"].values, dtype="datetime64[D]")
    _, keep = np.unique(days, return_index=True)
    dupes = days.size - keep.size
    if dupes:
        print(f"  dropped {dupes} duplicate dates")
    merged = merged.isel(time=np.sort(keep))
    merged["land_mask"] = (("lat", "lon"), land, {"description": "1=land, 0=ocean"})
    merged = merged.transpose("time", "depth", "lat", "lon", missing_dims="ignore")
    merged.attrs.update(cubes[0].attrs)
    merged.attrs["merged_from"] = ", ".join(Path(f).name for f in files)

    if out.exists():
        out.unlink()
    merged.to_netcdf(out, encoding={v: {"zlib": True, "complevel": 4} for v in merged.data_vars})
    for c in cubes:
        c.close()

    t = np.asarray(merged["time"].values, dtype="datetime64[D]")
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {merged.sizes['time']} days: {t.min()} .. {t.max()}")

    # which months, so a missing assignment is obvious
    months = sorted({str(d)[:7] for d in t})
    print(f"  months ({len(months)}): {', '.join(months)}")

    gaps = np.diff(t).astype("timedelta64[D]").astype(int)
    big = int((gaps > 1).sum())
    print(f"  {big} gaps between consecutive days (expected -- the months are not adjacent)")

    print("\nbasin-mean profile -- MUST decrease with depth:")
    prof = merged["temp"].mean(dim=["time", "lat", "lon"], skipna=True)
    tt = np.asarray(prof.values, dtype=float)
    for z, v in zip(np.asarray(merged["depth"].values), tt):
        print(f"  {z:6.0f} m : {v:6.2f} degC")
    fin = tt[np.isfinite(tt)]
    if fin.size > 1 and np.any(np.diff(fin) > 0.5):
        print("\n  WARNING: temperature increases with depth somewhere -- do not train on this")
    else:
        print(f"\n  OK: {fin[0]:.1f} degC at the surface to {fin[-1]:.1f} degC at depth")

    print("\nnext: python scripts/02_build_dataset.py --max-per-day 2000")


if __name__ == "__main__":
    main()
