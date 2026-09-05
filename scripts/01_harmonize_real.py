"""Stage 01 (REAL) -- downloaded products -> harmonized.nc (Contract B).

    python scripts/01_harmonize_real.py [--dry-run]

Output is byte-for-byte the same SCHEMA as the synthetic 01_harmonize.py, so
02/03/04/05 need no changes at all. Swapping to real data is a file swap.

HOW TO USE
  1. Download the products into data/raw/<product>/ (see scripts/download/README.md).
  2. Edit the SOURCES block below: path glob + variable name for each field.
     Leave the variable name as None to let loaders.PRODUCT_VARS guess it.
  3. Run with --dry-run first: it opens every file and prints the variables and
     dimensions WITHOUT writing anything, so you can fix names quickly.
  4. Run for real.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.harmonize import harmonize, missing_report
from oceanembed.loaders import PRODUCT_VARS, load_glorys_temp, load_variable, open_any

# ===========================================================================
# EDIT BLOCK -- everything a data person needs to change lives between here
# and END EDIT BLOCK. Nothing else in the repo should need touching.
# ===========================================================================

# canonical name -> (path or glob, variable name in the file or None to autodetect)
SOURCES: dict[str, tuple[str, str | None]] = {
    "sst":   ("data/raw/ostia/*.nc",  None),   # OSTIA L4 -- analysed_sst, in KELVIN
    "sss":   ("data/raw/sss/*.nc",    None),   # SMAP/SMOS or Copernicus multiobs -- sos
    "sla":   ("data/raw/duacs/*.nc",  None),   # DUACS L4 -- sla
    "ucur":  ("data/raw/oscar/*.nc",  None),   # OSCAR -- u  (or Copernicus ugos)
    "vcur":  ("data/raw/oscar/*.nc",  None),   # OSCAR -- v
    "uwind": ("data/raw/ccmp/*.nc",   None),   # CCMP -- uwnd  (or ERA5 u10)
    "vwind": ("data/raw/ccmp/*.nc",   None),   # CCMP -- vwnd
}

# GLORYS reanalysis -- the training target
TARGET: tuple[str, str | None] = ("data/raw/glorys/*.nc", None)   # thetao

# Optional explicit land mask; None means "derive it from all-time NaNs in SST/temp"
LAND_MASK: tuple[str, str] | None = None      # e.g. ("data/raw/glorys/mask.nc", "mask")

# ===========================================================================
# END EDIT BLOCK
# ===========================================================================


def inspect(path_glob: str, canonical: str, var: str | None) -> None:
    """Print what is actually inside a product file -- run this before anything else."""
    try:
        ds = open_any(path_glob)
    except FileNotFoundError as exc:
        print(f"  {canonical:6s} MISSING   {exc}")
        return
    guess = var
    if guess is None:
        for cand in PRODUCT_VARS.get(canonical, []):
            if cand in ds:
                guess = cand
                break
    print(f"  {canonical:6s} {path_glob}")
    print(f"         dims      {dict(ds.sizes)}")
    print(f"         variables {list(ds.data_vars)[:10]}")
    print(f"         resolved  {guess!r}" + ("" if guess else "   <-- NOT FOUND, set it in SOURCES"))
    if guess and guess in ds:
        v = ds[guess]
        # take one whole time step so every depth level is represented; slicing
        # the first N flat values would only ever sample the surface layer
        sub = v.isel(time=0) if "time" in v.dims else v
        arr = np.asarray(sub.values, dtype="float64").ravel()
        sample = arr[np.isfinite(arr)]
        if sample.size:
            print(f"         range     {sample.min():.3f} .. {sample.max():.3f} "
                  f"{'(looks like KELVIN -- will be converted)' if np.median(sample) > 100 else ''}")
            if "depth" in sub.dims and sub.sizes["depth"] > 1:
                prof = sub.mean(dim=[d for d in sub.dims if d != "depth"], skipna=True)
                zz = np.asarray(sub["depth"].values, dtype=float)
                tt = np.asarray(prof.values, dtype=float)
                shown = sorted(set([0, len(zz) // 3, 2 * len(zz) // 3, len(zz) - 1]))
                pairs = "  ".join(f"{zz[i]:.0f}m={tt[i]:.1f}" for i in shown)
                print(f"         profile   {pairs}  (should decrease with depth)")
    ds.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="inspect files and exit")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    print("inspecting sources:")
    for canonical, (path, var) in SOURCES.items():
        inspect(path, canonical, var)
    inspect(TARGET[0], "temp", TARGET[1])
    if args.dry_run:
        print("\ndry run -- nothing written. Fix any 'NOT FOUND' entries in the EDIT BLOCK.")
        return

    surface = {}
    for canonical, (path, var) in SOURCES.items():
        surface[canonical] = load_variable(path, canonical, var)
    temp = load_glorys_temp(TARGET[0], TARGET[1])

    lm = None
    if LAND_MASK:
        lm = open_any(LAND_MASK[0])[LAND_MASK[1]]

    ds = harmonize(surface, temp, cfg=cfg, land_mask=lm).load()

    out = resolve(cfg["paths"]["harmonized"])
    if out.exists():
        out.unlink()
    ds.to_netcdf(out, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})

    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(ds)
    print("\nmissing (%% of ocean cells) -- record these in DATA_SOURCES.md:")
    for k, v in missing_report(ds, cfg).items():
        print(f"  {k:6s} {v:6.2f}%")

    # Physical sanity check. If the depth interpolation went wrong (levels
    # reversed, wrong units, bad fill values) it shows up here immediately:
    # temperature must fall from a warm surface to a cold deep ocean.
    print("\nbasin-mean temperature profile -- MUST decrease with depth:")
    prof = ds["temp"].mean(dim=["time", "lat", "lon"], skipna=True)
    zz = np.asarray(ds["depth"].values, dtype=float)
    tt = np.asarray(prof.values, dtype=float)
    for z, t in zip(zz, tt):
        print(f"  {z:6.0f} m : {t:6.2f} degC")
    finite = tt[np.isfinite(tt)]
    if finite.size < 2:
        print("\n  WARNING: profile is mostly NaN -- something is wrong upstream")
    elif np.any(np.diff(finite) > 0.5):
        print("\n  WARNING: temperature INCREASES with depth somewhere above.")
        print("  Check the depth axis direction and units before training on this.")
    else:
        print(f"\n  OK: falls from {finite[0]:.1f} degC at the surface to "
              f"{finite[-1]:.1f} degC at depth")

    print("\nnext: python scripts/02_build_dataset.py")


if __name__ == "__main__":
    main()
