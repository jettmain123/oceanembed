"""Stage 00-check -- verify the real downloads BEFORE spending time on them.

    python scripts/00_check_real_data.py

Opens every product in data/raw/, works out whether the pipeline can actually
use it, and prints a verdict with the exact next command to run.

Checks per product:
  - files present, and how many
  - the canonical variable is findable (via loaders.PRODUCT_VARS)
  - dimensions, grid spacing, and whether lat/lon cover the project domain
  - time axis decodes, and what dates it spans
  - units sanity (SST in Kelvin, depth range for GLORYS)
  - how much of the field is NaN

Then it checks the one thing no single product can tell you: whether all the
products share enough dates to build a training set at all.

Nothing is modified. Safe to run as many times as you like.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr

from oceanembed import load_config, resolve
from oceanembed.harmonize import to_datetime64
from oceanembed.loaders import PRODUCT_VARS

# folder -> which canonical fields we expect to pull out of it
EXPECTED = {
    "ostia": ["sst"],
    "sss": ["sss"],
    "duacs": ["sla"],
    "oscar": ["ucur", "vcur"],
    "ccmp": ["uwind", "vwind"],
    "glorys": ["temp"],
}

OK, WARN, BAD = "PASS", "WARN", "FAIL"
_tally = {OK: 0, WARN: 0, BAD: 0}


def say(level, msg):
    _tally[level] = _tally.get(level, 0) + 1
    print(f"  [{level}] {msg}")


def open_one(path):
    """Open a file, decoding times if we can, falling back if the calendar is odd."""
    try:
        return xr.open_dataset(path), True
    except Exception:
        return xr.open_dataset(path, decode_times=False), False


def coord_of(ds, names):
    for n in names:
        if n in ds.coords or n in ds.dims:
            return n
    return None


def describe_time(ds, decoded):
    tname = coord_of(ds, ["time", "valid_time", "t"])
    if tname is None:
        return None, "no time coordinate"
    vals = np.asarray(ds[tname].values)
    note = ""
    if not np.issubdtype(vals.dtype, np.datetime64):
        # cftime objects (non-standard calendar) -- the pipeline coerces these,
        # so accept them here too rather than calling the product unusable
        try:
            vals = to_datetime64(vals)
            note = "  [cftime calendar -> coerced, same as the pipeline does]"
        except Exception as exc:
            return None, f"time present but undecodable ({vals.dtype}): {exc}"
    days = np.asarray(vals, dtype="datetime64[D]")
    return days, f"{days.min()} .. {days.max()} ({np.unique(days).size} unique days){note}"


def check_product(folder, canon_list, cfg, root):
    print(f"\n=== {folder} ===")
    files = sorted(glob.glob(str(root / folder / "*.nc")))
    if not files:
        say(BAD, f"no .nc files in data/raw/{folder}/")
        return None

    print(f"  {len(files)} files, e.g. {Path(files[0]).name}")
    ds, decoded = open_one(files[0])

    dom = cfg["domain"]
    latn = coord_of(ds, ["lat", "latitude", "nav_lat"])
    lonn = coord_of(ds, ["lon", "longitude", "nav_lon"])
    if latn is None or lonn is None:
        say(BAD, f"no lat/lon coordinate found. coords={list(ds.coords)}")
        ds.close()
        return None

    lat = np.asarray(ds[latn].values, dtype=float)
    lon = np.asarray(ds[lonn].values, dtype=float)
    dlat = abs(float(np.median(np.diff(lat)))) if lat.size > 1 else float("nan")
    dlon = abs(float(np.median(np.diff(lon)))) if lon.size > 1 else float("nan")
    print(f"  grid      {lat.size} lat x {lon.size} lon  spacing ~{dlat:.3f} x {dlon:.3f} deg")
    print(f"  lat range {lat.min():.2f} .. {lat.max():.2f}   lon range {lon.min():.2f} .. {lon.max():.2f}")

    # domain coverage
    need = (dom["lat_min"], dom["lat_max"], dom["lon_min"], dom["lon_max"])
    covers = (lat.min() <= need[0] + 1.0 and lat.max() >= need[1] - 1.0
              and lon.min() <= need[2] + 1.0 and lon.max() >= need[3] - 1.0)
    if covers:
        say(OK, "covers the project domain (5-30N, 45-105E)")
    else:
        say(WARN, "does NOT fully cover 5-30N/45-105E -- harmonize will leave NaN at the edges")
    if lon.max() > 180:
        say(WARN, "longitudes look like 0-360; harmonize converts them, just be aware")

    # time
    days, tmsg = describe_time(ds, decoded)
    print(f"  time      {tmsg}")
    if days is None:
        say(WARN, "time not usable from this single file -- checked again across all files below")

    # variables
    found = {}
    for canon in canon_list:
        hit = None
        for cand in PRODUCT_VARS.get(canon, []):
            if cand in ds:
                hit = cand
                break
        if hit is None:
            say(BAD, f"no variable for '{canon}'. tried {PRODUCT_VARS.get(canon)}; "
                     f"file has {list(ds.data_vars)[:12]}")
        else:
            found[canon] = hit
            v = ds[hit]
            arr = np.asarray(v.values, dtype="float64").ravel()
            finite = arr[np.isfinite(arr)]
            nan_pct = 100.0 * (1 - finite.size / max(arr.size, 1))
            rng = f"{finite.min():.2f} .. {finite.max():.2f}" if finite.size else "all NaN"
            units = v.attrs.get("units", "?")
            print(f"  {canon:6s} <- '{hit}'  dims {tuple(v.dims)}  range {rng} [{units}]  NaN {nan_pct:.1f}%")
            if finite.size == 0:
                say(BAD, f"'{hit}' is entirely NaN in this file")
            elif nan_pct > 80:
                say(WARN, f"'{hit}' is {nan_pct:.0f}% NaN -- a lot of missing data")
            else:
                say(OK, f"'{canon}' found and has real values")
            if canon == "sst" and finite.size and np.median(finite) > 100:
                say(OK, "SST is in Kelvin -- harmonize converts it automatically")

    # depth, for GLORYS
    if "temp" in canon_list:
        dn = coord_of(ds, ["depth", "lev", "deptht", "z"])
        if dn is None:
            say(BAD, "GLORYS has no depth dimension -- cannot be the training target")
        else:
            dv = np.asarray(ds[dn].values, dtype=float)
            print(f"  depth     {dv.size} levels, {dv.min():.1f} .. {dv.max():.1f} m")
            if dv.max() < 900:
                say(BAD, f"deepest level is {dv.max():.0f} m but we need 1000 m -- re-download with "
                         f"--maximum-depth 1100")
            else:
                say(OK, "depth range reaches 1000 m")

    ds.close()

    # time across every file (cheap: read only the time coord)
    all_days = []
    undecodable = 0
    for f in files:
        try:
            d2, dec2 = open_one(f)
            dd, _ = describe_time(d2, dec2)
            if dd is not None:
                all_days.append(np.unique(dd))
            else:
                undecodable += 1
            d2.close()
        except Exception:
            undecodable += 1
    if all_days:
        allv = np.unique(np.concatenate(all_days))
        print(f"  ALL FILES {allv.min()} .. {allv.max()}  ({allv.size} unique days)")
        say(OK, f"{allv.size} distinct days available")
        if undecodable:
            say(WARN, f"{undecodable} file(s) had an undecodable time axis")
        return allv
    say(BAD, "no usable time axis in any file -- harmonize cannot align this product")
    return None


def check_argo(cfg, root):
    print("\n=== argo (independent validation) ===")
    files = sorted(glob.glob(str(root / "argo" / "*.nc")))
    if not files:
        say(WARN, "no .nc in data/raw/argo/ -- you can still train, but you lose the "
                  "independent validation story")
        return
    path = files[0]
    print(f"  {len(files)} file(s), using {Path(path).name}")
    try:
        ds, decoded = open_one(path)
    except Exception as e:
        say(BAD, f"cannot open: {e}")
        return

    print(f"  dims      {dict(ds.sizes)}")
    print(f"  coords    {list(ds.coords)}")
    print(f"  variables {list(ds.data_vars)[:12]}")

    tvar = None
    for cand in PRODUCT_VARS["temp"] + ["TEMP", "temperature", "temp_argo"]:
        if cand in ds:
            tvar = cand
            break
    if tvar is None:
        say(BAD, f"no temperature variable found. tried {PRODUCT_VARS['temp']}")
        ds.close()
        return

    v = ds[tvar]
    print(f"  temp      '{tvar}' dims {tuple(v.dims)}")
    arr = np.asarray(v.values, dtype="float64").ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size:
        print(f"  values    {finite.min():.2f} .. {finite.max():.2f} degC, "
              f"{100 * (1 - finite.size / max(arr.size, 1)):.1f}% NaN")
        if finite.max() > 100:
            say(WARN, "values look like Kelvin -- harmonize converts, but double check")
    else:
        say(BAD, "temperature is entirely NaN")

    # 02b_argo_colocate.py needs a GRIDDED cube
    dims = set(str(d) for d in v.dims)
    gridded = {"depth", "lat", "lon"}.issubset(
        {d.replace("latitude", "lat").replace("longitude", "lon") for d in dims}
    )
    if gridded:
        say(OK, "ARGO is gridded (depth/lat/lon dims) -- 02b_argo_colocate.py will work as-is")
    else:
        say(WARN,
            f"ARGO is NOT gridded -- dims are {tuple(v.dims)}. This looks like a list of "
            f"individual profiles converted from CSV, which is normal for a raw ARGO export. "
            f"02b_argo_colocate.py expects (time,depth,lat,lon). See the note printed at the end.")
    ds.close()


def main():
    cfg = load_config()
    root = resolve(cfg["paths"]["raw"])
    print(f"checking {root}")
    print(f"domain: {cfg['domain']['lat_min']}-{cfg['domain']['lat_max']}N, "
          f"{cfg['domain']['lon_min']}-{cfg['domain']['lon_max']}E")

    spans = {}
    for folder, canon in EXPECTED.items():
        spans[folder] = check_product(folder, canon, cfg, root)

    check_argo(cfg, root)

    # ---- the cross-product question: is there a shared window? ----
    print("\n=== common dates across all surface products + GLORYS ===")
    usable = {k: v for k, v in spans.items() if v is not None and v.size}
    missing = [k for k in EXPECTED if k not in usable]
    if missing:
        say(BAD, f"no usable dates from: {', '.join(missing)}")
    if usable:
        common = None
        for k, v in usable.items():
            common = v if common is None else np.intersect1d(common, v)
        print(f"  products with dates: {', '.join(usable)}")
        if common is not None and common.size:
            print(f"  OVERLAP   {common.min()} .. {common.max()}  ({common.size} days)")
            if common.size >= 25:
                say(OK, f"{common.size} shared days -- enough to train")
            elif common.size >= 10:
                say(WARN, f"only {common.size} shared days -- trainable but thin; "
                          f"the val and holdout blocks will be tiny")
            else:
                say(BAD, f"only {common.size} shared days -- too few to split into train/val/test")
            n = common.size
            n_argo = max(1, round(0.15 * n))
            n_val = max(1, round(0.15 * (n - n_argo)))
            print(f"  split     ~{n - n_argo - n_val} train / {n_val} val / {n_argo} holdout days")
        else:
            say(BAD, "NO overlapping dates -- the products cover different periods, "
                     "so no training sample can be built")

    # ---- verdict ----
    print("\n" + "=" * 62)
    print(f"SUMMARY   {_tally[OK]} pass   {_tally[WARN]} warn   {_tally[BAD]} fail")
    print("=" * 62)
    if _tally[BAD]:
        print("\nNOT READY. Fix the FAIL lines above, then run this again.")
    else:
        print("\nREADY. Next:")
        print("  python scripts/01_harmonize_real.py --dry-run   # inspect, writes nothing")
        print("  python scripts/01_harmonize_real.py             # -> data/processed/harmonized.nc")
        print("  python scripts/02_build_dataset.py")
        print("  python scripts/03_train.py")
        print("  python scripts/03b_baseline.py")
        print("  python scripts/04_evaluate.py")
    print("\nIf ARGO came out as a profile list rather than a grid, that is fine and normal --")
    print("it just needs a point-based co-location step instead of 02b. Say so and it can be")
    print("written. You can train and evaluate without it in the meantime.")

    # exit code, so a collection script can branch on this instead of parsing text
    return 1 if _tally[BAD] else 0


if __name__ == "__main__":
    sys.exit(main())
