"""Convert INCOIS LAS gridded ARGO text exports → per-month NetCDF files.

Input files (INCOIS Live Access Server export):
  data/raw/argo/argo_2022.txt
  data/raw/argo/argo_2023.txt
  data/raw/argo/argo_2024.txt

Each .txt file has a header followed by CSV rows:
  DATETIME, TIME, LON, LAT, HT, TEMP
  TIME is days since 15-JAN-1901 00:00:00
  Bad flag: -1.E+34  (→ NaN)
  HT is depth in metres.
  Each unique (DATETIME, HT) combination covers the full (lat, lon) grid.

Output:
  data/raw/argo/argo_YYYY-MM.nc   — one NetCDF per calendar month.

Each output NetCDF has:
  dims  : time (n_dates_in_month), depth (n_depths), lat (n_lat), lon (n_lon)
  var   : TEMP  float32 (time, depth, lat, lon)  units=degC  _FillValue=NaN
  attrs : source, bad_flag_original, time_origin

Usage:
  python scripts/convert_argo_txt_to_nc.py          # convert all three years
  python scripts/convert_argo_txt_to_nc.py --year 2023   # one year only
  python scripts/convert_argo_txt_to_nc.py --dry-run     # parse only, write nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ARGO_DIR = Path("data/raw/argo")
YEARS = [2022, 2023, 2024]
BAD_FLAG = -1.0e34
BAD_THRESHOLD = -1.0e33  # anything below this is treated as missing

# INCOIS LAS time origin: "DAYS since 15-JAN-1901 00:00:00"
TIME_ORIGIN = pd.Timestamp("1901-01-15")


def days_to_timestamp(days: float) -> pd.Timestamp:
    return TIME_ORIGIN + pd.Timedelta(days=float(days))


# ---------------------------------------------------------------------------
# PARSE
# ---------------------------------------------------------------------------

def parse_txt(path: Path, verbose: bool = True) -> pd.DataFrame:
    """Read an INCOIS LAS ARGO text file → clean DataFrame.

    Skips the metadata header (lines before the CSV header row
    'DATETIME,TIME,...') and returns a DataFrame with columns:
      date (pd.Timestamp), lon (float), lat (float), depth_m (float), temp (float32)
    Bad flags are replaced with NaN.
    """
    path = Path(path)
    if verbose:
        print(f"  reading  {path.name}  ({path.stat().st_size / 1e6:.1f} MB) ...", flush=True)

    # Find the header row index so we can skip the metadata block
    header_line = None
    with open(path, "r") as fh:
        for i, line in enumerate(fh):
            if line.strip().startswith("DATETIME,"):
                header_line = i
                break

    if header_line is None:
        raise ValueError(f"Could not find CSV header in {path}")

    df = pd.read_csv(
        path,
        skiprows=header_line,
        header=0,
        names=["datetime_str", "time_days", "lon", "lat", "depth_m", "temp"],
        dtype={"time_days": float, "lon": float, "lat": float,
               "depth_m": float, "temp": str},
        quotechar='"',
        skipinitialspace=True,
        engine="c",
    )

    # Parse temperature: strip whitespace, coerce bad flags to NaN
    df["temp"] = pd.to_numeric(df["temp"].str.strip(), errors="coerce")
    df.loc[df["temp"] < BAD_THRESHOLD, "temp"] = np.nan
    df["temp"] = df["temp"].astype("float32")

    # Convert TIME (days since 15-JAN-1901) to actual date (no time component)
    unique_days = df["time_days"].unique()
    day_map = {d: days_to_timestamp(d).normalize() for d in unique_days}
    df["date"] = df["time_days"].map(day_map)

    df = df.drop(columns=["datetime_str", "time_days"])
    df = df.reset_index(drop=True)

    if verbose:
        print(f"    rows   {len(df):,}")
        print(f"    dates  {df['date'].nunique()}  "
              f"({df['date'].min().date()} ... {df['date'].max().date()})")
        print(f"    depths {sorted(df['depth_m'].unique())}")
        print(f"    lat    {df['lat'].min()}-{df['lat'].max()}  "
              f"lon {df['lon'].min()}-{df['lon'].max()}")
        frac_nan = df["temp"].isna().mean()
        print(f"    missing {frac_nan*100:.1f}%")

    return df


# ---------------------------------------------------------------------------
# PIVOT to xarray
# ---------------------------------------------------------------------------

def df_to_xarray(df_month: pd.DataFrame) -> xr.Dataset:
    """Pivot a single-month DataFrame into an xr.Dataset (time, depth, lat, lon)."""

    dates = np.sort(df_month["date"].unique())
    depths = np.sort(df_month["depth_m"].unique()).astype(np.float32)
    lats = np.sort(df_month["lat"].unique()).astype(np.float32)
    lons = np.sort(df_month["lon"].unique()).astype(np.float32)

    nt, nz, ny, nx = len(dates), len(depths), len(lats), len(lons)
    temp_arr = np.full((nt, nz, ny, nx), np.nan, dtype=np.float32)

    # Build index maps
    date_idx = {d: i for i, d in enumerate(dates)}
    depth_idx = {d: i for i, d in enumerate(depths)}
    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}

    # Vectorised fill using pandas index lookups
    df_valid = df_month.dropna(subset=["temp"])
    it = df_valid["date"].map(date_idx).values
    iz = df_valid["depth_m"].map(depth_idx).values
    iy = df_valid["lat"].map(lat_idx).values
    ix = df_valid["lon"].map(lon_idx).values
    temp_arr[it, iz, iy, ix] = df_valid["temp"].values

    # time as datetime64[ns]
    times = pd.DatetimeIndex(dates).values

    ds = xr.Dataset(
        {
            "TEMP": xr.Variable(
                ["time", "depth", "lat", "lon"],
                temp_arr,
                attrs={
                    "long_name": "Sea water temperature",
                    "units": "degC",
                },
            )
        },
        coords={
            "time": ("time", times),
            "depth": ("depth", depths, {"units": "m", "positive": "down"}),
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
        attrs={
            "source": "INCOIS Live Access Server - Gridded ARGO (argo_10dv)",
            "time_origin": "days since 15-JAN-1901 00:00:00",
            "bad_flag_original": str(BAD_FLAG),
            "note": "Bad flags replaced with NaN. Generated by convert_argo_txt_to_nc.py",
        },
    )
    return ds


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def convert_year(year: int, dry_run: bool = False, verbose: bool = True) -> None:
    txt_path = ARGO_DIR / f"argo_{year}.txt"
    if not txt_path.exists():
        print(f"  SKIP -- {txt_path} not found")
        return

    print(f"\n{'='*60}")
    print(f"  YEAR {year}")
    print(f"{'='*60}")

    df = parse_txt(txt_path, verbose=verbose)

    # Split by YYYY-MM and write one NetCDF per month
    df["ym"] = df["date"].dt.to_period("M")
    for ym, grp in df.groupby("ym"):
        out_path = ARGO_DIR / f"argo_{ym}.nc"
        if verbose:
            print(f"\n  month {ym}  ->  {out_path.name}", flush=True)
            print(f"    rows {len(grp):,}   dates {grp['date'].nunique()}", flush=True)

        if dry_run:
            print("    [dry-run] skipping write")
            continue

        ds = df_to_xarray(grp.drop(columns=["ym"]))
        encoding = {
            "TEMP": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32(np.nan),
            }
        }
        ds.to_netcdf(out_path, encoding=encoding)
        size_mb = out_path.stat().st_size / 1e6
        print(f"    wrote  {out_path}  ({size_mb:.1f} MB)")

    print(f"\n  done year {year}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert INCOIS LAS ARGO text exports to per-month NetCDF."
    )
    ap.add_argument("--year", type=int, choices=YEARS,
                    help="Process only this year (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse files and print stats but write nothing")
    args = ap.parse_args()

    years = [args.year] if args.year else YEARS

    print("ARGO TXT -> NetCDF converter")
    print(f"Input dir : {ARGO_DIR.resolve()}")
    print(f"Years     : {years}")
    print(f"Dry run   : {args.dry_run}")

    for year in years:
        convert_year(year, dry_run=args.dry_run)

    if not args.dry_run:
        nc_files = sorted(ARGO_DIR.glob("argo_????-??.nc"))
        print(f"\n{'='*60}")
        print(f"Done. {len(nc_files)} NetCDF files written:")
        for f in nc_files:
            print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
        print()
        print("Next steps:")
        print("  1. Run the main pipeline:")
        print("       python scripts/01_harmonize_real.py    # or 01_harmonize.py")
        print("       python scripts/02_build_dataset.py")
        print("  2. Edit ARGO_PATH + ARGO_VAR in scripts/02b_argo_colocate.py:")
        print('       ARGO_PATH = "data/raw/argo/argo_*.nc"')
        print('       ARGO_VAR  = "TEMP"')
        print("  3. python scripts/02b_argo_colocate.py --dry-run")
        print("  4. python scripts/02b_argo_colocate.py")


if __name__ == "__main__":
    main()
