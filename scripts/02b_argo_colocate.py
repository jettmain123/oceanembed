"""Stage 02b (REAL) -- splice real ARGO profiles into dataset.npz as the holdout.

    python scripts/02b_argo_colocate.py [--dry-run]

This is the credibility centrepiece: after this runs, the numbers in
scorecard.json are measured against real independent observations that the model
never saw, not against held-out reanalysis.

WHAT IT DOES
  1. opens the gridded ARGO file (INCOIS Live Access Server),
  2. puts it on the same 0.25 deg grid and the 15 standard depths,
  3. for every ARGO cell/day with a full valid profile, cuts the SURFACE patch
     for that same cell/day out of harmonized.nc,
  4. replaces arX / arY / arM in dataset.npz -- trX/vaX and the scalers are left
     untouched, so the model does not have to be retrained.

HOW TO USE
  1. Download gridded ARGO for the same box and window as harmonized.nc.
  2. Set ARGO_PATH and ARGO_VAR in the EDIT BLOCK.
  3. --dry-run to check variables and the overlap, then run for real.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.dataset import channel_names
from oceanembed.harmonize import interp_to_depths, normalize_coords, regrid_to_target
from oceanembed.loaders import open_any

# ===========================================================================
# EDIT BLOCK
# ===========================================================================

ARGO_PATH = "data/raw/argo/*.nc"      # gridded ARGO from INCOIS LAS
ARGO_VAR: str | None = None           # e.g. "TEMP" / "temperature"; None autodetects
MAX_PROFILES = 20000                  # cap so the holdout stays a sane size

# ===========================================================================
# END EDIT BLOCK
# ===========================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    hpath = resolve(cfg["paths"]["harmonized"])
    dpath = resolve(cfg["paths"]["dataset"])
    for p in (hpath, dpath):
        if not p.exists():
            raise SystemExit(f"missing {p} -- run stages 01 and 02 first")

    print(f"opening ARGO: {ARGO_PATH}")
    ads = open_any(ARGO_PATH)
    print(f"  variables {list(ads.data_vars)}")
    print(f"  dims      {dict(ads.sizes)}")
    from oceanembed.loaders import pick_var

    argo = normalize_coords(pick_var(ads, "temp", ARGO_VAR))
    print(f"  using     {argo.name!r}  dims {dict(argo.sizes)}")

    argo = regrid_to_target(argo, cfg)
    argo = interp_to_depths(argo, cfg)

    ds = xr.open_dataset(hpath)
    svars = list(cfg["surface_vars"])
    P = int(cfg["patch"]["size"])
    half = P // 2
    add_coords = bool(cfg["patch"].get("add_coords", True))
    depths = np.asarray(ds["depth"].values, dtype=np.float32)
    lat = np.asarray(ds["lat"].values, dtype=np.float32)
    lon = np.asarray(ds["lon"].values, dtype=np.float32)
    land = np.asarray(ds["land_mask"].values) > 0.5
    dom = cfg["domain"]
    lat_n = (lat - dom["lat_min"]) / (dom["lat_max"] - dom["lat_min"]) * 2 - 1
    lon_n = (lon - dom["lon_min"]) / (dom["lon_max"] - dom["lon_min"]) * 2 - 1

    # match each ARGO time to the nearest day in the harmonized cube
    model_t = np.asarray(ds["time"].values, dtype="datetime64[D]")
    argo_t = np.asarray(argo["time"].values, dtype="datetime64[D]") if "time" in argo.dims else None
    if argo_t is None:
        raise SystemExit("ARGO file has no time dimension -- cannot co-locate")
    nearest = np.array([int(np.argmin(np.abs(model_t - t))) for t in argo_t])
    gap_days = np.array([int(abs((model_t[j] - t) / np.timedelta64(1, "D"))) for t, j in zip(argo_t, nearest)])
    print(f"\nARGO times {argo_t.size}, matched to model days with median gap {np.median(gap_days):.0f} d")

    if args.dry_run:
        print("dry run -- nothing written.")
        return

    from numpy.lib.stride_tricks import sliding_window_view

    Xs, Ys, Ms = [], [], []
    for ia, it in enumerate(nearest):
        prof = np.asarray(argo.isel(time=ia).values, dtype=np.float32)     # (nz,ny,nx)
        surf = np.stack([np.asarray(ds[v].isel(time=int(it)).values, np.float32) for v in svars])

        win = sliding_window_view(surf, (P, P), axis=(1, 2))
        good_patch = np.isfinite(win).all(axis=(0, 3, 4))
        c = (slice(half, lat.size - half), slice(half, lon.size - half))
        ok = good_patch & np.isfinite(prof[:, c[0], c[1]]).all(axis=0) & (~land[c])
        iy, ix = np.nonzero(ok)
        if iy.size == 0:
            continue

        patches = np.moveaxis(win[:, iy, ix, :, :], 0, 1).astype(np.float32)
        if add_coords:
            doy = float(np.datetime64(model_t[int(it)], "D").astype(object).timetuple().tm_yday)
            extra = np.empty((patches.shape[0], 4, P, P), np.float32)
            extra[:, 0] = lat_n[iy + half][:, None, None]
            extra[:, 1] = lon_n[ix + half][:, None, None]
            extra[:, 2] = np.float32(np.sin(2 * np.pi * doy / 365.25))
            extra[:, 3] = np.float32(np.cos(2 * np.pi * doy / 365.25))
            patches = np.concatenate([patches, extra], axis=1)

        Xs.append(patches)
        Ys.append(prof[:, iy + half, ix + half].T.astype(np.float32))
        Ms.append(np.stack([np.full(iy.size, it, np.float64), lat[iy + half], lon[ix + half]], axis=1))
        if sum(len(x) for x in Xs) >= MAX_PROFILES:
            break

    if not Xs:
        raise SystemExit("no ARGO profiles co-located -- check the box, window and variable name")

    arX = np.concatenate(Xs)[:MAX_PROFILES]
    arY = np.concatenate(Ys)[:MAX_PROFILES]
    arM = np.concatenate(Ms)[:MAX_PROFILES]
    print(f"co-located {arX.shape[0]} real ARGO profiles  X{arX.shape} Y{arY.shape}")

    d = dict(np.load(dpath, allow_pickle=True))
    old = d["arX"].shape[0]
    d["arX"], d["arY"], d["arM"] = arX, arY, arM
    d["argo_source"] = np.array(ARGO_PATH)
    np.savez(dpath, **d)

    print(f"replaced the synthetic holdout ({old} profiles) with real ARGO in {dpath}")
    print("train/val splits and scalers untouched -- no retraining needed.")
    print("channels:", channel_names(cfg))
    print("\nnext: python scripts/04_evaluate.py")


if __name__ == "__main__":
    main()
