"""Stage 06 -- export everything the website needs as static files.

    python scripts/06_export_web.py [--out web/data] [--max-days N]

Runs the model over EVERY valid ocean cell on every day and writes plain binary
files. The site then needs no backend at all: the model is deterministic over a
fixed domain, so every answer it could give is computed once, here.

Output:
    index.json     dates, depths, grid, scaling, headline metrics
    day_000.bin    uint16 [2, 15, ny, nx] -- [prediction, reference]

uint16 with a scale/offset rather than float16, because Uint16Array works in
every browser while Float16Array does not. 0 is the missing-data sentinel, so
real values map to 1..65534 -- about 0.001 degC of precision, far finer than the
model's error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.evaluate import load_card
from oceanembed.inference import Predictor

T_MIN, T_MAX = -2.0, 40.0          # covers every ocean temperature we will meet
SCALE = (T_MAX - T_MIN) / 65533.0


def encode(arr: np.ndarray) -> np.ndarray:
    """float degC -> uint16, 0 meaning missing."""
    out = np.zeros(arr.shape, dtype=np.uint16)
    ok = np.isfinite(arr)
    q = np.clip((arr[ok] - T_MIN) / SCALE, 0, 65532) + 1
    out[ok] = q.astype(np.uint16)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/data")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--index-only", action="store_true",
                    help="rewrite index.json only; the day files are unchanged")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hpath = resolve(cfg["paths"]["harmonized"])
    if not hpath.exists():
        raise SystemExit(f"missing {hpath} -- run stage 01 first")
    ds = xr.open_dataset(hpath)
    pred_ = Predictor.load(cfg)
    print(pred_)

    svars = list(cfg["surface_vars"])
    P = int(cfg["patch"]["size"])
    half = P // 2
    add_coords = bool(cfg["patch"].get("add_coords", True))
    lat = np.asarray(ds["lat"].values, np.float32)
    lon = np.asarray(ds["lon"].values, np.float32)
    depths = np.asarray(ds["depth"].values, np.float32)
    land = np.asarray(ds["land_mask"].values) > 0.5
    ny, nx, nz = lat.size, lon.size, depths.size
    times = np.asarray(ds["time"].values)
    n_days = times.size if args.max_days is None else min(args.max_days, times.size)

    dom = cfg["domain"]
    lat_n = (lat - dom["lat_min"]) / (dom["lat_max"] - dom["lat_min"]) * 2 - 1
    lon_n = (lon - dom["lon_min"]) / (dom["lon_max"] - dom["lon_min"]) * 2 - 1

    print(f"exporting {n_days} days, {ny} x {nx} cells, {nz} depths -> {out_dir}")
    dates = []
    for it in range(n_days):
        if args.index_only:
            dates.append(str(np.datetime64(times[it], "D")))
            continue
        surf = np.stack([np.asarray(ds[v].isel(time=it).values, np.float32) for v in svars])
        ref = np.asarray(ds["temp"].isel(time=it).values, np.float32)      # (nz,ny,nx)

        win = sliding_window_view(surf, (P, P), axis=(1, 2))
        ok = np.isfinite(win).all(axis=(0, 3, 4)) & (~land[half:ny - half, half:nx - half])
        iy, ix = np.nonzero(ok)

        pred = np.full((nz, ny, nx), np.nan, np.float32)
        if iy.size:
            patches = np.moveaxis(win[:, iy, ix, :, :], 0, 1).astype(np.float32)
            if add_coords:
                d = np.datetime64(times[it], "D").astype(object)
                extra = np.empty((patches.shape[0], 4, P, P), np.float32)
                extra[:, 0] = lat_n[iy + half][:, None, None]
                extra[:, 1] = lon_n[ix + half][:, None, None]
                extra[:, 2] = np.sin(2 * np.pi * d.timetuple().tm_yday / 365.25)
                extra[:, 3] = np.cos(2 * np.pi * d.timetuple().tm_yday / 365.25)
                patches = np.concatenate([patches, extra], axis=1)
            y = pred_.predict(patches)                                     # (n,nz)
            pred[:, iy + half, ix + half] = y.T

        stack = np.stack([encode(pred), encode(ref)])                      # (2,nz,ny,nx)
        (out_dir / f"day_{it:03d}.bin").write_bytes(stack.tobytes())
        dates.append(str(np.datetime64(times[it], "D")))
        if (it + 1) % 10 == 0 or it == n_days - 1:
            print(f"  {it + 1}/{n_days} days")

    card = load_card(resolve(cfg["paths"]["scorecard"])) or {}
    base = load_card(resolve(cfg["paths"]["baseline_scorecard"])) or {}
    index = {
        "dates": dates,
        "depths": [float(z) for z in depths],
        "shape": [int(nz), int(ny), int(nx)],
        "lat": [float(lat[0]), float(lat[-1])],
        "lon": [float(lon[0]), float(lon[-1])],
        "encoding": {"t_min": T_MIN, "scale": SCALE, "missing": 0},
        "metrics": {
            "mean_corr": card.get("mean_corr"),
            "mean_rmse": card.get("mean_rmse"),
            "baseline_rmse": base.get("mean_rmse"),
            "bands": card.get("bands"),
            "per_depth": [
                {"depth_m": r["depth_m"], "corr": r["corr"], "rmse": r["rmse"]}
                for r in card.get("per_depth", [])
            ],
            "vs_climatology": card.get("vs_climatology", {}).get("per_depth", []),
            "mean_skill_vs_clim": card.get("vs_climatology", {}).get("mean_skill_vs_clim"),
        },
        "domain": f"{dom['lat_min']}-{dom['lat_max']}N, {dom['lon_min']}-{dom['lon_max']}E",
        "n_holdout": card.get("n_profiles"),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")

    total = sum(f.stat().st_size for f in out_dir.glob("*")) / 1e6
    print(f"\nwrote {len(dates)} day files + index.json  ({total:.1f} MB total)")
    print(f"per day: {(out_dir / 'day_000.bin').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
