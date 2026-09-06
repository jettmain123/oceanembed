"""Stage 06 -- export everything the website needs as static files.

    python scripts/06_export_web.py [--out web/data] [--max-days N]

Runs the model over EVERY valid ocean cell on every day and writes plain binary
files. The site then needs no backend at all: the model is deterministic over a
fixed domain, so every answer it could give is computed once, here.

Output:
    index.json     dates, depths, grid, scaling, headline metrics
    day_000.bin    uint16 [2, 15, ny, nx]  temperature  [prediction, reference]
                   then float32 [2, 3, ny, nx] products [TCHP, D26, MLD], appended

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
from oceanembed.products import d26, mld, tchp
from oceanembed.risk import DISCLAIMER, build_alerts, risk_index

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
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth day across the WHOLE record, so the viewer "
                         "can range over every year rather than one recent month")
    ap.add_argument("--dense-tail", type=int, default=0,
                    help="always keep the last N days at full daily cadence, on top "
                         "of the stride -- day-by-day detail where events are")
    ap.add_argument("--mc", type=int, default=10,
                    help="Monte-Carlo passes for the uncertainty layer")
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
    # Which days end up in the viewer.
    #
    #   --stride N       every Nth day over the whole record, so the date control
    #                    spans every year we hold rather than one recent month
    #   --dense-tail M   plus the last M days at full daily cadence, because that
    #                    is where the cyclone season and the holdout live
    #   --max-days K     hard cap, keeping the most recent K of whatever survives
    #
    # Days whose surface stack is entirely missing are dropped rather than
    # exported blank: 90 days of Q1 2022 have no wind at all, and a date the
    # viewer cannot draw is worse than a date it does not offer.
    keep = set(range(0, times.size, max(1, args.stride)))
    if args.dense_tail:
        keep |= set(range(max(0, times.size - args.dense_tail), times.size))
    day_idx = sorted(keep)
    if args.max_days:
        day_idx = day_idx[-args.max_days:]

    usable = []
    for it in day_idx:
        ok = True
        for v in svars:
            a = ds[v].isel(time=it).values
            if not np.isfinite(a).any():
                ok = False
                break
        if ok:
            usable.append(it)
    dropped = len(day_idx) - len(usable)
    if dropped:
        print(f"  dropping {dropped} day(s) with a fully missing surface variable")
    day_idx = usable
    n_days = len(day_idx)

    dom = cfg["domain"]
    lat_n = (lat - dom["lat_min"]) / (dom["lat_max"] - dom["lat_min"]) * 2 - 1
    lon_n = (lon - dom["lon_min"]) / (dom["lon_max"] - dom["lon_min"]) * 2 - 1

    print(f"exporting {n_days} days, {ny} x {nx} cells, {nz} depths -> {out_dir}")
    dates = []
    risk_days = []      # TCHP per day, to build the climatology the risk rule needs
    pending = []        # (path, prod array) -- risk is filled in after the climatology
    for seq, it in enumerate(day_idx):
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
            # meta is REQUIRED for an anomaly-trained checkpoint -- the model
            # returns a departure from the local climatology and cannot place it
            # without knowing which cell and day each row came from.
            meta = np.stack([np.full(iy.size, it, np.float64),
                             lat[iy + half], lon[ix + half]], axis=1)
            y, mc_std = pred_.predict_mc(patches, n=args.mc, meta=meta)      # (n,nz) each
            pred[:, iy + half, ix + half] = y.T

        # Operational products, computed per cell from the SAME profiles. This is
        # what a forecaster acts on -- TCHP above ~50 kJ/cm2 is the cyclone
        # rapid-intensification threshold.
        prod = np.full((2, 5, ny, nx), np.nan, np.float32)
        for w, field in ((0, pred), (1, ref)):
            flat = field.reshape(nz, -1).T                                  # (ny*nx, nz)
            good = np.isfinite(flat).all(axis=1)
            if good.any():
                g = flat[good]
                prod[w, 0].reshape(-1)[good] = tchp(g, depths)
                prod[w, 1].reshape(-1)[good] = d26(g, depths)
                prod[w, 2].reshape(-1)[good] = mld(g, depths)
        # uncertainty and rule-based risk, for the prediction only -- the
        # reference has neither, so those slots stay NaN
        if iy.size:
            k100 = int(np.argmin(np.abs(depths - 100)))
            prod[0, 3][iy + half, ix + half] = mc_std[:, k100]
        risk_days.append(prod[0, 0].copy())

        stack = np.stack([encode(pred), encode(ref)])                      # (2,nz,ny,nx)
        # named by POSITION in the exported series, not by absolute day index --
        # the viewer fetches day_000.bin upward and pairs them with index.json
        pending.append((out_dir / f"day_{seq:03d}.bin", stack, prod))
        dates.append(str(np.datetime64(times[it], "D")))
        if (it + 1) % 10 == 0 or it == n_days - 1:
            print(f"  {seq + 1}/{n_days} days")

    # Second pass: the risk rule needs to know what is NORMAL at each location,
    # because an absolute TCHP threshold flags nearly the whole basin every day.
    alerts_by_day = {}
    if not args.index_only and risk_days:
        clim = np.nanmean(np.stack(risk_days), axis=0)
        for (path, stack, prod), date in zip(pending, dates):
            R = risk_index(prod[0, 0], prod[0, 3], tchp_anomaly=prod[0, 0] - clim)
            prod[0, 4] = R.astype(np.float32)
            prod[1, 4] = (prod[0, 0] - clim).astype(np.float32)   # the anomaly itself
            path.write_bytes(stack.tobytes() + prod.tobytes())
            a = build_alerts(date, R, prod[0, 0], prod[0, 3], lat, lon)
            if a:
                alerts_by_day[date] = a

    card = load_card(resolve(cfg["paths"]["scorecard"])) or {}
    base = load_card(resolve(cfg["paths"]["baseline_scorecard"])) or {}
    index = {
        "dates": dates,
        "depths": [float(z) for z in depths],
        "shape": [int(nz), int(ny), int(nx)],
        "lat": [float(lat[0]), float(lat[-1])],
        "lon": [float(lon[0]), float(lon[-1])],
        "encoding": {"t_min": T_MIN, "scale": SCALE, "missing": 0},
        "products": {
            "names": ["TCHP", "D26", "MLD", "UNC", "RISK"],
            "units": ["kJ/cm2", "m", "m", "degC", "level"],
            "labels": ["Cyclone heat potential", "26 degC isotherm depth",
                       "Mixed layer depth", "Model uncertainty at 100 m",
                       "RI risk level"],
            "byte_offset": int(2 * nz * ny * nx * 2),   # after the uint16 temperature block
            "dtype": "float32",
            "shape": [2, 5, int(ny), int(nx)],
        },
        "alerts": alerts_by_day,
        "disclaimer": DISCLAIMER,
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
