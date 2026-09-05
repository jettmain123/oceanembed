"""Stage 08 -- the decision-support layer: TCHP risk, uncertainty, coastal exposure, alerts.

    python scripts/08_risk.py [--replay 2024-10-20:2024-10-25] [--region bay]

Runs the whole chain over every day in the harmonized cube:

    profile -> TCHP + D26 -> MC uncertainty -> rule-based risk -> coastal
    segments near flagged water -> structured alert objects

Everything after the model is a transparent rule. A threshold can be audited and
argued with by a domain expert; a second neural network trained in an afternoon
cannot.

NOTHING HERE IS A WARNING PRODUCT. It shows where the ocean could sustain rapid
intensification. It does not detect cyclones, know anything about the atmosphere,
or recommend any action. The alerts are demonstration objects and are not sent
anywhere.
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
from oceanembed.inference import Predictor
from oceanembed.products import d26, tchp
from oceanembed.risk import (DISCLAIMER, TCHP_HIGH, build_alerts, coastal_exposure,
                             risk_index)

LEVELS = ["LOW", "WATCH", "ELEVATED", "HIGH"]


def day_fields(ds, cfg, pred_, it, mc=12):
    """Predicted TCHP, D26 and MC spread for every valid ocean cell on one day."""
    svars = list(cfg["surface_vars"])
    P = int(cfg["patch"]["size"]); half = P // 2
    add_coords = bool(cfg["patch"].get("add_coords", True))
    lat = np.asarray(ds["lat"].values, np.float32)
    lon = np.asarray(ds["lon"].values, np.float32)
    depths = np.asarray(ds["depth"].values, np.float64)
    land = np.asarray(ds["land_mask"].values) > 0.5
    ny, nx = lat.size, lon.size

    surf = np.stack([np.asarray(ds[v].isel(time=it).values, np.float32) for v in svars])
    win = sliding_window_view(surf, (P, P), axis=(1, 2))
    ok = np.isfinite(win).all(axis=(0, 3, 4)) & (~land[half:ny - half, half:nx - half])
    iy, ix = np.nonzero(ok)

    T = np.full((ny, nx), np.nan); D = np.full((ny, nx), np.nan)
    U = np.full((ny, nx), np.nan)
    if iy.size == 0:
        return T, D, U

    patches = np.moveaxis(win[:, iy, ix, :, :], 0, 1).astype(np.float32)
    if add_coords:
        dom = cfg["domain"]
        t = np.datetime64(ds["time"].values[it], "D").astype(object)
        extra = np.empty((patches.shape[0], 4, P, P), np.float32)
        extra[:, 0] = ((lat[iy + half] - dom["lat_min"]) /
                       (dom["lat_max"] - dom["lat_min"]) * 2 - 1)[:, None, None]
        extra[:, 1] = ((lon[ix + half] - dom["lon_min"]) /
                       (dom["lon_max"] - dom["lon_min"]) * 2 - 1)[:, None, None]
        extra[:, 2] = np.sin(2 * np.pi * t.timetuple().tm_yday / 365.25)
        extra[:, 3] = np.cos(2 * np.pi * t.timetuple().tm_yday / 365.25)
        patches = np.concatenate([patches, extra], axis=1)

    mean, std = pred_.predict_mc(patches, n=mc)
    T[iy + half, ix + half] = tchp(mean, depths)
    D[iy + half, ix + half] = d26(mean, depths)
    # spread at 100 m -- the thermocline, where the model is most input-sensitive
    k100 = int(np.argmin(np.abs(depths - 100)))
    U[iy + half, ix + half] = std[:, k100]
    return T, D, U


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=12, help="Monte-Carlo passes per day")
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    hpath = resolve(cfg["paths"]["harmonized"])
    if not hpath.exists():
        raise SystemExit(f"missing {hpath} -- run stage 01 first")
    ds = xr.open_dataset(hpath)
    pred_ = Predictor.load(cfg)
    lat = np.asarray(ds["lat"].values, np.float32)
    lon = np.asarray(ds["lon"].values, np.float32)
    times = np.asarray(ds["time"].values)
    n = times.size if args.max_days is None else min(args.max_days, times.size)

    print(pred_)
    print(f"\n{DISCLAIMER}\n")
    # Pass 1: TCHP for every day, so we can say what is NORMAL at each location.
    # Without this the absolute 50 kJ/cm2 threshold flags the whole basin every
    # day -- the tropical Indian Ocean simply is that warm.
    print("  building the TCHP climatology (pass 1 of 2)...")
    fields = []
    for it in range(n):
        fields.append(day_fields(ds, cfg, pred_, it, mc=args.mc))
        if (it + 1) % 10 == 0:
            print(f"    {it + 1}/{n} days")
    clim = np.nanmean(np.stack([f[0] for f in fields]), axis=0)
    print("  climatology built; scoring days against it (pass 2 of 2)\n")

    print(f"{'date':>12} {'HIGH cells':>11} {'peak TCHP':>10} {'spread':>8} "
          f"{'segments':>9}  top exposed")
    print("-" * 88)

    daily, all_alerts = [], []
    for it in range(n):
        date = str(np.datetime64(times[it], "D"))
        T, D, U = fields[it]
        A = T - clim                      # anomaly against the local norm
        R = risk_index(T, U, tchp_anomaly=A)
        alerts = build_alerts(date, R, T, U, lat, lon)
        all_alerts.extend(alerts)

        nhigh = int((R >= 3).sum())
        peak = float(np.nanmax(T)) if np.isfinite(T).any() else float("nan")
        spread = float(np.nanmean(U[R >= 3])) if nhigh else float("nan")
        top = ", ".join(a["segment"] for a in alerts[:2]) if alerts else "-"
        print(f"{date:>12} {nhigh:11d} {peak:10.1f} {spread:8.3f} "
              f"{len(alerts):9d}  {top}")
        daily.append({"date": date, "high_cells": nhigh, "peak_tchp": peak,
                      "peak_tchp_anomaly": float(np.nanmax(A)) if np.isfinite(A).any() else None,
                      "mean_spread": spread, "n_segments": len(alerts),
                      "level_counts": {LEVELS[i]: int((R == i).sum()) for i in range(4)}})

    # which stretches of coast came up most often across the record
    tally = {}
    for a in all_alerts:
        k = (a["segment"], a["region"])
        e = tally.setdefault(k, {"days": 0, "peak": 0.0})
        e["days"] += 1
        e["peak"] = max(e["peak"], a["peak_tchp_kj_cm2"])
    print(f"\nCoastal segments nearest flagged water, over {n} days:")
    print(f"  {'segment':34s} {'region':26s} {'days':>5} {'peak TCHP':>10}")
    print("  " + "-" * 78)
    for (seg, reg), e in sorted(tally.items(), key=lambda kv: -kv[1]["days"])[:12]:
        print(f"  {seg:34s} {reg:26s} {e['days']:5d} {e['peak']:10.1f}")

    abs_only = sum(int((risk_index(f[0], f[2]) >= 3).sum()) for f in fields)
    with_anom = sum(d["level_counts"]["HIGH"] for d in daily)
    print("\nWhy the anomaly term is there:")
    print(f"  absolute TCHP >= {TCHP_HIGH:g} alone     : {abs_only:,} cell-days flagged HIGH")
    print(f"  also requiring a local anomaly: {with_anom:,} cell-days")
    print("  The first number is most of the basin, most days. The tropical Indian")
    print("  Ocean is above the rule-of-thumb threshold nearly all the time, so an")
    print("  absolute threshold alone does not discriminate. That is a genuine")
    print("  result about this ocean, not a tuning convenience.")

    out = resolve(cfg["paths"]["outputs"]) / "risk_summary.json"
    out.write_text(json.dumps({
        "disclaimer": DISCLAIMER,
        "tchp_high_threshold": TCHP_HIGH,
        "daily": daily,
        "alerts": all_alerts,
        "segment_tally": [{"segment": s, "region": r, **e} for (s, r), e in tally.items()],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({len(all_alerts)} demonstration alert objects)")
    print("\nThese alert objects are NOT transmitted anywhere and are not connected")
    print("to any alerting system. They exist to show the shape of the output.")


if __name__ == "__main__":
    main()
