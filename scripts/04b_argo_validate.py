"""Stage 04b -- validate against real ARGO floats, three ways.

    python scripts/04b_argo_validate.py [--holdout-only]

Every other number in this project is measured against GLORYS, which is what
trained us. That answers "did we learn our teacher?", not "are we right".

ARGO floats are physical instruments that descended through the water column.
They never entered training. This script co-locates them with both

    our reconstruction   (predicted from the surface patch on that day)
    GLORYS              (the reanalysis value at that same cell and day)

and reports the error of each against the float. The gap between the two is the
honest cost of replacing a supercomputer assimilation with a satellite-only
network -- and GLORYS's own error against ARGO is the floor nobody trained on
GLORYS can beat.

Unlike 02b, this writes nothing into dataset.npz; it only reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.dataset import split_train_val_argo
from oceanembed.harmonize import interp_to_depths, normalize_coords, regrid_to_target
from oceanembed.inference import Predictor
from oceanembed.loaders import open_any, pick_var

ARGO_PATH = "data/raw/argo/argo_????-??.nc"
ARGO_VAR = "TEMP"
MAX_PROFILES = 40000


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else float("nan")


def bias(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(a[m] - b[m])) if m.sum() else float("nan")


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() < 1e-9 or b[m].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-only", action="store_true", default=True)
    ap.add_argument("--all-days", dest="holdout_only", action="store_false",
                    help="include ARGO days the model trained on (optimistic)")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    hpath = resolve(cfg["paths"]["harmonized"])
    out_dir = resolve(cfg["paths"]["outputs"])
    if not hpath.exists():
        raise SystemExit(f"missing {hpath}")

    print(f"opening ARGO: {ARGO_PATH}")
    ads = open_any(ARGO_PATH)
    argo = normalize_coords(pick_var(ads, "temp", ARGO_VAR))
    argo = interp_to_depths(regrid_to_target(argo, cfg), cfg)
    print(f"  {argo.name!r} regridded -> {dict(argo.sizes)}")

    ds = xr.open_dataset(hpath)
    svars = list(cfg["surface_vars"])
    P = int(cfg["patch"]["size"])
    half = P // 2
    depths = np.asarray(ds["depth"].values, dtype=np.float64)
    lat = np.asarray(ds["lat"].values, dtype=np.float32)
    lon = np.asarray(ds["lon"].values, dtype=np.float32)
    land = np.asarray(ds["land_mask"].values) > 0.5

    model_t = np.asarray(ds["time"].values, dtype="datetime64[D]")
    argo_t = np.asarray(argo["time"].values, dtype="datetime64[D]")
    nearest = np.array([int(np.argmin(np.abs(model_t - t))) for t in argo_t])

    tr_d, va_d, ho_d = split_train_val_argo(model_t.size, cfg)
    on_holdout = np.isin(nearest, ho_d)
    print(f"\nARGO times {argo_t.size}; {int(on_holdout.sum())} land on holdout days "
          f"(model days {ho_d.min()}-{ho_d.max()}, "
          f"{model_t[ho_d.min()]} .. {model_t[ho_d.max()]})")

    if args.holdout_only:
        if not on_holdout.any():
            raise SystemExit("no ARGO times on holdout days -- rerun with --all-days "
                             "and label the result as in-sample-surface")
        keep = np.nonzero(on_holdout)[0]
        argo = argo.isel(time=keep)
        argo_t, nearest = argo_t[keep], nearest[keep]
        print(f"keeping only holdout days: {argo_t.size} ARGO times")

    from numpy.lib.stride_tricks import sliding_window_view

    Xs, Ya, Yg, Ms = [], [], [], []
    for ia, it in enumerate(nearest):
        prof = np.asarray(argo.isel(time=ia).values, dtype=np.float32)
        surf = np.stack([np.asarray(ds[v].isel(time=int(it)).values, np.float32)
                         for v in svars])
        glor = np.asarray(ds["temp"].isel(time=int(it)).values, dtype=np.float32)

        win = sliding_window_view(surf, (P, P), axis=(1, 2))
        good = np.isfinite(win).all(axis=(0, 3, 4))
        c = (slice(half, lat.size - half), slice(half, lon.size - half))
        ok = (good & np.isfinite(prof[:, c[0], c[1]]).all(axis=0)
              & np.isfinite(glor[:, c[0], c[1]]).all(axis=0) & (~land[c]))
        iy, ix = np.nonzero(ok)
        if iy.size == 0:
            continue
        Xs.append(np.moveaxis(win[:, iy, ix, :, :], 0, 1).astype(np.float32))
        Ya.append(prof[:, iy + half, ix + half].T.astype(np.float32))
        Yg.append(glor[:, iy + half, ix + half].T.astype(np.float32))
        Ms.append(np.stack([np.full(iy.size, it, np.float64),
                            lat[iy + half], lon[ix + half]], axis=1))
        if sum(len(x) for x in Xs) >= MAX_PROFILES:
            break

    if not Xs:
        raise SystemExit("no ARGO profiles co-located")

    X = np.concatenate(Xs)[:MAX_PROFILES]
    A = np.concatenate(Ya)[:MAX_PROFILES]     # ARGO truth
    G = np.concatenate(Yg)[:MAX_PROFILES]     # GLORYS at the same cell/day
    M = np.concatenate(Ms)[:MAX_PROFILES]
    print(f"co-located {X.shape[0]} profiles with both ARGO and GLORYS present")

    pred_ = Predictor.load(cfg)
    print(pred_)
    Y = pred_.predict(X, meta=M)               # our reconstruction

    print("\nError against real ARGO floats, by depth")
    print("(GLORYS is the reanalysis we trained on -- its own error is the floor)\n")
    print(f"{'depth':>7} {'n':>7} {'ours':>8} {'GLORYS':>8} {'gap':>8} "
          f"{'our bias':>9} {'our corr':>9}")
    print("-" * 62)
    rows = []
    for k, z in enumerate(depths):
        m = np.isfinite(A[:, k]) & np.isfinite(Y[:, k]) & np.isfinite(G[:, k])
        if m.sum() < 5:
            continue
        ro = rmse(Y[:, k], A[:, k])
        rg = rmse(G[:, k], A[:, k])
        print(f"{z:7.0f} {int(m.sum()):7d} {ro:8.2f} {rg:8.2f} {ro - rg:+8.2f} "
              f"{bias(Y[:, k], A[:, k]):+9.2f} {corr(Y[:, k], A[:, k]):9.3f}")
        rows.append({"depth_m": float(z), "n": int(m.sum()), "rmse": ro,
                     "glorys_rmse": rg, "gap": ro - rg,
                     "bias": bias(Y[:, k], A[:, k]),
                     "corr": corr(Y[:, k], A[:, k])})

    ours = float(np.mean([r["rmse"] for r in rows]))
    glo = float(np.mean([r["glorys_rmse"] for r in rows]))
    print("-" * 62)
    print(f"{'mean':>7} {'':>7} {ours:8.2f} {glo:8.2f} {ours - glo:+8.2f}")

    pct = 100 * (ours - glo) / glo if glo else float("nan")
    print(f"\n  Our RMSE against real floats: {ours:.2f} C")
    print(f"  GLORYS's own RMSE against the same floats: {glo:.2f} C")
    print(f"  We are {pct:.0f}% worse than the supercomputer reanalysis we imitate,")
    print("  using only satellite-observable surface fields and seconds of compute.")

    payload = {
        "n_profiles": int(X.shape[0]),
        "n_argo_days": int(argo_t.size),
        "holdout_only": bool(args.holdout_only),
        "our_rmse_vs_argo_degC": ours,
        "glorys_rmse_vs_argo_degC": glo,
        "excess_over_glorys_degC": ours - glo,
        "excess_over_glorys_pct": pct,
        "per_depth": rows,
        "note": "ARGO floats never entered training. GLORYS is the label we "
                "trained on, so its error against ARGO is the floor for any "
                "model distilled from it.",
    }
    out = out_dir / "argo_validation.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    ds.close()


if __name__ == "__main__":
    main()
