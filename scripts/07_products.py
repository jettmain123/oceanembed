"""Stage 07 -- validate the operational products, not just the temperature field.

    python scripts/07_products.py

Forecasters do not act on a temperature field. They act on:

  D26   depth of the 26 degC isotherm
  TCHP  tropical cyclone heat potential -- the quantity that predicts whether a
        storm intensifies rapidly (>~50 kJ/cm2 is the danger threshold)
  MLD   mixed layer depth

All three come straight out of the profile we already predict. This script
computes them from our reconstruction AND from the GLORYS reference on the same
holdout, and reports how closely they agree -- so the claim is measured rather
than asserted.

It also computes subsurface temperature anomalies and flags marine heatwaves
against a per-location climatology built from the training days only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.inference import Predictor
from oceanembed.products import (anomaly, climatology, d26, heatwave_flag, mld, tchp)


def agreement(name, unit, ref, pred, thresholds=None):
    m = np.isfinite(ref) & np.isfinite(pred)
    r, p = ref[m], pred[m]
    err = p - r
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(r, p)[0, 1]) if r.size > 2 and r.std() > 1e-9 else float("nan")
    row = {"product": name, "unit": unit, "n": int(m.sum()), "rmse": rmse,
           "bias": bias, "corr": corr,
           "ref_mean": float(r.mean()), "ref_std": float(r.std())}
    print(f"  {name:6s} n={m.sum():6d}  corr {corr:6.3f}  RMSE {rmse:8.2f} {unit:8s} "
          f"bias {bias:+7.2f}  (reference mean {r.mean():.1f})")
    if thresholds:
        for t in thresholds:
            hit_r, hit_p = r >= t, p >= t
            tp = int((hit_r & hit_p).sum()); fp = int((~hit_r & hit_p).sum())
            fn = int((hit_r & ~hit_p).sum())
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            print(f"         above {t:g} {unit}: precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}")
            row.setdefault("thresholds", []).append(
                {"value": t, "precision": prec, "recall": rec, "f1": f1,
                 "n_reference_above": int(hit_r.sum())})
    return row


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    dpath = resolve(cfg["paths"]["dataset"])
    if not dpath.exists():
        raise SystemExit(f"missing {dpath} -- run scripts/02_build_dataset.py first")

    d = np.load(dpath, allow_pickle=True)
    arX, arY, arM = d["arX"], d["arY"], d["arM"]
    trY, trM = d["trY"], d["trM"]
    depths = np.asarray(d["depths"], dtype=np.float64)

    pred_ = Predictor.load(cfg)
    print(pred_)
    y = pred_.predict(arX)

    print(f"\nOperational products on {arY.shape[0]} holdout profiles")
    print("predicted vs the same quantity computed from GLORYS:\n")
    rows = []
    rows.append(agreement("D26", "m", d26(arY, depths), d26(y, depths)))
    rows.append(agreement("TCHP", "kJ/cm2", tchp(arY, depths), tchp(y, depths),
                          thresholds=[50, 80, 100]))
    rows.append(agreement("MLD", "m", mld(arY, depths), mld(y, depths)))

    print("\n  TCHP above ~50 kJ/cm2 is the rule-of-thumb rapid-intensification")
    print("  threshold. Precision and recall there matter more than RMSE: they say")
    print("  whether we would have raised the same alarm GLORYS would.")

    # ---- anomalies and marine heatwaves ---------------------------------
    key_tr = np.round(trM[:, 1:3], 2)
    key_ar = np.round(arM[:, 1:3], 2)
    lut, cmean, cp90 = climatology(trY, key_tr, pct=90.0)

    a_pred = anomaly(y, key_ar, lut, cmean)
    a_ref = anomaly(arY, key_ar, lut, cmean)
    hw_pred = heatwave_flag(y, key_ar, lut, cp90)
    hw_ref = heatwave_flag(arY, key_ar, lut, cp90)

    print("\nSubsurface anomaly and marine-heatwave detection")
    print("(climatology from TRAIN days only; heatwave = above the local 90th percentile)\n")
    print(f"  {'depth':>7} {'anom corr':>10} {'anom RMSE':>10} {'HW recall':>10} "
          f"{'HW precision':>13} {'HW base rate':>13}")
    print("  " + "-" * 68)
    hw_rows = []
    for k, z in enumerate(depths):
        m = np.isfinite(a_ref[:, k]) & np.isfinite(a_pred[:, k])
        if m.sum() < 5:
            continue
        c = float(np.corrcoef(a_ref[m, k], a_pred[m, k])[0, 1])
        rm = float(np.sqrt(np.mean((a_pred[m, k] - a_ref[m, k]) ** 2)))
        R, P = hw_ref[m, k], hw_pred[m, k]
        tp = int((R & P).sum()); fp = int((~R & P).sum()); fn = int((R & ~P).sum())
        rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
        base = float(R.mean())
        print(f"  {z:7.0f} {c:10.3f} {rm:10.3f} {rec:10.3f} {prec:13.3f} {base:13.1%}")
        hw_rows.append({"depth_m": float(z), "anomaly_corr": c, "anomaly_rmse": rm,
                        "hw_recall": rec, "hw_precision": prec, "hw_base_rate": base})

    print("\n  Detecting a warm anomaly 100 m DOWN, daily, from satellites alone is")
    print("  something surface-only monitoring cannot do at all. Note the record is")
    print("  short: a real marine-heatwave definition also requires 5 days of")
    print("  persistence, which needs more months than we currently have.")

    out = resolve(cfg["paths"]["outputs"]) / "products_scorecard.json"
    out.write_text(json.dumps({
        "n_holdout": int(arY.shape[0]),
        "products": rows,
        "anomaly_and_heatwave": hw_rows,
        "note": "products computed from the predicted profile and compared against "
                "the same computation on GLORYS; climatology from train days only",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
