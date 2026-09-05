"""Stage 03d -- how well does the model cope when a satellite is missing?

    python scripts/03d_sensor_robustness.py

Satellites go down. Products have gaps. SMAP salinity is patchy, altimeters miss
swaths, a sensor gets decommissioned. An assimilation system like GLORYS needs
its full input stack and a supercomputer rerun to handle that; a learned model
can simply degrade.

This measures the degradation honestly: blank each input channel in turn and
report what it costs. Train with `03_train.py --channel-dropout 0.15` first and
the model learns to survive it.

The most useful column is not the RMSE -- it is which variable the model actually
depends on. A channel whose removal costs nothing was never being used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.dataset import channel_names
from oceanembed.evaluate import scorecard
from oceanembed.inference import Predictor
from oceanembed.products import tchp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="store_true",
                    help="also test losing two channels at once")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    dpath = resolve(cfg["paths"]["dataset"])
    if not dpath.exists():
        raise SystemExit(f"missing {dpath} -- run scripts/02_build_dataset.py first")
    d = np.load(dpath, allow_pickle=True)
    arX, arY = d["arX"], d["arY"]
    depths = np.asarray(d["depths"], dtype=np.float64)
    xm = d["xm"]
    names = channel_names(cfg)[:arX.shape[1]]

    pred_ = Predictor.load(cfg)
    print(pred_)
    print(f"holdout {arX.shape[0]} profiles, {len(names)} input channels\n")

    def run(drop):
        """Blank the given channels by setting them to their training mean."""
        X = arX.copy()
        for c in drop:
            X[:, c] = xm[0, c]          # the mean IS the no-information value
        y = pred_.predict(X, meta=d['arM'])
        card = scorecard(arY, y, depths)
        return card, y

    full, y_full = run([])
    base_rmse = full["mean_rmse"]
    base_tchp = tchp(y_full, depths)
    ref_tchp = tchp(arY, depths)
    base_tchp_rmse = float(np.sqrt(np.nanmean((base_tchp - ref_tchp) ** 2)))

    print(f"{'missing input':>16} {'mean RMSE':>10} {'change':>9} {'thermocline':>12} "
          f"{'TCHP RMSE':>10}")
    print("-" * 64)
    print(f"{'nothing (full)':>16} {base_rmse:10.4f} {'--':>9} "
          f"{full['bands']['thermocline_75_300m']['mean_rmse']:12.4f} {base_tchp_rmse:10.2f}")

    rows = []
    for c, nm in enumerate(names):
        card, y = run([c])
        t = tchp(y, depths)
        t_rmse = float(np.sqrt(np.nanmean((t - ref_tchp) ** 2)))
        delta = 100 * (card["mean_rmse"] - base_rmse) / base_rmse
        print(f"{nm:>16} {card['mean_rmse']:10.4f} {delta:+8.1f}% "
              f"{card['bands']['thermocline_75_300m']['mean_rmse']:12.4f} {t_rmse:10.2f}")
        rows.append({"missing": nm, "mean_rmse": card["mean_rmse"],
                     "pct_worse": delta, "mean_corr": card["mean_corr"],
                     "thermocline_rmse": card["bands"]["thermocline_75_300m"]["mean_rmse"],
                     "tchp_rmse": t_rmse})

    pair_rows = []
    if args.pairs:
        print("\nlosing two at once:")
        import itertools
        for a, b in itertools.combinations(range(len(names)), 2):
            card, _ = run([a, b])
            delta = 100 * (card["mean_rmse"] - base_rmse) / base_rmse
            pair_rows.append({"missing": [names[a], names[b]],
                              "mean_rmse": card["mean_rmse"], "pct_worse": delta})
        pair_rows.sort(key=lambda r: -r["pct_worse"])
        for r in pair_rows[:6]:
            print(f"  {r['missing'][0]:>7} + {r['missing'][1]:<7} "
                  f"{r['mean_rmse']:8.4f} {r['pct_worse']:+7.1f}%")

    worst = max(rows, key=lambda r: r["pct_worse"])
    least = min(rows, key=lambda r: r["pct_worse"])
    print(f"\n  most depended on : {worst['missing']}  ({worst['pct_worse']:+.1f}% without it)")
    print(f"  least depended on: {least['missing']}  ({least['pct_worse']:+.1f}% without it)")
    print("\n  The point is not that accuracy drops -- of course it does. The point is")
    print("  that the system still RUNS and still returns a calibrated profile. A")
    print("  physics assimilation cannot do that without a full rerun.")

    out = resolve(cfg["paths"]["outputs"]) / "sensor_robustness.json"
    out.write_text(json.dumps({
        "baseline": {"mean_rmse": base_rmse, "tchp_rmse": base_tchp_rmse},
        "single_channel_missing": rows,
        "pairs_missing": pair_rows,
        "channels": names,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
