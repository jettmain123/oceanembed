"""Stage 03b -- classical baseline for the comparison story.

    python scripts/03b_baseline.py [--model rf|ridge] [--trees N] [--max-train N]

Trains one RandomForest per depth on the PATCH-CENTRE features only -- i.e. the
same information a per-cell tabular model (samples.parquet) would have. This is
the honest control: if OceanEmbed beats it, the gain comes from the N x N spatial
context (eddies and fronts), not from having more data or a bigger fit.

Writes outputs/baseline_scorecard.json in the same shape as scorecard.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.dataset import channel_names
from oceanembed.evaluate import format_table, scorecard


def centre_features(X: np.ndarray) -> np.ndarray:
    """The centre pixel of every channel -- no spatial context at all."""
    c = X.shape[2] // 2
    return X[:, :, c, c].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf", "ridge"], default="rf")
    ap.add_argument("--trees", type=int, default=120)
    ap.add_argument("--max-depth", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=20000, help="subsample train rows for speed")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    dpath = resolve(cfg["paths"]["dataset"])
    if not dpath.exists():
        raise SystemExit(f"missing {dpath} -- run scripts/02_build_dataset.py first")
    d = np.load(dpath, allow_pickle=True)

    trX, trY = centre_features(d["trX"]), d["trY"]
    arX, arY = centre_features(d["arX"]), d["arY"]
    depths = d["depths"]
    names = channel_names(cfg)

    rng = np.random.default_rng(cfg["train"]["seed"])
    if args.max_train and len(trX) > args.max_train:
        idx = rng.choice(len(trX), args.max_train, replace=False)
        trX, trY = trX[idx], trY[idx]

    print(f"baseline: {args.model} on {trX.shape[0]} rows x {trX.shape[1]} centre features")
    print("features:", names)

    t0 = time.time()
    if args.model == "rf":
        from sklearn.ensemble import RandomForestRegressor

        pred = np.empty((arX.shape[0], depths.size), dtype=np.float32)
        importances = np.zeros((depths.size, trX.shape[1]), dtype=np.float32)
        for k, z in enumerate(depths):
            rf = RandomForestRegressor(
                n_estimators=args.trees, max_depth=args.max_depth,
                n_jobs=-1, random_state=int(cfg["train"]["seed"]), min_samples_leaf=4,
            )
            rf.fit(trX, trY[:, k])
            pred[:, k] = rf.predict(arX)
            importances[k] = rf.feature_importances_
            print(f"  depth {int(z):5d} m fitted")
    else:
        from sklearn.linear_model import RidgeCV

        rg = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(trX, trY)
        pred = rg.predict(arX).astype(np.float32)
        importances = np.abs(rg.coef_).astype(np.float32)
    dt = time.time() - t0

    card = scorecard(arY, pred, depths, label=f"baseline_{args.model}_centre_only")
    out = resolve(cfg["paths"]["baseline_scorecard"])
    card["seconds"] = round(dt, 1)
    card["n_train"] = int(trX.shape[0])
    card["features"] = names
    out.write_text(json.dumps(card, indent=2), encoding="utf-8")

    print(f"\ntrained in {dt:.1f} s -- evaluated on the ARGO-style holdout ({arX.shape[0]} profiles)")
    print(format_table(card))
    print(f"\nwrote {out}")

    top = np.argsort(-importances.mean(axis=0))[:5]
    print("\nmost useful centre features (mean over depths):")
    for i in top:
        print(f"  {names[i]:8s} {importances.mean(axis=0)[i]:.3f}")


if __name__ == "__main__":
    main()
