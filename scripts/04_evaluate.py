"""Stage 04 -- evaluate on the independent ARGO-style holdout.

    python scripts/04_evaluate.py [--backend auto|torch|numpy]

Writes outputs/scorecard.json (Contract E) plus three judge-facing figures:
    skill_vs_depth.png     corr and RMSE against depth, model vs baseline
    example_profiles.png   predicted vs reference profiles
    spatial_map.png        where the error lives, at one depth on one day
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.evaluate import compare_table, format_table, load_card, scorecard
from oceanembed.inference import Predictor


def plot_skill(card, base, depths, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    corr = [r["corr"] for r in card["per_depth"]]
    rmse = [r["rmse"] for r in card["per_depth"]]
    bias = [r["bias"] for r in card["per_depth"]]

    ax[0].plot(corr, depths, "o-", color="#1f6feb", label="OceanEmbed")
    ax[1].plot(rmse, depths, "o-", color="#1f6feb", label="OceanEmbed")
    ax[2].plot(bias, depths, "o-", color="#1f6feb", label="OceanEmbed")
    if base:
        ax[0].plot([r["corr"] for r in base["per_depth"]], depths, "s--", color="#999", label="RF baseline")
        ax[1].plot([r["rmse"] for r in base["per_depth"]], depths, "s--", color="#999", label="RF baseline")
        ax[2].plot([r["bias"] for r in base["per_depth"]], depths, "s--", color="#999", label="RF baseline")

    for a, t, x in zip(ax, ["correlation", "RMSE (degC)", "bias (degC)"], [None, None, 0.0]):
        a.set_xlabel(t)
        a.grid(alpha=0.3)
        if x is not None:
            a.axvline(x, color="k", lw=0.8)
    ax[0].set_ylabel("depth (m)")
    ax[0].invert_yaxis()
    ax[0].set_yscale("symlog", linthresh=50)
    ax[0].set_ylim(1100, -1)
    ax[0].legend(loc="lower left", fontsize=9)
    # mark the thermocline band -- the physics talking point
    for a in ax:
        a.axhspan(75, 300, color="orange", alpha=0.08)
    fig.suptitle("Skill against depth on the independent holdout (shaded: thermocline 75-300 m)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_profiles(y_true, y_pred, meta, depths, path, n=6, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(y_true), size=min(n, len(y_true)), replace=False)
    cols = 3
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.1 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for a, i in zip(axes, idx):
        a.plot(y_true[i], depths, "k-o", ms=3.5, lw=1.6, label="reference")
        a.plot(y_pred[i], depths, "-s", ms=3.5, lw=1.6, color="#1f6feb", label="OceanEmbed")
        rmse = float(np.sqrt(np.mean((y_pred[i] - y_true[i]) ** 2)))
        a.set_title(f"{meta[i, 1]:.2f}N {meta[i, 2]:.2f}E  RMSE {rmse:.2f} degC", fontsize=9)
        a.set_xlabel("temperature (degC)")
        a.grid(alpha=0.3)
    for a in axes[len(idx):]:
        a.axis("off")
    axes[0].invert_yaxis()
    axes[0].set_yscale("symlog", linthresh=50)
    axes[0].set_ylim(1100, -1)
    axes[0].set_ylabel("depth (m)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Reconstructed vs reference profiles (holdout never seen in training)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_spatial(y_true, y_pred, meta, depths, path, depth_target=100.0):
    k = int(np.argmin(np.abs(np.asarray(depths) - depth_target)))
    day = meta[:, 0]
    pick = day == np.unique(day)[len(np.unique(day)) // 2]
    lat, lon = meta[pick, 1], meta[pick, 2]
    t, p = y_true[pick, k], y_pred[pick, k]
    err = p - t

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True, sharey=True)
    vmin, vmax = float(np.min(t)), float(np.max(t))
    s0 = ax[0].scatter(lon, lat, c=t, s=10, cmap="turbo", vmin=vmin, vmax=vmax)
    s1 = ax[1].scatter(lon, lat, c=p, s=10, cmap="turbo", vmin=vmin, vmax=vmax)
    lim = float(np.percentile(np.abs(err), 99)) or 1.0
    s2 = ax[2].scatter(lon, lat, c=err, s=10, cmap="coolwarm", vmin=-lim, vmax=lim)
    for a, s, t_ in zip(ax, [s0, s1, s2],
                        [f"reference T at {depths[k]:.0f} m", "OceanEmbed prediction", "error (pred - ref)"]):
        a.set_title(t_, fontsize=10)
        a.set_xlabel("longitude")
        a.grid(alpha=0.25)
        fig.colorbar(s, ax=a, shrink=0.85)
    ax[0].set_ylabel("latitude")
    fig.suptitle("Spatial structure of the reconstruction on one holdout day")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    ap.add_argument("--depth-map", type=float, default=100.0)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    out_dir = resolve(cfg["paths"]["outputs"])
    dpath = resolve(cfg["paths"]["dataset"])
    if not dpath.exists():
        raise SystemExit(f"missing {dpath} -- run scripts/02_build_dataset.py first")

    d = np.load(dpath, allow_pickle=True)
    arX, arY, arM = d["arX"], d["arY"], d["arM"]
    depths = d["depths"]

    pred_ = Predictor.load(cfg, prefer=None if args.backend == "auto" else args.backend)
    print(pred_, "\ncheckpoint:", pred_.meta.get("path"))
    y_pred = pred_.predict(arX)

    card = scorecard(arY, y_pred, depths, label=f"oceanembed_{pred_.meta.get('encoder')}")
    card["backend"] = pred_.backend
    card["checkpoint"] = pred_.meta.get("path")
    card["holdout"] = {
        "n_profiles": int(arX.shape[0]),
        "day_index_range": [float(arM[:, 0].min()), float(arM[:, 0].max())],
        "lat_range": [float(arM[:, 1].min()), float(arM[:, 1].max())],
        "lon_range": [float(arM[:, 2].min()), float(arM[:, 2].max())],
        "note": "independent block -- never used for training or model selection",
    }
    spath = resolve(cfg["paths"]["scorecard"])
    spath.write_text(json.dumps(card, indent=2), encoding="utf-8")

    print(f"\nevaluated on {arX.shape[0]} holdout profiles")
    print(format_table(card))

    base = load_card(resolve(cfg["paths"]["baseline_scorecard"]))
    if base:
        print("\nOceanEmbed vs point-only RF baseline:")
        print(compare_table(card, base))
    else:
        print("\n(no baseline_scorecard.json -- run scripts/03b_baseline.py for the comparison)")

    plot_skill(card, base, depths, out_dir / "skill_vs_depth.png")
    plot_profiles(arY, y_pred, arM, depths, out_dir / "example_profiles.png",
                  rng=np.random.default_rng(cfg["train"]["seed"]))
    plot_spatial(arY, y_pred, arM, depths, out_dir / "spatial_map.png", args.depth_map)

    print(f"\nwrote {spath}")
    for p in ("skill_vs_depth.png", "example_profiles.png", "spatial_map.png"):
        print(f"wrote {out_dir / p}")


if __name__ == "__main__":
    main()
