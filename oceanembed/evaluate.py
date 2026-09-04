"""Metrics -- per-depth correlation, RMSE and bias (Contract E).

The whole credibility story lives here: skill should be high in the mixed layer
and decay below the thermocline, where the surface stops constraining T. We
report that honestly rather than quoting one basin-wide number.
"""
from __future__ import annotations

import numpy as np

__all__ = ["per_depth_metrics", "scorecard", "format_table", "compare_table", "load_card"]


def per_depth_metrics(y_true: np.ndarray, y_pred: np.ndarray, depths) -> list[dict]:
    """corr / RMSE / bias / MAE at each depth, ignoring NaNs."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float32)
    rows = []
    for k, z in enumerate(depths):
        t, p = y_true[:, k], y_pred[:, k]
        m = np.isfinite(t) & np.isfinite(p)
        t, p = t[m], p[m]
        if t.size < 3:
            rows.append({"depth_m": float(z), "n": int(t.size), "corr": float("nan"),
                         "rmse": float("nan"), "bias": float("nan"), "mae": float("nan"),
                         "std_obs": float("nan"), "skill_score": float("nan")})
            continue
        err = p - t
        rmse = float(np.sqrt(np.mean(err ** 2)))
        std_obs = float(np.std(t))
        # Murphy skill score against the climatological (mean) forecast
        denom = float(np.mean((t - t.mean()) ** 2))
        skill = float(1.0 - np.mean(err ** 2) / denom) if denom > 1e-12 else float("nan")
        corr = float(np.corrcoef(t, p)[0, 1]) if np.std(t) > 1e-9 and np.std(p) > 1e-9 else float("nan")
        rows.append({
            "depth_m": float(z), "n": int(t.size), "corr": corr, "rmse": rmse,
            "bias": float(np.mean(err)), "mae": float(np.mean(np.abs(err))),
            "std_obs": std_obs, "skill_score": skill,
        })
    return rows


def scorecard(y_true, y_pred, depths, label: str = "oceanembed") -> dict:
    """Contract E payload: per-depth rows plus headline means."""
    rows = per_depth_metrics(y_true, y_pred, depths)
    finite = [r for r in rows if np.isfinite(r["corr"])]
    z = np.asarray([r["depth_m"] for r in rows])
    rmse = np.asarray([r["rmse"] for r in rows])

    def band(lo, hi):
        m = (z >= lo) & (z <= hi)
        sel = [r for r, ok in zip(rows, m) if ok and np.isfinite(r["corr"])]
        if not sel:
            return None
        return {
            "mean_corr": float(np.mean([r["corr"] for r in sel])),
            "mean_rmse": float(np.mean([r["rmse"] for r in sel])),
        }

    return {
        "label": label,
        "n_profiles": int(np.asarray(y_true).shape[0]),
        "depths_m": [float(v) for v in z],
        "per_depth": rows,
        "mean_corr": float(np.mean([r["corr"] for r in finite])) if finite else float("nan"),
        "mean_rmse": float(np.mean(rmse[np.isfinite(rmse)])),
        "mean_bias": float(np.mean([r["bias"] for r in finite])) if finite else float("nan"),
        "mean_skill_score": float(np.mean([r["skill_score"] for r in finite])) if finite else float("nan"),
        "bands": {
            "mixed_layer_0_50m": band(0, 50),
            "thermocline_75_300m": band(75, 300),
            "deep_500_1000m": band(500, 1000),
        },
    }


def format_table(card: dict) -> str:
    """Human-readable per-depth table for the terminal and the report."""
    lines = [
        f"{'depth (m)':>10} {'n':>7} {'corr':>7} {'RMSE':>8} {'bias':>8} {'MAE':>8} {'skill':>7}",
        "-" * 60,
    ]
    for r in card["per_depth"]:
        lines.append(
            f"{r['depth_m']:10.0f} {r['n']:7d} {r['corr']:7.3f} {r['rmse']:8.3f} "
            f"{r['bias']:8.3f} {r['mae']:8.3f} {r['skill_score']:7.3f}"
        )
    lines.append("-" * 60)
    lines.append(
        f"{'MEAN':>10} {card['n_profiles']:7d} {card['mean_corr']:7.3f} "
        f"{card['mean_rmse']:8.3f} {card['mean_bias']:8.3f} {'':>8} {card['mean_skill_score']:7.3f}"
    )
    for name, b in card.get("bands", {}).items():
        if b:
            lines.append(f"  {name:22s} corr {b['mean_corr']:6.3f}   RMSE {b['mean_rmse']:6.3f} degC")
    return "\n".join(lines)


def compare_table(card: dict, baseline: dict) -> str:
    """Side-by-side model vs baseline, per depth."""
    lines = [
        f"{'depth (m)':>10} {'RMSE model':>11} {'RMSE base':>10} {'improve':>9} "
        f"{'corr model':>11} {'corr base':>10}",
        "-" * 68,
    ]
    for r, b in zip(card["per_depth"], baseline["per_depth"]):
        imp = 100.0 * (b["rmse"] - r["rmse"]) / b["rmse"] if b["rmse"] > 1e-9 else float("nan")
        lines.append(
            f"{r['depth_m']:10.0f} {r['rmse']:11.3f} {b['rmse']:10.3f} {imp:8.1f}% "
            f"{r['corr']:11.3f} {b['corr']:10.3f}"
        )
    imp = 100.0 * (baseline["mean_rmse"] - card["mean_rmse"]) / baseline["mean_rmse"]
    lines.append("-" * 68)
    lines.append(
        f"{'MEAN':>10} {card['mean_rmse']:11.3f} {baseline['mean_rmse']:10.3f} {imp:8.1f}% "
        f"{card['mean_corr']:11.3f} {baseline['mean_corr']:10.3f}"
    )
    return "\n".join(lines)


def load_card(path) -> dict | None:
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
