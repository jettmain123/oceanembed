"""Stage 13 -- the one figure that carries the whole result.

    python scripts/13_headline_figure.py

Two panels, because there are two claims and they rest on DIFFERENT references.
Putting them on one axis is what made the first version of this figure
misleading.

LEFT -- "we track the reanalysis"
    Our error and GLORYS's error at every depth, both measured against gridded
    ARGO: an independent, float-derived product neither model was trained on.
    The claim is relative: our curve sits close to the reanalysis we are
    distilled from.

    The reference is INCOIS argo_10dv -- a 1 degree, 10-day OBJECTIVE ANALYSIS,
    not raw float profiles. It is smoothed. Our model runs at 0.25 degrees
    daily, four times finer and ten times more frequent, so real eddies we
    resolve and the analysis has smoothed away are counted against us. That
    makes this panel a conservative test, and it is also why climatology is NOT
    drawn here: a smooth field scores well against a smooth reference by
    construction, so plotting it would invite exactly the wrong conclusion.

RIGHT -- "and we beat the do-nothing answer"
    Skill above climatology at every depth, measured against GLORYS on held-out
    days, which is where that comparison is meaningful. Positive means the
    surface data is telling us something the local average does not already
    know. It goes negative below 500 m, and the figure shows that rather than
    cropping it.

Writes outputs/headline_result.png (and .pdf for slides).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.dataset import split_train_val_argo
from oceanembed.harmonize import interp_to_depths, normalize_coords, regrid_to_target
from oceanembed.inference import Predictor
from oceanembed.loaders import open_any, pick_var

ARGO_PATH = "data/raw/argo/argo_????-??.nc"
ARGO_VAR = "TEMP"
MAX_PROFILES = 40000

OURS = "#1f6feb"
GLO = "#111827"
POS = "#1b7a4b"
NEG = "#c2453a"
INK = "#12242f"
MUTED = "#5a6b76"


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else np.nan


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    out_dir = resolve(cfg["paths"]["outputs"])
    ds = xr.open_dataset(resolve(cfg["paths"]["harmonized"]))

    argo = normalize_coords(pick_var(open_any(ARGO_PATH), "temp", ARGO_VAR))
    argo = interp_to_depths(regrid_to_target(argo, cfg), cfg)

    svars = list(cfg["surface_vars"])
    P = int(cfg["patch"]["size"]); half = P // 2
    depths = np.asarray(ds["depth"].values, dtype=np.float64)
    lat = np.asarray(ds["lat"].values, np.float32)
    lon = np.asarray(ds["lon"].values, np.float32)
    land = np.asarray(ds["land_mask"].values) > 0.5
    model_t = np.asarray(ds["time"].values, dtype="datetime64[D]")
    argo_t = np.asarray(argo["time"].values, dtype="datetime64[D]")
    nearest = np.array([int(np.argmin(np.abs(model_t - t))) for t in argo_t])

    _, _, ho = split_train_val_argo(model_t.size, cfg)
    keep = np.nonzero(np.isin(nearest, ho))[0]
    argo = argo.isel(time=keep); argo_t = argo_t[keep]; nearest = nearest[keep]

    X, A, G, M = [], [], [], []
    for ia, it in enumerate(nearest):
        prof = np.asarray(argo.isel(time=ia).values, np.float32)
        surf = np.stack([np.asarray(ds[v].isel(time=int(it)).values, np.float32)
                         for v in svars])
        glor = np.asarray(ds["temp"].isel(time=int(it)).values, np.float32)
        win = sliding_window_view(surf, (P, P), axis=(1, 2))
        c = (slice(half, lat.size - half), slice(half, lon.size - half))
        ok = (np.isfinite(win).all(axis=(0, 3, 4))
              & np.isfinite(prof[:, c[0], c[1]]).all(axis=0)
              & np.isfinite(glor[:, c[0], c[1]]).all(axis=0) & (~land[c]))
        iy, ix = np.nonzero(ok)
        if not iy.size:
            continue
        X.append(np.moveaxis(win[:, iy, ix, :, :], 0, 1).astype(np.float32))
        A.append(prof[:, iy + half, ix + half].T.astype(np.float32))
        G.append(glor[:, iy + half, ix + half].T.astype(np.float32))
        M.append(np.stack([np.full(iy.size, it, np.float64),
                           lat[iy + half], lon[ix + half]], axis=1))
        if sum(len(x) for x in X) >= MAX_PROFILES:
            break

    X = np.concatenate(X)[:MAX_PROFILES]
    A = np.concatenate(A)[:MAX_PROFILES]
    G = np.concatenate(G)[:MAX_PROFILES]
    M = np.concatenate(M)[:MAX_PROFILES]

    pred_ = Predictor.load(cfg)
    Y = pred_.predict(X, meta=M)
    print(f"{X.shape[0]} co-located profiles on {argo_t.size} held-out ARGO days")

    r_ours = np.array([rmse(Y[:, k], A[:, k]) for k in range(len(depths))])
    r_glo = np.array([rmse(G[:, k], A[:, k]) for k in range(len(depths))])
    m_ours, m_glo = float(np.nanmean(r_ours)), float(np.nanmean(r_glo))
    gap = 100 * (m_ours - m_glo) / m_glo

    # skill above climatology, measured against GLORYS on the held-out block
    card = json.loads(resolve(cfg["paths"]["scorecard"]).read_text(encoding="utf-8"))
    sk = {r["depth_m"]: r["skill_vs_clim"] for r in card["vs_climatology"]["per_depth"]}
    skill = np.array([100 * sk.get(float(z), np.nan) for z in depths])
    mean_skill = 100 * card["vs_climatology"]["mean_skill_vs_clim"]

    # ---------------------------------------------------------------- plot --
    fig = plt.figure(figsize=(13.6, 7.0), dpi=200)
    fig.patch.set_facecolor("white")

    fig.suptitle("Reconstructed from satellites alone -- and checked twice",
                 size=17.5, weight="bold", x=0.052, ha="left", y=0.962, color=INK)
    fig.text(0.052, 0.902,
             "Left: against an independent float product neither model was trained on.   "
             "Right: against the reanalysis, on days never seen in training.",
             size=10.3, color=MUTED)

    # ---- LEFT: ours vs GLORYS against gridded ARGO -------------------------
    axL = fig.add_axes([0.052, 0.205, 0.330, 0.615])
    axL.plot(r_glo, depths, "s--", color=GLO, lw=2.2, ms=5.5,
             label=f"GLORYS reanalysis   {m_glo:.2f} °C")
    axL.plot(r_ours, depths, "o-", color=OURS, lw=3.0, ms=6.5,
             label=f"OceanEmbed             {m_ours:.2f} °C")
    axL.set_yscale("symlog", linthresh=50)
    axL.set_ylim(1100, -2)
    axL.set_xlim(0, float(np.nanmax([r_ours, r_glo])) * 1.12)
    axL.set_yticks([0, 10, 30, 50, 100, 200, 300, 500, 700, 1000])
    axL.set_yticklabels(["0", "10", "30", "50", "100", "200", "300", "500", "700", "1000"])
    axL.set_xlabel("RMSE vs gridded ARGO  (°C)", size=10.3)
    axL.set_ylabel("depth (m)", size=10.3)
    axL.grid(alpha=0.25)
    axL.legend(loc="lower right", fontsize=9.6, framealpha=0.95)
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)
    axL.set_title("We track the reanalysis we learn from",
                  size=11.6, loc="left", weight="bold", color=INK, pad=9)

    # ---- RIGHT: skill above climatology, vs GLORYS -------------------------
    axR = fig.add_axes([0.455, 0.205, 0.245, 0.615])
    ypos = np.arange(len(depths))
    axR.barh(ypos, skill, color=[POS if v >= 0 else NEG for v in skill], height=0.68)
    axR.axvline(0, color="#33414d", lw=1.1)
    axR.set_yticks(ypos)
    axR.set_yticklabels([f"{z:.0f}" for z in depths], size=8.6)
    axR.invert_yaxis()
    axR.set_xlabel("skill above climatology  (%)", size=10.3)
    axR.set_ylabel("depth (m)", size=10.3)
    axR.grid(axis="x", alpha=0.25)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)
    axR.set_title("And we beat the do-nothing answer",
                  size=11.6, loc="left", weight="bold", color=INK, pad=9)
    # top-left of the panel: the deep bars run left along the bottom, so anything
    # anchored there collides with them
    axR.text(0.03, 0.97, "negative = the local average\nis already better here",
             transform=axR.transAxes, ha="left", va="top", size=8.4,
             color=NEG, style="italic")

    # ---- the numbers -------------------------------------------------------
    fig.text(0.745, 0.760, f"{gap:.0f}%", size=46, weight="bold", color=OURS)
    fig.text(0.745, 0.718, "behind a supercomputer", size=11.4, color=INK)
    fig.text(0.745, 0.688, "reanalysis, on an independent", size=11.4, color=INK)
    fig.text(0.745, 0.658, "reference.", size=11.4, color=INK)

    fig.text(0.745, 0.545, f"+{mean_skill:.0f}%", size=46, weight="bold", color=POS)
    fig.text(0.745, 0.503, "better than climatology", size=11.4, color=INK)
    fig.text(0.745, 0.473, "on average across all", size=11.4, color=INK)
    fig.text(0.745, 0.443, "15 depths.", size=11.4, color=INK)

    fig.text(0.745, 0.330, f"{card['mean_corr']:.3f}", size=30, weight="bold", color=INK)
    fig.text(0.745, 0.296, "correlation, "
             f"{card['holdout']['n_profiles']:,}", size=10.2, color=MUTED)
    fig.text(0.745, 0.266, "held-out profiles.", size=10.2, color=MUTED)

    fig.text(0.052, 0.020,
             "Reference on the left is gridded ARGO (INCOIS argo_10dv), a 1° 10-day "
             "objective analysis built from float profiles -- not the raw profiles. It "
             "is smoothed, while we run at 0.25° daily, so\nstructure that we resolve "
             "and it has smoothed away counts as our error. That makes the left panel a "
             "conservative test, and it is why climatology is not drawn there: a smooth "
             "field\nscores well against a smooth reference by construction. Skill above "
             "climatology is therefore measured on the right, against GLORYS on held-out "
             "days, where that comparison means something.",
             size=8.1, color=MUTED, style="italic", linespacing=1.6)

    for ext in ("png", "pdf"):
        p = out_dir / f"headline_result.{ext}"
        fig.savefig(p, dpi=200, facecolor="white")
        print(f"wrote {p}")
    plt.close(fig)

    (out_dir / "headline_result.json").write_text(json.dumps({
        "n_profiles": int(X.shape[0]),
        "vs_gridded_argo": {"oceanembed": m_ours, "glorys": m_glo,
                            "gap_pct": gap},
        "vs_glorys_holdout": {"mean_corr": card["mean_corr"],
                              "mean_rmse": card["mean_rmse"],
                              "mean_skill_vs_clim_pct": mean_skill},
        "per_depth": [{"depth_m": float(z), "ours_vs_argo": float(a),
                       "glorys_vs_argo": float(b), "skill_vs_clim_pct": float(c)}
                      for z, a, b, c in zip(depths, r_ours, r_glo, skill)],
        "reference_note": "Left-panel reference is INCOIS argo_10dv gridded ARGO, "
                          "a 1 degree 10-day objective analysis -- not raw float "
                          "profiles. Smoothed, so it is a conservative test for a "
                          "0.25 degree daily model and an unfair one for "
                          "climatology comparisons.",
    }, indent=2), encoding="utf-8")
    ds.close()


if __name__ == "__main__":
    main()
