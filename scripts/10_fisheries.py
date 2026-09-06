"""Stage 10 -- habitat, fronts, barrier-layer conditions and full-column heat.

    python scripts/10_fisheries.py [--days 12] [--front-depth 100]

Four layers, all cheap functions of the profile we already predict:

  thermal habitat   suitability for a species' preferred temperature band and
                    isothermal layer depth -- the same category of product as
                    INCOIS's Potential Fishing Zone advisory, which today uses
                    SST and chlorophyll and has no subsurface term at all
  fronts            horizontal temperature-gradient magnitude. Computed at
                    depth, where a satellite cannot see it
  barrier layer     likelihood of the fresh-over-warm structure that lets Bay of
                    Bengal cyclones intensify without the usual self-cooling
  OHC               full-column heat content anomaly, which moves continuously
                    where TCHP is pinned at zero until the 26 C threshold trips

Every layer is ALSO computed from the GLORYS reference on the same days and the
two are compared, so each one carries its own error bar rather than being
asserted. Nothing here is trained on catch data; a habitat score describes water
properties, not the presence of fish.
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
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.habitat import (SPECIES, barrier_layer_proxy, front_index,
                                habitat_suitability, ild, ohc)
from oceanembed.inference import Predictor
from oceanembed.products import tchp


def predict_day(pred_, ds, it, cfg, svars, P, land, lat, lon):
    """Full (nz, ny, nx) reconstruction for one day index."""
    ny, nx = land.shape
    nz = ds.sizes["depth"]
    half = P // 2
    surf = np.stack([np.asarray(ds[v].isel(time=it).values, np.float32) for v in svars])
    win = sliding_window_view(surf, (P, P), axis=(1, 2))
    ok = np.isfinite(win).all(axis=(0, 3, 4)) & (~land[half:ny - half, half:nx - half])
    iy, ix = np.nonzero(ok)
    out = np.full((nz, ny, nx), np.nan, np.float32)
    if not iy.size:
        return out
    patches = np.moveaxis(win[:, iy, ix, :, :], 0, 1).astype(np.float32)
    meta = np.stack([np.full(iy.size, it, np.float64),
                     lat[iy + half], lon[ix + half]], axis=1)
    out[:, iy + half, ix + half] = pred_.predict(patches, meta=meta).T
    return out


def layers(field, depths, sss, lat, lon, front_depth):
    """All derived layers for one (nz, ny, nx) temperature field."""
    nz, ny, nx = field.shape
    flat = field.reshape(nz, -1).T
    good = np.isfinite(flat).all(axis=1)
    out = {k: np.full(ny * nx, np.nan) for k in
           ("ild", "ohc", "tchp", "barrier")}
    for k in SPECIES:
        out[k] = np.full(ny * nx, np.nan)
    if good.any():
        g = flat[good]
        i_ld = ild(g, depths)
        out["ild"][good] = i_ld
        out["ohc"][good] = ohc(g, depths, z_max=700.0)
        out["tchp"][good] = tchp(g, depths)
        out["barrier"][good] = barrier_layer_proxy(
            np.asarray(sss, np.float64).reshape(-1)[good], i_ld)
        for k in SPECIES:
            out[k][good] = habitat_suitability(g, depths, k, i_ld)
    out = {k: v.reshape(ny, nx) for k, v in out.items()}

    kf = int(np.argmin(np.abs(np.asarray(depths) - front_depth)))
    out["front_surface"] = front_index(field[0], lat, lon)
    out["front_deep"] = front_index(field[kf], lat, lon)
    out["front_depth_m"] = float(depths[kf])
    return out


def agree(name, unit, ref, prd, rows):
    m = np.isfinite(ref) & np.isfinite(prd)
    if m.sum() < 10:
        return
    r, p = ref[m], prd[m]
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    bias = float(np.mean(p - r))
    corr = float(np.corrcoef(r, p)[0, 1]) if r.std() > 1e-9 and p.std() > 1e-9 else float("nan")
    print(f"  {name:22s} n={m.sum():7d}  corr {corr:6.3f}  RMSE {rmse:8.3f} {unit:9s} "
          f"bias {bias:+8.3f}")
    rows.append({"layer": name, "unit": unit, "n": int(m.sum()),
                 "corr": corr, "rmse": rmse, "bias": bias,
                 "ref_mean": float(r.mean())})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=12,
                    help="how many days to sample, spread across the record")
    ap.add_argument("--front-depth", type=float, default=100.0)
    ap.add_argument("--species", default="skipjack_tuna,indian_oil_sardine")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    out_dir = resolve(cfg["paths"]["outputs"])
    hpath = resolve(cfg["paths"]["harmonized"])
    if not hpath.exists():
        raise SystemExit(f"missing {hpath}")

    ds = xr.open_dataset(hpath)
    svars = list(cfg["surface_vars"])
    P = int(cfg["patch"]["size"])
    depths = np.asarray(ds["depth"].values, dtype=np.float64)
    lat = np.asarray(ds["lat"].values, dtype=np.float32)
    lon = np.asarray(ds["lon"].values, dtype=np.float32)
    land = np.asarray(ds["land_mask"].values) > 0.5
    times = np.asarray(ds["time"].values, dtype="datetime64[D]")

    pred_ = Predictor.load(cfg)
    print(pred_)

    # sample days spread across the whole record, so the seasonal panel has
    # something to show
    idx = np.unique(np.linspace(0, len(times) - 1, args.days).astype(int))
    print(f"\ncomputing derived layers on {idx.size} days spread over "
          f"{times[0]} .. {times[-1]}\n")

    keep, rows_pred, rows_ref = [], [], []
    for it in idx:
        sss = np.asarray(ds["sss"].isel(time=int(it)).values, np.float32)
        pf = predict_day(pred_, ds, int(it), cfg, svars, P, land, lat, lon)
        rf = np.asarray(ds["temp"].isel(time=int(it)).values, np.float32)
        Lp = layers(pf, depths, sss, lat, lon, args.front_depth)
        Lr = layers(rf, depths, sss, lat, lon, args.front_depth)
        keep.append({"it": int(it), "date": str(times[it]), "pred": Lp, "ref": Lr})
        print(f"  {times[it]}  done")

    # ---- validation: every derived layer against the same layer from GLORYS
    print("\nDerived layers, predicted vs the identical computation on GLORYS:\n")
    rows = []
    cat = lambda side, k: np.concatenate([d[side][k].ravel() for d in keep])
    agree("isothermal layer depth", "m", cat("ref", "ild"), cat("pred", "ild"), rows)
    agree("OHC (0-700 m)", "kJ/cm2", cat("ref", "ohc"), cat("pred", "ohc"), rows)
    agree("TCHP", "kJ/cm2", cat("ref", "tchp"), cat("pred", "tchp"), rows)
    fd = keep[0]["pred"]["front_depth_m"]
    agree("front index, surface", "C/100km",
          cat("ref", "front_surface"), cat("pred", "front_surface"), rows)
    agree(f"front index, {fd:.0f} m", "C/100km",
          cat("ref", "front_deep"), cat("pred", "front_deep"), rows)
    agree("barrier-layer proxy", "0-1", cat("ref", "barrier"), cat("pred", "barrier"), rows)
    for k in SPECIES:
        agree(f"habitat: {k}", "0-1", cat("ref", k), cat("pred", k), rows)

    # ---- how often would the advisory agree on the actual call?
    print("\n  Agreement on the call a user would actually act on:")
    hit_rows = []
    for k in SPECIES:
        r, p = cat("ref", k), cat("pred", k)
        m = np.isfinite(r) & np.isfinite(p)
        R, Q = r[m] >= 0.6, p[m] >= 0.6
        tp = int((R & Q).sum()); fp = int((~R & Q).sum()); fn = int((R & ~Q).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"    {k:22s} suitable-zone precision {prec:.3f}  recall {rec:.3f}  "
              f"F1 {f1:.3f}  (base rate {R.mean():.1%})")
        hit_rows.append({"species": k, "precision": prec, "recall": rec,
                         "f1": f1, "base_rate": float(R.mean())})

    # ---- figures ---------------------------------------------------------
    ext = [lon.min(), lon.max(), lat.min(), lat.max()]
    mid = keep[len(keep) // 2]
    sp_list = [s for s in args.species.split(",") if s in SPECIES]

    # 1. habitat, predicted vs reference
    fig, ax = plt.subplots(len(sp_list), 2, figsize=(12, 4.0 * len(sp_list)),
                           squeeze=False)
    for i, s in enumerate(sp_list):
        for j, side in enumerate(("pred", "ref")):
            im = ax[i, j].imshow(mid[side][s], origin="lower", extent=ext,
                                 cmap="YlGn", vmin=0, vmax=1)
            ax[i, j].set_title(f"{SPECIES[s]['label']}\n"
                               f"{'OceanEmbed' if j == 0 else 'GLORYS'}  {mid['date']}",
                               size=9)
            ax[i, j].set_xlabel("longitude"); ax[i, j].set_ylabel("latitude")
            fig.colorbar(im, ax=ax[i, j], shrink=0.85, label="suitability")
    fig.suptitle("Thermal habitat suitability -- advisory on water properties, "
                 "not a claim about fish", size=11)
    fig.tight_layout()
    fig.savefig(out_dir / "habitat_map.png", dpi=130)
    plt.close(fig)

    # 2. fronts: surface vs depth, side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, k, t in ((ax[0], "front_surface", "surface (a satellite sees this)"),
                    (ax[1], "front_deep", f"{fd:.0f} m (it cannot)")):
        v = mid["pred"][k]
        im = a.imshow(v, origin="lower", extent=ext, cmap="magma",
                      vmin=0, vmax=float(np.nanpercentile(v, 98)))
        a.set_title(f"temperature front index, {t}", size=9.5)
        a.set_xlabel("longitude"); a.set_ylabel("latitude")
        fig.colorbar(im, ax=a, shrink=0.85, label="degC / 100 km")
    fig.suptitle(f"Thermal fronts on {mid['date']} -- fish aggregate along sharp "
                 "gradients", size=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fronts_map.png", dpi=130)
    plt.close(fig)

    # 3. barrier-layer likelihood and ILD
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    im0 = ax[0].imshow(mid["pred"]["ild"], origin="lower", extent=ext,
                       cmap="viridis_r", vmin=0, vmax=100)
    ax[0].set_title("isothermal layer depth (m)", size=9.5)
    fig.colorbar(im0, ax=ax[0], shrink=0.85)
    im1 = ax[1].imshow(np.asarray(ds["sss"].isel(time=mid["it"]).values),
                       origin="lower", extent=ext, cmap="YlGnBu_r", vmin=30, vmax=36)
    ax[1].set_title("sea surface salinity (input)", size=9.5)
    fig.colorbar(im1, ax=ax[1], shrink=0.85)
    im2 = ax[2].imshow(mid["pred"]["barrier"], origin="lower", extent=ext,
                       cmap="OrRd", vmin=0, vmax=1)
    ax[2].set_title("barrier-layer likelihood (0-1)", size=9.5)
    fig.colorbar(im2, ax=ax[2], shrink=0.85)
    for a in ax:
        a.set_xlabel("longitude")
    ax[0].set_ylabel("latitude")
    fig.suptitle(f"Barrier-layer conditions, {mid['date']} -- fresh water over a "
                 "deep isothermal layer (Bay of Bengal signature)", size=11)
    fig.tight_layout()
    fig.savefig(out_dir / "barrier_layer.png", dpi=130)
    plt.close(fig)

    # 4. seasonal shift of the suitable zone
    s0 = sp_list[0]
    n = min(len(keep), 12)
    pick = np.linspace(0, len(keep) - 1, n).astype(int)
    cols = 4
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(3.3 * cols, 2.9 * rws),
                             squeeze=False)
    axes = axes.ravel()
    track = []
    for a, j in zip(axes, pick):
        d = keep[j]
        v = d["pred"][s0]
        a.imshow(v, origin="lower", extent=ext, cmap="YlGn", vmin=0, vmax=1)
        m = np.isfinite(v) & (v >= 0.6)
        if m.any():
            yy, xx = np.nonzero(m)
            cy, cx = float(lat[yy].mean()), float(lon[xx].mean())
            a.plot(cx, cy, "o", ms=7, mfc="none", mec="#d1701a", mew=2)
            track.append({"date": d["date"], "centroid_lat": cy,
                          "centroid_lon": cx, "suitable_fraction": float(m.mean())})
        a.set_title(d["date"], size=8)
        a.set_xticks([]); a.set_yticks([])
    for a in axes[len(pick):]:
        a.axis("off")
    fig.suptitle(f"Seasonal shift of the suitable zone -- {SPECIES[s0]['label']}\n"
                 "circle marks the centroid of cells scoring above 0.6", size=10)
    fig.tight_layout()
    fig.savefig(out_dir / "seasonal_shift.png", dpi=130)
    plt.close(fig)

    payload = {
        "n_days": int(idx.size),
        "dates": [d["date"] for d in keep],
        "front_depth_m": fd,
        "layers_vs_glorys": rows,
        "suitable_zone_agreement": hit_rows,
        "seasonal_centroid_track": track,
        "species": {k: {kk: vv for kk, vv in v.items()} for k, v in SPECIES.items()},
        "caveats": [
            "Habitat scores are threshold-and-taper functions of temperature and "
            "isothermal layer depth. They are not trained on catch data and are "
            "not validated as catch predictors.",
            "The barrier-layer field is a LIKELIHOOD, not a thickness. A true "
            "barrier layer is isothermal layer depth minus density mixed layer "
            "depth, and density needs salinity at depth, which this project does "
            "not predict.",
            "Species preference bands are broad literature values and should be "
            "replaced with regionally tuned ones by a fisheries scientist.",
        ],
    }
    out = out_dir / "fisheries_scorecard.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    for f in ("habitat_map.png", "fronts_map.png", "barrier_layer.png",
              "seasonal_shift.png"):
        print(f"wrote {out_dir / f}")
    ds.close()


if __name__ == "__main__":
    main()
