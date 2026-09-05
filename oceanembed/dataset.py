"""Dataset construction -- harmonized.nc (Contract B) -> dataset.npz (Contract C).

For every selected ocean cell we cut an N x N patch out of the 7 surface fields
(that patch is what lets the encoder see eddies and fronts) and optionally append
four constant channels: normalised lat, normalised lon, sin(doy), cos(doy).

    X : (N, C, P, P)   C = 7 surface  (+ 4 coord channels if patch.add_coords)
    Y : (N, 15)        temperature at the standard depths
    M : (N, 3)         meta -- time index, lat, lon

Splits are chronological: earliest days train, then val, and the LAST block of
days is the independent ARGO-style holdout that training never sees.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from . import load_config
from .climatology import build_from_samples

__all__ = [
    "channel_names",
    "coord_vars",
    "coord_planes",
    "extract_samples",
    "fit_scalers",
    "apply_scalers",
    "invert_target",
    "split_train_val_argo",
    "build",
    "ProfileDataset",
]


ALL_COORDS = ["lat_n", "lon_n", "sin_doy", "cos_doy"]


def coord_vars(cfg: dict | None = None) -> list[str]:
    """Which coordinate channels to append, in order.

    Split out from a single on/off flag because the two kinds do very different
    things. lat_n/lon_n let the model memorise a spatial climatology, which
    inflates scores without any surface inference behind them. sin_doy/cos_doy
    tell it what time of year it is, which is real physical context once the
    record spans more than one season -- without them the model cannot tell
    January from July.
    """
    cfg = cfg or load_config()
    if not cfg["patch"].get("add_coords", True):
        return []
    return [c for c in cfg["patch"].get("coord_vars", ALL_COORDS) if c in ALL_COORDS]


def coord_planes(cfg, names, lat_n, lon_n, doy, P):
    """(n, k, P, P) constant planes for the requested coordinate channels.

    lat_n/lon_n are per-sample arrays already normalised to [-1, 1]; doy is the
    day of year for the sample's day.
    """
    n = len(lat_n)
    out = np.empty((n, len(names), P, P), dtype=np.float32)
    for j, nm in enumerate(names):
        if nm == "lat_n":
            out[:, j] = np.asarray(lat_n, np.float32)[:, None, None]
        elif nm == "lon_n":
            out[:, j] = np.asarray(lon_n, np.float32)[:, None, None]
        elif nm == "sin_doy":
            out[:, j] = np.float32(np.sin(2 * np.pi * doy / 365.25))
        elif nm == "cos_doy":
            out[:, j] = np.float32(np.cos(2 * np.pi * doy / 365.25))
    return out


def channel_names(cfg: dict | None = None) -> list[str]:
    """Ordered channel names of X. Keep this in sync with extract_samples."""
    cfg = cfg or load_config()
    return list(cfg["surface_vars"]) + coord_vars(cfg)


def extract_samples(
    ds: xr.Dataset,
    cfg: dict | None = None,
    time_indices=None,
    max_per_day: int | None = None,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
):
    """Cut P x P patches around valid ocean cells.

    Returns X (N,C,P,P) float32, Y (N,15) float32, M (N,3) float64
    where M columns are [time_index, lat, lon].
    """
    cfg = cfg or load_config()
    svars = list(cfg["surface_vars"])
    P = int(cfg["patch"]["size"])
    cnames = coord_vars(cfg)
    half = P // 2
    if max_per_day is None:
        max_per_day = int(cfg["dataset"]["max_samples_per_day"])
    if rng is None:
        rng = np.random.default_rng(cfg["dataset"]["seed"])

    lat = np.asarray(ds["lat"].values, dtype=np.float32)
    lon = np.asarray(ds["lon"].values, dtype=np.float32)
    depths = np.asarray(ds["depth"].values, dtype=np.float32)
    ny, nx, nz = lat.size, lon.size, depths.size
    land = np.asarray(ds["land_mask"].values) > 0.5

    dom = cfg["domain"]
    lat_n = (lat - dom["lat_min"]) / max(dom["lat_max"] - dom["lat_min"], 1e-6) * 2.0 - 1.0
    lon_n = (lon - dom["lon_min"]) / max(dom["lon_max"] - dom["lon_min"], 1e-6) * 2.0 - 1.0

    times = np.asarray(ds["time"].values)
    if time_indices is None:
        time_indices = np.arange(times.size)
    time_indices = np.asarray(time_indices)
    doy = np.asarray(
        [float(np.datetime64(t, "D").astype("datetime64[D]").astype(object).timetuple().tm_yday) for t in times]
    )

    n_ch = len(svars) + len(cnames)

    Xs, Ys, Ms = [], [], []
    kept, seen = 0, 0
    for it in time_indices:
        it = int(it)
        surf = np.stack([np.asarray(ds[v].isel(time=it).values, dtype=np.float32) for v in svars])
        tgt = np.asarray(ds["temp"].isel(time=it).values, dtype=np.float32)   # (nz,ny,nx)

        # patch windows over the interior only -- no padding, no wrap-around
        win = sliding_window_view(surf, (P, P), axis=(1, 2))                  # (C0,ny-P+1,nx-P+1,P,P)
        good_patch = np.isfinite(win).all(axis=(0, 3, 4))                     # (ny-P+1,nx-P+1)
        centre = (slice(half, ny - half), slice(half, nx - half))
        good_centre = np.isfinite(tgt[:, centre[0], centre[1]]).all(axis=0) & (~land[centre])
        ok = good_patch & good_centre
        seen += int(good_centre.size)

        iy, ix = np.nonzero(ok)
        if iy.size == 0:
            continue
        if iy.size > max_per_day:
            pick = rng.choice(iy.size, size=max_per_day, replace=False)
            iy, ix = iy[pick], ix[pick]

        patches = win[:, iy, ix, :, :]                                        # (C0,n,P,P)
        patches = np.moveaxis(patches, 0, 1).astype(np.float32)               # (n,C0,P,P)
        n = patches.shape[0]

        if cnames:
            extra = coord_planes(cfg, cnames, lat_n[iy + half], lon_n[ix + half],
                                 doy[it], P)
            patches = np.concatenate([patches, extra], axis=1)

        y = tgt[:, iy + half, ix + half].T.astype(np.float32)                 # (n,nz)
        m = np.stack([np.full(n, it, dtype=np.float64), lat[iy + half], lon[ix + half]], axis=1)

        Xs.append(patches)
        Ys.append(y)
        Ms.append(m)
        kept += n

    if not Xs:
        raise RuntimeError("no valid samples extracted -- check land_mask / NaNs")

    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    M = np.concatenate(Ms)
    assert X.shape[1] == n_ch and X.shape[2] == P and Y.shape[1] == nz
    if verbose:
        print(f"[dataset] {len(time_indices)} days -> {kept} samples, X{X.shape} Y{Y.shape}")
    return X, Y, M


def fit_scalers(X: np.ndarray, Y: np.ndarray):
    """Per-channel input scalers and per-depth target scalers. TRAIN SPLIT ONLY."""
    xm = X.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    xs = X.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    xs[xs < 1e-6] = 1.0
    ym = Y.mean(axis=0).astype(np.float32)
    ys = Y.std(axis=0).astype(np.float32)
    ys[ys < 1e-6] = 1.0
    return xm, xs, ym, ys


def apply_scalers(X, Y, xm, xs, ym, ys):
    Xn = ((X - xm) / xs).astype(np.float32)
    Yn = ((Y - ym) / ys).astype(np.float32) if Y is not None else None
    return Xn, Yn


def invert_target(Yn: np.ndarray, ym: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Standardised prediction -> degrees Celsius."""
    return (np.asarray(Yn) * np.asarray(ys).reshape(1, -1) + np.asarray(ym).reshape(1, -1)).astype(np.float32)


def split_train_val_argo(n_times: int, cfg: dict | None = None):
    """Chronological day split -> (train_days, val_days, argo_days).

    The ARGO block is the LAST stretch of days, so the holdout is independent in
    time as well as in space. It is never used for training or model selection.
    """
    cfg = cfg or load_config()
    val_frac = float(cfg["train"]["val_fraction"])
    argo_frac = float(cfg["dataset"]["argo_fraction"])
    days = np.arange(n_times)
    n_argo = max(1, int(round(argo_frac * n_times)))
    argo = days[n_times - n_argo:]
    rest = days[: n_times - n_argo]
    n_val = max(1, int(round(val_frac * rest.size)))
    val = rest[rest.size - n_val:]
    train = rest[: rest.size - n_val]
    return train, val, argo


def build(ds: xr.Dataset, cfg: dict | None = None, verbose: bool = True) -> dict:
    """Full Contract C payload from a harmonized cube."""
    cfg = cfg or load_config()
    rng = np.random.default_rng(cfg["dataset"]["seed"])
    n_times = int(ds.sizes["time"])
    tr_d, va_d, ar_d = split_train_val_argo(n_times, cfg)
    if verbose:
        print(f"[dataset] chronological split -> train {tr_d.size} d | val {va_d.size} d | argo {ar_d.size} d")

    trX, trY, trM = extract_samples(ds, cfg, tr_d, rng=rng, verbose=verbose)
    vaX, vaY, vaM = extract_samples(ds, cfg, va_d, rng=rng, verbose=verbose)
    arX, arY, arM = extract_samples(ds, cfg, ar_d, rng=rng, verbose=verbose)

    xm, xs, ym, ys = fit_scalers(trX, trY)   # TRAIN STATS ONLY
    if verbose:
        print("[dataset] scalers fitted on the train split only")

    # Climatology from the TRAIN samples only, carried with the dataset so the
    # trainer and the predictor use the identical one. Building it from the whole
    # record would leak the holdout into the anomaly target.
    clim = build_from_samples(trY, trM)
    if verbose:
        print(f"[dataset] climatology over {clim.keys.shape[0]} locations "
              f"(train days only)")

    return {
        "trX": trX, "trY": trY, "trM": trM,
        "vaX": vaX, "vaY": vaY, "vaM": vaM,
        "arX": arX, "arY": arY, "arM": arM,
        "xm": xm, "xs": xs, "ym": ym, "ys": ys,
        "depths": np.asarray(ds["depth"].values, dtype=np.float32),
        "channels": np.array(channel_names(cfg)),
        "train_days": tr_d, "val_days": va_d, "argo_days": ar_d,
        **clim.pack(),
    }


class ProfileDataset:
    """torch Dataset over standardised (X, Y). Falls back to plain indexing.

    Imported lazily so the module works with no torch installed.
    """

    def __init__(self, X, Y, xm=None, xs=None, ym=None, ys=None):
        if xm is not None:
            X, Y = apply_scalers(X, Y, xm, xs, ym, ys)
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.Y = np.ascontiguousarray(Y, dtype=np.float32) if Y is not None else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        try:
            import torch
            x = torch.from_numpy(self.X[i])
            y = torch.from_numpy(self.Y[i]) if self.Y is not None else torch.zeros(0)
            return x, y
        except ImportError:
            return self.X[i], (self.Y[i] if self.Y is not None else None)
