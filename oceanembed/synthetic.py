"""Physically-plausible synthetic ocean cube.

The point of this module is that the subsurface temperature profile genuinely
depends on the surface state, so a model trained on it learns a real mapping
rather than memorising noise:

  * a mesoscale eddy field (random-phase spectral noise) is imprinted on BOTH
    the sea-level anomaly and the surface temperature/currents;
  * mixed-layer depth and thermocline scale are driven by SLA (warm anticyclonic
    eddies depress isotherms), wind stirring and the seasonal cycle;
  * T(z) is a two-layer profile: near-constant in the mixed layer, sharp decay
    through the thermocline, near-invariant deep water.

Canonical names (Contract A) are written into a raw cube on a coarser grid;
harmonize.py regrids it onto the 0.25 deg / daily target grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import load_config

__all__ = ["make_cube", "profile_from_state", "land_mask"]


def _spectral_field(ny: int, nx: int, rng: np.random.Generator, scale: float = 6.0) -> np.ndarray:
    """Smooth random field with a mesoscale-like spectrum, normalised to unit std."""
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    k[0, 0] = 1e-6
    amp = np.exp(-((k * scale * max(ny, nx) / 60.0) ** 2))
    phase = rng.uniform(0, 2 * np.pi, size=(ny, nx))
    f = np.real(np.fft.ifft2(amp * np.exp(1j * phase)))
    return ((f - f.mean()) / (f.std() + 1e-9)).astype(np.float32)


def land_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Coarse land mask for the North Indian Ocean box (1=land, 0=ocean).

    Not a coastline product -- just enough real geometry that the pipeline has to
    handle land, NaNs and patches that fall partly on land.
    """
    LA, LO = np.meshgrid(np.asarray(lat), np.asarray(lon), indexing="ij")
    land = np.zeros(LA.shape, dtype=np.float32)

    # Indian subcontinent: wedge that narrows towards the southern tip (~8N)
    frac = np.clip((LA - 8.0) / 22.0, 0.0, 1.0)          # 0 at the tip, 1 in the north
    half_width = 1.0 + 11.0 * frac
    centre = 77.5 + 1.5 * frac
    land[(LA > 8.0) & (np.abs(LO - centre) < half_width)] = 1.0

    # Africa / Arabian peninsula (western and north-western edge)
    land[LO < 47.5] = 1.0
    land[(LO < 52.5) & (LA > 12.0)] = 1.0
    land[(LO < 58.0) & (LA > 22.0)] = 1.0

    # Indochina / Malay peninsula (north-eastern corner)
    land[(LO > 97.5) & (LA > 6.0)] = 1.0
    land[(LO > 100.5) & (LA > 2.0)] = 1.0

    # Sri Lanka
    land[((LA - 7.9) ** 2 / 1.6 ** 2 + (LO - 80.7) ** 2 / 1.0 ** 2) < 1.0] = 1.0
    return land


def eddy_diagnostics(sla: np.ndarray, smooth: int = 5):
    """Non-local eddy diagnostics from a 2-D SLA field.

    Returns (sla_smooth, eke) where eke is the smoothed magnitude of the SLA
    gradient -- a proxy for eddy kinetic energy / frontal intensity.

    These are the terms that make the subsurface depend on the NEIGHBOURHOOD
    rather than on the single cell, which is precisely why the model is given an
    N x N patch and why a point-only baseline cannot match it.
    """
    from scipy.ndimage import uniform_filter

    sla = np.asarray(sla, dtype=np.float32)
    sla_s = uniform_filter(sla, size=smooth, mode="nearest")
    gy, gx = np.gradient(sla)
    eke = uniform_filter(np.sqrt(gy ** 2 + gx ** 2), size=smooth, mode="nearest")
    return sla_s.astype(np.float32), eke.astype(np.float32)


def profile_from_state(sst, sla, wind_speed, doy, depths, sla_smooth=None, eke_n=None):
    """Deterministic T(z) from the surface state.

    sst/sla/wind_speed broadcast over any shape S; returns shape S + (nz,).

    sla_smooth / eke_n are the non-local terms from eddy_diagnostics (eke_n
    standardised). If omitted the profile degrades to the pointwise
    approximation -- good enough for a quick demo sketch, but the generated cube
    always passes them so the true mapping needs spatial context.
    """
    z = np.asarray(depths, dtype=np.float32)
    sst = np.asarray(sst, dtype=np.float32)
    sla = np.asarray(sla, dtype=np.float32)
    ws = np.asarray(wind_speed, dtype=np.float32)
    sla_s = sla if sla_smooth is None else np.asarray(sla_smooth, dtype=np.float32)
    eke = np.zeros_like(sla) if eke_n is None else np.asarray(eke_n, dtype=np.float32)

    seas = np.sin(2 * np.pi * (float(doy) - 40.0) / 365.25)

    # Mixed layer: deepened by wind stirring, winter cooling and warm eddies;
    # energetic fronts (high EKE) stir it deeper still
    mld = np.clip(25.0 + 14.0 * ws / 8.0 - 12.0 * seas + 120.0 * sla + 13.0 * eke, 10.0, 140.0)
    # Thermocline scale: set by the eddy-SCALE sea level (smoothed, not pointwise)
    # and sharpened where the front is strong
    hscale = np.clip(
        130.0 + 300.0 * np.clip(sla_s + 0.15, 0.0, None) - 34.0 * eke, 70.0, 420.0
    )
    # Deep water is nearly invariant
    t_deep = 3.6 + 0.9 * np.clip(sla, -0.3, 0.3)

    sst_b, mld_b, hs_b, td_b = sst[..., None], mld[..., None], hscale[..., None], t_deep[..., None]
    below = np.clip(z - mld_b, 0.0, None)
    t = td_b + (sst_b - td_b) * np.exp(-((below / hs_b) ** 1.35))
    # slight cooling inside the mixed layer so T(0) != T(mld) exactly
    t = t - 0.004 * np.minimum(z, mld_b)
    return t.astype(np.float32)


def make_cube(cfg: dict | None = None, verbose: bool = True) -> xr.Dataset:
    """Generate the raw synthetic cube on a coarser grid than the target."""
    cfg = cfg or load_config()
    dom, syn = cfg["domain"], cfg["synthetic"]
    rng = np.random.default_rng(syn["seed"])

    res = dom["resolution"] * syn["coarse_factor"]
    lat = np.arange(dom["lat_min"], dom["lat_max"] + 1e-6, res, dtype=np.float32)
    lon = np.arange(dom["lon_min"], dom["lon_max"] + 1e-6, res, dtype=np.float32)
    time = pd.date_range(syn["start_date"], periods=syn["n_days"], freq="D")
    depths = np.asarray(cfg["depths"], dtype=np.float32)
    ny, nx, nt, nz = lat.size, lon.size, time.size, depths.size
    if verbose:
        print(f"[synthetic] raw grid {ny} lat x {nx} lon x {nt} days x {nz} depths (res {res} deg)")

    land = land_mask(lat, lon)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")

    # Eddies persist ~20 days: cross-fade between random key frames
    n_key = max(2, nt // 20 + 2)
    keys = np.stack([_spectral_field(ny, nx, rng, scale=5.0) for _ in range(n_key)])
    keys_b = np.stack([_spectral_field(ny, nx, rng, scale=9.0) for _ in range(n_key)])

    sst = np.empty((nt, ny, nx), np.float32)
    sss, sla = np.empty_like(sst), np.empty_like(sst)
    ucur, vcur = np.empty_like(sst), np.empty_like(sst)
    uwind, vwind = np.empty_like(sst), np.empty_like(sst)
    temp = np.empty((nt, nz, ny, nx), np.float32)

    doy = np.asarray(time.dayofyear, dtype=np.float32)
    eke_ref = None                      # fixed EKE normalisation, set on day 0
    f_cor = 2 * 7.292e-5 * np.sin(np.deg2rad(np.clip(LA, 2.0, None)))
    dy = res * 111e3
    dx = res * 111e3 * np.cos(np.deg2rad(LA))

    for it in range(nt):
        pos = it / 20.0
        i0 = int(np.floor(pos)) % n_key
        i1 = (i0 + 1) % n_key
        w = pos - np.floor(pos)
        eddy = (1 - w) * keys[i0] + w * keys[i1]
        eddy_b = (1 - w) * keys_b[i0] + w * keys_b[i1]
        eddy /= eddy.std() + 1e-9
        eddy_b /= eddy_b.std() + 1e-9

        d = float(doy[it])
        seas = np.sin(2 * np.pi * (d - 40.0) / 365.25)

        # sea level anomaly (m) -- the eddy field itself
        s = 0.11 * eddy + 0.03 * eddy_b
        # SST: meridional gradient + seasonal cycle + the SAME eddy imprint
        t0 = 29.6 - 0.20 * (LA - 5.0) + 1.5 * seas + 7.6 * s + 0.15 * eddy_b
        # monsoon winds reverse with the season
        uw = -6.5 * seas + 1.4 * eddy_b
        vw = 2.6 * seas + 1.1 * eddy
        ws = np.sqrt(uw ** 2 + vw ** 2)
        # western-boundary upwelling: strong winds cool the surface
        t0 = t0 - 0.09 * np.clip(ws - 6.0, 0.0, None) * (LO < 70.0)
        # geostrophic currents from the SLA gradient
        dsdy, dsdx = np.gradient(s)
        uc = np.clip(-9.81 / f_cor * dsdy / dy, -1.5, 1.5) + 0.04 * eddy_b
        vc = np.clip(9.81 / f_cor * dsdx / dx, -1.5, 1.5) + 0.04 * eddy
        # salinity: fresh Bay of Bengal, salty Arabian Sea
        sal = 35.6 - 0.045 * np.clip(LO - 78.0, 0.0, None) - 2.7 * s + 0.2 * eddy_b
        sal = sal - 0.8 * np.clip((LA - 15.0) / 15.0, 0.0, None) * (LO > 85.0)

        # non-local eddy terms: the subsurface depends on the neighbourhood
        s_smooth, eke = eddy_diagnostics(s)
        if eke_ref is None:
            eke_ref = (float(eke.mean()), float(eke.std()) + 1e-9)
        eke_n = np.clip((eke - eke_ref[0]) / eke_ref[1], -2.0, 3.0).astype(np.float32)

        prof = profile_from_state(t0, s, ws, d, depths, sla_smooth=s_smooth, eke_n=eke_n)
        prof = np.moveaxis(prof, -1, 0)                        # (nz,ny,nx)
        prof += rng.normal(0.0, 0.06, size=prof.shape).astype(np.float32)

        sst[it], sss[it], sla[it] = t0, sal, s
        ucur[it], vcur[it] = uc, vc
        uwind[it], vwind[it] = uw, vw
        temp[it] = prof

    # NaN over land, as real gridded products have
    lm3 = np.broadcast_to(land[None, :, :] == 1.0, sst.shape)
    for arr in (sst, sss, sla, ucur, vcur, uwind, vwind):
        arr[lm3] = np.nan
    temp[np.broadcast_to(land[None, None, :, :] == 1.0, temp.shape)] = np.nan

    ds = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), sst, {"units": "degC", "long_name": "sea surface temperature"}),
            "sss": (("time", "lat", "lon"), sss, {"units": "psu", "long_name": "sea surface salinity"}),
            "sla": (("time", "lat", "lon"), sla, {"units": "m", "long_name": "sea level anomaly"}),
            "ucur": (("time", "lat", "lon"), ucur, {"units": "m s-1"}),
            "vcur": (("time", "lat", "lon"), vcur, {"units": "m s-1"}),
            "uwind": (("time", "lat", "lon"), uwind, {"units": "m s-1"}),
            "vwind": (("time", "lat", "lon"), vwind, {"units": "m s-1"}),
            "temp": (("time", "depth", "lat", "lon"), temp, {"units": "degC", "long_name": "sea water temperature"}),
            "land_mask": (("lat", "lon"), land, {"description": "1=land, 0=ocean"}),
        },
        coords={"time": time, "depth": depths, "lat": lat, "lon": lon},
        attrs={
            "title": "OceanEmbed synthetic cube",
            "note": "Subsurface T is a deterministic function of the surface state plus noise.",
            "source": "oceanembed.synthetic.make_cube",
        },
    )
    return ds
