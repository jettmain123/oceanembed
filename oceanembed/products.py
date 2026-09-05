"""Operational products derived from a reconstructed temperature profile.

A temperature field is a research output. What INCOIS actually issues advisories
from are the quantities below, and every one of them is a few lines of arithmetic
on the profile we already predict.

This is where the project stops being "we estimate T(z)" and becomes "we estimate
the number you act on".

  D26   depth of the 26 degC isotherm
  TCHP  tropical cyclone heat potential -- heat stored above that isotherm, the
        quantity that predicts whether a cyclone intensifies
  MLD   mixed layer depth
  subsurface temperature anomaly and marine-heatwave flags

Convenient accident of physics: in the North Indian Ocean D26 sits at 50-150 m,
so TCHP is an integral over exactly the depth range where the model has genuine
skill. Our weakness below 300 m does not touch it.

Everything is vectorised over a leading sample axis: pass (N, nz), get (N,).
"""
from __future__ import annotations

import numpy as np

__all__ = ["d26", "tchp", "mld", "isotherm_depth", "anomaly",
           "climatology", "heatwave_flag", "RHO", "CP"]

RHO = 1026.0        # kg m-3, seawater
CP = 3993.0         # J kg-1 K-1, specific heat
J_PER_M2_TO_KJ_PER_CM2 = 1e-7


def isotherm_depth(T, depths, t_iso=26.0):
    """Depth where the profile first crosses `t_iso`, linearly interpolated.

    Returns 0 where the surface is already colder than the isotherm, and the
    deepest level where the whole profile stays warmer (rare at 1000 m).
    NaN profiles give NaN.
    """
    T = np.atleast_2d(np.asarray(T, dtype=np.float64))
    z = np.asarray(depths, dtype=np.float64)
    n, nz = T.shape
    out = np.full(n, np.nan)

    valid = np.isfinite(T).all(axis=1)
    if not valid.any():
        return out
    Tv = T[valid]

    below = Tv < t_iso                                  # first level colder
    any_below = below.any(axis=1)
    first = np.where(any_below, below.argmax(axis=1), nz - 1)

    res = np.empty(Tv.shape[0])
    # never crosses: the isotherm is deeper than our deepest level
    res[~any_below] = z[-1]
    # crosses at the surface already
    surf = any_below & (first == 0)
    res[surf] = 0.0
    # normal case -- interpolate between the bracketing levels
    mid = any_below & (first > 0)
    if mid.any():
        i = first[mid]
        t_hi, t_lo = Tv[mid, i - 1], Tv[mid, i]          # warmer above, colder below
        z_hi, z_lo = z[i - 1], z[i]
        frac = (t_hi - t_iso) / np.where(np.abs(t_hi - t_lo) < 1e-9, np.nan, t_hi - t_lo)
        res[mid] = z_hi + np.clip(frac, 0, 1) * (z_lo - z_hi)
    out[valid] = res
    return out


def d26(T, depths):
    """Depth of the 26 degC isotherm (m) -- the base of the cyclone-relevant layer."""
    return isotherm_depth(T, depths, 26.0)


def tchp(T, depths, t_iso=26.0):
    """Tropical cyclone heat potential, kJ cm-2.

        TCHP = rho * cp * integral_0^D26 (T(z) - 26) dz

    Roughly 50 kJ/cm2 is the rule-of-thumb threshold above which a storm can
    intensify rapidly; 100+ is a strongly favourable ocean.

    Integrated on a fine vertical grid rather than across our 15 coarse levels,
    because the integrand goes to zero exactly at D26 and trapezoids over 50 m
    gaps would bias the result.
    """
    T = np.atleast_2d(np.asarray(T, dtype=np.float64))
    z = np.asarray(depths, dtype=np.float64)
    n = T.shape[0]
    out = np.full(n, np.nan)
    valid = np.isfinite(T).all(axis=1)
    if not valid.any():
        return out

    Tv = T[valid]
    dz = isotherm_depth(Tv, z, t_iso)
    fine = np.linspace(0.0, 1.0, 101)                   # 0..D26, per profile
    zz = dz[:, None] * fine[None, :]                    # (m, 101)
    # interp each profile onto its own fine grid
    Ti = np.empty_like(zz)
    for k in range(Tv.shape[0]):
        Ti[k] = np.interp(zz[k], z, Tv[k])
    excess = np.clip(Ti - t_iso, 0.0, None)
    integral = np.trapezoid(excess, zz, axis=1) if hasattr(np, "trapezoid") \
        else np.trapz(excess, zz, axis=1)               # degC * m
    out[valid] = RHO * CP * integral * J_PER_M2_TO_KJ_PER_CM2
    return out


def mld(T, depths, dt=0.5, ref_depth=10.0):
    """Mixed layer depth (m): where temperature falls `dt` below the near-surface value.

    The 0.5 degC threshold relative to 10 m is the common temperature criterion
    (de Boyer Montegut and others use density; temperature is what we predict).
    """
    T = np.atleast_2d(np.asarray(T, dtype=np.float64))
    z = np.asarray(depths, dtype=np.float64)
    n = T.shape[0]
    out = np.full(n, np.nan)
    valid = np.isfinite(T).all(axis=1)
    if not valid.any():
        return out
    Tv = T[valid]
    t_ref = np.array([np.interp(ref_depth, z, row) for row in Tv])
    # depth at which the profile reaches t_ref - dt
    target = t_ref - dt
    res = np.empty(Tv.shape[0])
    for k in range(Tv.shape[0]):
        below = np.nonzero(Tv[k] < target[k])[0]
        if below.size == 0:
            res[k] = z[-1]
            continue
        i = below[0]
        if i == 0:
            res[k] = 0.0
            continue
        t_hi, t_lo = Tv[k, i - 1], Tv[k, i]
        frac = (t_hi - target[k]) / (t_hi - t_lo) if abs(t_hi - t_lo) > 1e-9 else 0.0
        res[k] = z[i - 1] + np.clip(frac, 0, 1) * (z[i] - z[i - 1])
    out[valid] = res
    return out


# ---------------------------------------------------------------------------
# anomalies and marine heatwaves
# ---------------------------------------------------------------------------
def climatology(T, keys, pct=90.0):
    """Per-location mean and upper percentile of the profile, from training days.

    T    (N, nz) profiles
    keys (N, 2)  the lat/lon of each profile, rounded -- the grouping key

    Returns (lookup, mean, pctl) where lookup maps a key tuple to a row index.
    """
    T = np.asarray(T, dtype=np.float64)
    keys = np.asarray(keys)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    m = np.empty((uniq.shape[0], T.shape[1]))
    p = np.empty_like(m)
    for i in range(uniq.shape[0]):
        rows = T[inv == i]
        m[i] = rows.mean(axis=0)
        p[i] = np.percentile(rows, pct, axis=0)
    return {tuple(k): i for i, k in enumerate(map(tuple, uniq))}, m, p


def anomaly(T, keys, lookup, clim_mean):
    """Temperature minus the local climatological mean, per depth."""
    T = np.asarray(T, dtype=np.float64)
    out = np.full_like(T, np.nan)
    gm = clim_mean.mean(axis=0)
    for j, k in enumerate(map(tuple, np.asarray(keys))):
        i = lookup.get(k)
        out[j] = T[j] - (clim_mean[i] if i is not None else gm)
    return out


def heatwave_flag(T, keys, lookup, clim_pctl):
    """True where the profile exceeds the local upper percentile, per depth.

    The standard marine-heatwave definition also requires the exceedance to last
    five days. We flag the instantaneous exceedance; with only weeks of record
    the duration criterion is not yet meaningful. Say so when reporting it.
    """
    T = np.asarray(T, dtype=np.float64)
    out = np.zeros(T.shape, dtype=bool)
    gp = clim_pctl.mean(axis=0)
    for j, k in enumerate(map(tuple, np.asarray(keys))):
        i = lookup.get(k)
        ref = clim_pctl[i] if i is not None else gp
        out[j] = np.isfinite(T[j]) & (T[j] > ref)
    return out
