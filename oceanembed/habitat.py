"""Derived layers built on top of the reconstructed temperature field.

Everything here is a cheap function of a profile we already predict, so it costs
nothing extra at inference time and inherits the model's own validation.

  ild                  isothermal layer depth (temperature criterion)
  ohc                  full-column ocean heat content
  front_index          horizontal temperature-gradient magnitude at any depth
  habitat_suitability  thermal habitat score against a species preference band
  barrier_layer_proxy  likelihood of barrier-layer conditions (see the caveat)

NAMING, because two different things are often both called "the mixed layer":

  ILD  isothermal layer depth -- where TEMPERATURE has fallen dt below the
       near-surface value. Computable from a temperature profile alone.
  MLD  mixed layer depth -- where DENSITY has risen by the amount that same
       temperature drop would cause. Needs salinity at depth.

In most of the world ocean the two nearly coincide and the distinction is
pedantic. In the Bay of Bengal it is the whole point: river discharge lays fresh
water over warm salty water, so density stratifies far shallower than
temperature does, and ILD - MLD (the barrier layer) can exceed 50 m.

We predict temperature only. `products.mld` is therefore an ILD under a
different name, and a true barrier layer thickness is NOT computable from what
this project currently has on disk. `barrier_layer_proxy` below returns a
likelihood from surface salinity, not a thickness in metres, and says so.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ild", "ohc", "ohc_anomaly", "front_index", "habitat_suitability",
           "barrier_layer_proxy", "SPECIES", "suitability_all"]

RHO_CP = 4.09e3 * 1025.0 / 1e7      # J m-3 K-1 -> kJ cm-2 per (K m)


# ---------------------------------------------------------------------------
# vertical structure
# ---------------------------------------------------------------------------
def ild(T, depths, dt=0.5, ref_depth=10.0):
    """Isothermal layer depth (m): where T falls `dt` below the value at
    `ref_depth`.

    This is exactly what products.mld computes; it is re-exported here under the
    name that is actually correct so the barrier-layer discussion stays honest.
    """
    from .products import mld as _mld
    return _mld(T, depths, dt=dt, ref_depth=ref_depth)


def ohc(T, depths, z_max=700.0, t_ref=0.0):
    """Ocean heat content above `z_max`, kJ cm-2.

        OHC = rho * cp * integral_0^z_max (T(z) - t_ref) dz

    TCHP is the special case z_max = D26, t_ref = 26. Reporting full-column OHC
    separately matters because TCHP is identically zero wherever the surface is
    below 26 C -- it cannot show a patch warming up until it crosses the
    threshold, whereas OHC moves continuously and can flag it earlier.
    """
    T = np.atleast_2d(np.asarray(T, dtype=np.float64))
    z = np.asarray(depths, dtype=np.float64)
    out = np.full(T.shape[0], np.nan)
    valid = np.isfinite(T).all(axis=1)
    if not valid.any():
        return out
    Tv = T[valid]
    zz = np.linspace(0.0, float(z_max), 141)
    Ti = np.empty((Tv.shape[0], zz.size))
    for k in range(Tv.shape[0]):
        Ti[k] = np.interp(zz, z, Tv[k])
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    out[valid] = RHO_CP * trap(Ti - t_ref, zz, axis=1)
    return out


def ohc_anomaly(T, depths, meta, clim, z_max=700.0):
    """OHC minus the OHC of the local climatological profile, kJ cm-2.

    Anomaly rather than absolute, for the same reason the model predicts
    anomalies: the absolute number is dominated by "this is the tropics", so a
    map of it shows geography rather than events.
    """
    return ohc(T, depths, z_max) - ohc(clim.lookup(meta), depths, z_max)


# ---------------------------------------------------------------------------
# horizontal structure
# ---------------------------------------------------------------------------
def front_index(field2d, lat, lon):
    """Magnitude of the horizontal temperature gradient, degC per 100 km.

    Fish aggregate along sharp gradients because the convergence that maintains
    a front also concentrates plankton and therefore prey. INCOIS already
    derives fronts from satellite SST; the point of doing it on OUR field is
    that we can do it at 75 or 100 m, where a satellite cannot see and where the
    front is often sharper and differently placed than at the surface.

    field2d is (nlat, nlon) at one depth on one day. NaNs (land) propagate.
    """
    F = np.asarray(field2d, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    dlat_km = 111.0 * np.gradient(lat)
    dlon_km = 111.0 * np.cos(np.deg2rad(lat))[:, None] * np.gradient(lon)[None, :]

    dF_dy = np.gradient(F, axis=0) / dlat_km[:, None]
    dF_dx = np.gradient(F, axis=1) / dlon_km
    return np.hypot(dF_dx, dF_dy) * 100.0


# ---------------------------------------------------------------------------
# thermal habitat
# ---------------------------------------------------------------------------
# Approximate thermal preference bands from the fisheries literature. These are
# BROAD and species behaviour varies by region, season and life stage -- they are
# here to be overridden by a domain expert, not to be trusted to a decimal.
#
#   t_opt   : (low, high) preferred temperature band, degC
#   t_tol   : (low, high) tolerated band; outside this, score is 0
#   z_pref  : depth (m) at which the temperature criterion is evaluated
#   ild_pref: (low, high) preferred isothermal layer depth, m; None to ignore
SPECIES = {
    "skipjack_tuna": {
        "label": "Skipjack tuna (Katsuwonus pelamis)",
        "t_opt": (26.0, 30.0), "t_tol": (20.0, 32.0),
        "z_pref": 50.0, "ild_pref": (20.0, 80.0),
        "note": "surface-associated; favours a shallow, well-defined thermocline",
    },
    "yellowfin_tuna": {
        "label": "Yellowfin tuna (Thunnus albacares)",
        "t_opt": (22.0, 28.0), "t_tol": (18.0, 31.0),
        "z_pref": 100.0, "ild_pref": (40.0, 120.0),
        "note": "forages down to and below the thermocline",
    },
    "indian_oil_sardine": {
        "label": "Indian oil sardine (Sardinella longiceps)",
        "t_opt": (27.0, 29.0), "t_tol": (24.0, 30.5),
        "z_pref": 20.0, "ild_pref": (10.0, 50.0),
        "note": "coastal pelagic, shallow shelf waters",
    },
    "indian_mackerel": {
        "label": "Indian mackerel (Rastrelliger kanagurta)",
        "t_opt": (27.0, 29.0), "t_tol": (23.0, 31.0),
        "z_pref": 30.0, "ild_pref": (10.0, 60.0),
        "note": "coastal pelagic, often just above the thermocline",
    },
}


def _band_score(x, opt, tol):
    """1 inside the optimum, tapering linearly to 0 at the tolerance edge."""
    x = np.asarray(x, dtype=np.float64)
    lo_o, hi_o = opt
    lo_t, hi_t = tol
    s = np.zeros_like(x)
    s = np.where((x >= lo_o) & (x <= hi_o), 1.0, s)
    left = (x >= lo_t) & (x < lo_o)
    s = np.where(left, (x - lo_t) / max(lo_o - lo_t, 1e-9), s)
    right = (x > hi_o) & (x <= hi_t)
    s = np.where(right, (hi_t - x) / max(hi_t - hi_o, 1e-9), s)
    return np.clip(s, 0.0, 1.0)


def habitat_suitability(T, depths, species, ild_m=None):
    """Thermal habitat suitability in [0, 1] for one species.

    Deliberately a transparent threshold-and-taper score, not a learned model.
    Nobody trained it on catch data, so it cannot be validated as a catch
    predictor, and a score is an advisory about water properties -- not a claim
    that fish are there.

    T is (N, nz), ild_m is (N,) or None to skip the layer-depth term.
    """
    sp = SPECIES[species] if isinstance(species, str) else species
    T = np.atleast_2d(np.asarray(T, dtype=np.float64))
    z = np.asarray(depths, dtype=np.float64)

    t_at = np.array([np.interp(sp["z_pref"], z, row) for row in T])
    score = _band_score(t_at, sp["t_opt"], sp["t_tol"])

    if ild_m is not None and sp.get("ild_pref"):
        lo, hi = sp["ild_pref"]
        pad = 0.5 * (hi - lo)
        score = score * _band_score(np.asarray(ild_m, dtype=np.float64),
                                    (lo, hi), (lo - pad, hi + pad))
    return score


def suitability_all(T, depths, ild_m=None):
    """{species: score array} for everything in SPECIES."""
    return {k: habitat_suitability(T, depths, k, ild_m) for k in SPECIES}


# ---------------------------------------------------------------------------
# barrier layer -- the honest version
# ---------------------------------------------------------------------------
def barrier_layer_proxy(sss, ild_m, sss_ref=34.0, ild_ref=30.0):
    """Likelihood in [0, 1] that barrier-layer conditions are present.

    NOT a thickness. A barrier layer is ILD minus density-MLD, and density-MLD
    needs SALINITY AT DEPTH, which this project does not have -- we predict
    temperature only. Returning a number in metres here would be fabricating a
    quantity we cannot compute.

    What we can say is where the two ingredients co-occur: fresh surface water
    (the Bay of Bengal river plumes) sitting over a deep isothermal layer. That
    is the recognised signature, and both terms are things we actually hold --
    SSS is a model INPUT, ILD is a model OUTPUT.

    To compute a real thickness, add salinity as a second prediction head. That
    needs GLORYS `so` alongside `thetao` in the download, which is a data task,
    not a modelling one.
    """
    fresh = _band_score(np.asarray(sss, dtype=np.float64),
                        (0.0, sss_ref - 1.0), (0.0, sss_ref))
    deep = np.clip((np.asarray(ild_m, dtype=np.float64) - ild_ref) / 40.0, 0.0, 1.0)
    return fresh * deep
