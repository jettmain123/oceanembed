"""Rapid-intensification risk, coastal exposure and alert objects.

This is a DECISION-SUPPORT layer, not a warning system. It surfaces where the
ocean is capable of fuelling rapid cyclone intensification, with an uncertainty
attached so an operator knows where to trust it. It says nothing about whether a
storm exists, where one will go, or what anybody should do about it.

Deliberately rule-based. A threshold on TCHP and a threshold on model spread can
be audited, argued with and tuned by a domain expert. A second neural network
trained in an afternoon could not be.

  risk = HIGH   TCHP over threshold, ANOMALOUSLY high for that place, and the
                model is confident
  risk = ELEVATED  over threshold but either unremarkable for that place or the
                model is unsure
  risk = WATCH  approaching the threshold
  risk = LOW    otherwise

50 kJ/cm2 is the widely used rule-of-thumb threshold above which the upper ocean
can sustain rapid intensification. It is a necessary condition, not a sufficient
one -- atmospheric shear, moisture and an actual storm all matter, and none of
them are in this model.

WHY THE ANOMALY TERM EXISTS. Applying the absolute threshold alone to the North
Indian Ocean in October flags roughly the entire basin, every single day: the
tropical ocean sits above 50 kJ/cm2 most of the year. An alert that always fires
is not an alert. So HIGH also requires the heat to be unusually high FOR THAT
LOCATION, which is what makes a day stand out from the basin's warm baseline.

That is a real result about this ocean, not a tuning convenience, and it is worth
saying out loud: absolute ocean-heat thresholds do not discriminate here.
"""
from __future__ import annotations

import numpy as np

__all__ = ["TCHP_HIGH", "TCHP_WATCH", "TCHP_ANOM", "COASTAL_SEGMENTS", "risk_index",
           "coastal_exposure", "build_alerts", "DISCLAIMER"]

TCHP_HIGH = 50.0        # kJ/cm2 -- rule-of-thumb RI threshold
TCHP_WATCH = 35.0       # below the threshold but worth watching
CONF_GOOD = 0.5         # degC of MC spread at 100 m; above this we downgrade
TCHP_ANOM = 8.0         # kJ/cm2 above the local norm before a cell is notable

DISCLAIMER = (
    "Decision support only. This shows where the OCEAN could sustain rapid "
    "intensification, based on reconstructed subsurface temperature. It does not "
    "detect or forecast cyclones, does not account for atmospheric conditions, "
    "and is not an evacuation advisory. Proof of concept, not an operational "
    "warning product."
)

# Coastal reference points inside the model domain (5-30N, 45-105E).
# Public geography; used to name the segment nearest a flagged ocean area.
COASTAL_SEGMENTS = [
    # (name, region, lat, lon)
    ("Sundarbans / South 24 Parganas", "West Bengal, India", 21.70, 88.50),
    ("Balasore - Bhadrak",             "Odisha, India",      21.30, 87.00),
    ("Paradip - Kendrapara",           "Odisha, India",      20.30, 86.70),
    ("Puri - Ganjam",                  "Odisha, India",      19.70, 85.60),
    ("Srikakulam",                     "Andhra Pradesh, India", 18.30, 84.00),
    ("Visakhapatnam",                  "Andhra Pradesh, India", 17.70, 83.30),
    ("Kakinada - Godavari",            "Andhra Pradesh, India", 16.90, 82.20),
    ("Machilipatnam - Krishna",        "Andhra Pradesh, India", 16.15, 81.10),
    ("Nellore",                        "Andhra Pradesh, India", 14.40, 80.10),
    ("Chennai",                        "Tamil Nadu, India",  13.10, 80.30),
    ("Cuddalore - Puducherry",         "Tamil Nadu, India",  11.75, 79.80),
    ("Nagapattinam - Delta",           "Tamil Nadu, India",  10.80, 79.85),
    ("Gulf of Mannar",                 "Tamil Nadu, India",   9.10, 79.00),
    ("Kanyakumari",                    "Tamil Nadu, India",   8.10, 77.55),
    ("Kochi - Alappuzha",              "Kerala, India",       9.90, 76.25),
    ("Kozhikode - Kannur",             "Kerala, India",      11.60, 75.55),
    ("Mangaluru - Udupi",              "Karnataka, India",   12.95, 74.75),
    ("Goa",                            "Goa, India",         15.40, 73.75),
    ("Ratnagiri",                      "Maharashtra, India", 16.99, 73.30),
    ("Mumbai - Thane",                 "Maharashtra, India", 19.05, 72.85),
    ("Surat - Bharuch",                "Gujarat, India",     21.20, 72.70),
    ("Porbandar - Dwarka",             "Gujarat, India",     21.90, 69.40),
    ("Kutch",                          "Gujarat, India",     23.00, 68.90),
    ("Port Blair",                     "Andaman & Nicobar, India", 11.65, 92.75),
    ("Chattogram",                     "Bangladesh",         22.30, 91.80),
    ("Khulna - Barisal",               "Bangladesh",         22.20, 90.10),
    ("Sittwe - Rakhine",               "Myanmar",            20.15, 92.90),
    ("Ayeyarwady delta",               "Myanmar",            16.20, 95.20),
    ("Colombo",                        "Sri Lanka",           6.93, 79.85),
    ("Trincomalee",                    "Sri Lanka",           8.58, 81.20),
]


def _km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km, vectorised over the first pair."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def risk_index(tchp, uncertainty=None, tchp_anomaly=None):
    """Rule-based RI risk level per cell. Returns an integer array.

        0 LOW   1 WATCH   2 ELEVATED   3 HIGH

    HIGH needs three things together: enough absolute heat to sustain rapid
    intensification, heat that is unusual for that location, and a confident
    estimate. Warm-but-normal water is ELEVATED. Warm-and-unusual water the model
    is unsure about is also ELEVATED -- an operator should see that distinction
    rather than have it averaged away.

    Pass tchp_anomaly=None to get the absolute-threshold-only behaviour, which
    flags almost the whole basin and is mainly useful for showing why the
    anomaly term is needed.
    """
    tchp = np.asarray(tchp, dtype=np.float64)
    out = np.zeros(tchp.shape, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        out[tchp >= TCHP_WATCH] = 1
        hot = tchp >= TCHP_HIGH
        out[hot] = 2
        if uncertainty is None:
            good = np.ones(tchp.shape, bool)
        else:
            good = np.asarray(uncertainty, dtype=np.float64) <= CONF_GOOD
        if tchp_anomaly is None:
            odd = np.ones(tchp.shape, bool)
        else:
            odd = np.asarray(tchp_anomaly, dtype=np.float64) >= TCHP_ANOM
        out[hot & good & odd] = 3
    out[~np.isfinite(tchp)] = 0
    return out


def coastal_exposure(risk, lat, lon, max_km=400.0, min_level=3):
    """Which coastal segments sit near flagged ocean cells.

    Returns a list of dicts, one per segment with any qualifying ocean nearby.
    `max_km` is how far offshore we still consider relevant -- 400 km is roughly
    a day of cyclone travel, not a claim about any particular storm's track.
    """
    risk = np.asarray(risk)
    LA, LO = np.meshgrid(np.asarray(lat), np.asarray(lon), indexing="ij")
    hit = risk >= min_level
    if not hit.any():
        return []
    hlat, hlon = LA[hit], LO[hit]

    rows = []
    for name, region, slat, slon in COASTAL_SEGMENTS:
        d = _km(hlat, hlon, slat, slon)
        near = d <= max_km
        if not near.any():
            continue
        rows.append({
            "segment": name, "region": region,
            "lat": float(slat), "lon": float(slon),
            "n_cells": int(near.sum()),
            "nearest_km": float(d[near].min()),
            "mean_km": float(d[near].mean()),
        })
    rows.sort(key=lambda r: (-r["n_cells"], r["nearest_km"]))
    return rows


def build_alerts(date, risk, tchp, uncertainty, lat, lon, max_km=400.0):
    """Structured alert objects for one day. Empty list when nothing qualifies.

    These are demonstration objects. They are not sent anywhere, and nothing in
    this repository is connected to any real alerting system.
    """
    exposure = coastal_exposure(risk, lat, lon, max_km=max_km, min_level=3)
    if not exposure:
        return []
    LA, LO = np.meshgrid(np.asarray(lat), np.asarray(lon), indexing="ij")
    hit = risk >= 3
    hlat, hlon = LA[hit], LO[hit]
    htchp = np.asarray(tchp)[hit]
    hunc = np.asarray(uncertainty)[hit] if uncertainty is not None else None

    alerts = []
    for row in exposure:
        d = _km(hlat, hlon, row["lat"], row["lon"])
        sel = d <= max_km
        peak = float(np.nanmax(htchp[sel]))
        conf = float(np.nanmean(hunc[sel])) if hunc is not None else float("nan")
        alerts.append({
            "issued": str(date),
            "segment": row["segment"],
            "region": row["region"],
            "level": "HIGH" if peak >= 80 else "ELEVATED",
            "peak_tchp_kj_cm2": round(peak, 1),
            "ocean_cells_flagged": row["n_cells"],
            "nearest_flagged_km": round(row["nearest_km"], 0),
            "model_spread_degC": round(conf, 3) if np.isfinite(conf) else None,
            "basis": f"reconstructed TCHP >= {TCHP_HIGH:g} kJ/cm2 within "
                     f"{max_km:g} km offshore",
            "status": "PROOF OF CONCEPT -- not issued, not transmitted",
            "disclaimer": DISCLAIMER,
        })
    return alerts
