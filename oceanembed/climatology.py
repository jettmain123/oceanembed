"""Per-location climatology, and the anomaly target built from it.

The model was predicting absolute temperature, so most of what it had to learn
was the climatology itself -- the fact that 1000 m is always about 7 degC here.
That is a lookup table, not inference, and any wobble the model introduced at
depth made it WORSE than simply quoting the table. Measured: -124% skill against
climatology at 700 m.

Predicting the ANOMALY instead fixes the incentive:

    A = T - climatology(lat, lon, depth)      model predicts A
    T = climatology + A_pred                  reconstructed at the end

  * predicting A = 0 IS climatology, so zero skill becomes the floor rather
    than something to fall below
  * where the surface carries no information, weight decay pulls A_pred toward
    0, which is the truthful answer at depth
  * the training objective becomes the metric we actually report

The climatology is built from TRAIN DAYS ONLY. Building it from everything would
leak the holdout into the target and quietly inflate every number after it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Climatology", "build_from_samples"]


class Climatology:
    """Mean profile per grid location, with a global fallback.

    Stored as a flat table plus a lookup from rounded (lat, lon), so it survives
    a round trip through npz without needing the original grid.
    """

    def __init__(self, keys: np.ndarray, means: np.ndarray, fallback: np.ndarray,
                 decimals: int = 2):
        self.keys = np.asarray(keys, dtype=np.float64)      # (K, 2) lat, lon
        self.means = np.asarray(means, dtype=np.float32)    # (K, nz)
        self.fallback = np.asarray(fallback, dtype=np.float32)
        self.decimals = int(decimals)
        self._lut = {tuple(k): i for i, k in enumerate(map(tuple, np.round(self.keys, decimals)))}

    # -- use ----------------------------------------------------------------
    def lookup(self, meta) -> np.ndarray:
        """(N, nz) climatological profile for each row of meta [t, lat, lon]."""
        k = np.round(np.asarray(meta)[:, 1:3], self.decimals)
        out = np.empty((k.shape[0], self.means.shape[1]), dtype=np.float32)
        miss = 0
        for j, key in enumerate(map(tuple, k)):
            i = self._lut.get(key)
            if i is None:
                out[j] = self.fallback
                miss += 1
            else:
                out[j] = self.means[i]
        self.last_missing = miss
        return out

    def to_anomaly(self, Y, meta):
        """Absolute temperature -> anomaly."""
        return (np.asarray(Y, dtype=np.float32) - self.lookup(meta)).astype(np.float32)

    def to_absolute(self, A, meta):
        """Anomaly -> absolute temperature."""
        return (np.asarray(A, dtype=np.float32) + self.lookup(meta)).astype(np.float32)

    # -- persistence --------------------------------------------------------
    def pack(self) -> dict:
        return {"clim_keys": self.keys, "clim_means": self.means,
                "clim_fallback": self.fallback,
                "clim_decimals": np.array(self.decimals)}

    @classmethod
    def unpack(cls, d):
        if "clim_keys" not in d:
            return None
        return cls(d["clim_keys"], d["clim_means"], d["clim_fallback"],
                   int(np.asarray(d["clim_decimals"])))


def build_from_samples(trY, trM, decimals: int = 2) -> Climatology:
    """Climatology from the TRAINING samples only.

    Using the whole record would leak the holdout into the target: the anomaly
    for a holdout day would be measured against a mean that already contains
    that day.
    """
    trY = np.asarray(trY, dtype=np.float64)
    key = np.round(np.asarray(trM)[:, 1:3], decimals)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    sums = np.zeros((uniq.shape[0], trY.shape[1]))
    cnts = np.zeros(uniq.shape[0], dtype=np.int64)
    np.add.at(sums, inv, trY)
    np.add.at(cnts, inv, 1)
    means = sums / np.maximum(cnts, 1)[:, None]
    return Climatology(uniq, means.astype(np.float32),
                       trY.mean(axis=0).astype(np.float32), decimals)
