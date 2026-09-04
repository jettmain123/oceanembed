"""NumPy fallback model -- the pipeline trains even with no working torch.

Architecture is the same shape as the torch path, just smaller: the flattened
patch goes through an MLP encoder to a 128-d latent, then a head predicts all 15
depths. Same weighted-MSE loss, same Adam optimiser, same checkpoint contract
(a dict of arrays), so 04_evaluate.py and the demo do not care which backend
trained the model.

Not a replacement for the CNN -- it is the safety net that keeps the end-to-end
demo alive on a machine where torch is broken.
"""
from __future__ import annotations

import numpy as np

__all__ = ["NumpyOceanEmbed", "weighted_mse_np"]


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def _dgelu(x):
    t = np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * 0.7978845608 * (1 + 3 * 0.044715 * x ** 2)


def weighted_mse_np(pred, target, weights=None):
    err = (pred - target) ** 2
    if weights is None:
        return float(err.mean())
    w = np.asarray(weights).reshape(1, -1)
    return float((err * w).sum() / (w.sum() * err.shape[0]))


class NumpyOceanEmbed:
    """MLP encoder (-> latent) + MLP head (-> 15 depths), trained with Adam."""

    def __init__(self, in_dim: int, n_depths: int = 15, embed_dim: int = 128,
                 hidden: int = 256, seed: int = 42):
        rng = np.random.default_rng(seed)

        def he(a, b):
            return (rng.normal(0, np.sqrt(2.0 / a), size=(a, b))).astype(np.float32)

        self.p = {
            "W1": he(in_dim, hidden), "b1": np.zeros(hidden, np.float32),
            "W2": he(hidden, embed_dim), "b2": np.zeros(embed_dim, np.float32),
            "W3": he(embed_dim, hidden), "b3": np.zeros(hidden, np.float32),
            "W4": he(hidden, n_depths), "b4": np.zeros(n_depths, np.float32),
        }
        self.embed_dim = embed_dim
        self.n_depths = n_depths
        self.in_dim = in_dim
        self._m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._t = 0

    # -- forward -------------------------------------------------------------
    def _flat(self, X):
        X = np.asarray(X, dtype=np.float32)
        return X.reshape(X.shape[0], -1)

    def embed(self, X):
        x = self._flat(X)
        z1 = x @ self.p["W1"] + self.p["b1"]
        a1 = _gelu(z1)
        return a1 @ self.p["W2"] + self.p["b2"]

    def forward(self, X, cache: bool = False):
        x = self._flat(X)
        z1 = x @ self.p["W1"] + self.p["b1"]; a1 = _gelu(z1)
        z2 = a1 @ self.p["W2"] + self.p["b2"]; a2 = _gelu(z2)          # latent
        z3 = a2 @ self.p["W3"] + self.p["b3"]; a3 = _gelu(z3)
        out = a3 @ self.p["W4"] + self.p["b4"]
        if cache:
            return out, (x, z1, a1, z2, a2, z3, a3)
        return out

    def predict(self, X, batch: int = 4096):
        outs = [self.forward(X[i:i + batch]) for i in range(0, len(X), batch)]
        return np.concatenate(outs).astype(np.float32)

    # -- training ------------------------------------------------------------
    def _backward(self, cache, pred, target, weights):
        x, z1, a1, z2, a2, z3, a3 = cache
        n = pred.shape[0]
        w = np.ones((1, pred.shape[1]), np.float32) if weights is None else np.asarray(weights, np.float32).reshape(1, -1)
        dout = 2.0 * (pred - target) * w / (w.sum() * n)

        g = {}
        g["W4"] = a3.T @ dout; g["b4"] = dout.sum(0)
        da3 = dout @ self.p["W4"].T
        dz3 = da3 * _dgelu(z3)
        g["W3"] = a2.T @ dz3; g["b3"] = dz3.sum(0)
        da2 = dz3 @ self.p["W3"].T
        dz2 = da2 * _dgelu(z2)
        g["W2"] = a1.T @ dz2; g["b2"] = dz2.sum(0)
        da1 = dz2 @ self.p["W2"].T
        dz1 = da1 * _dgelu(z1)
        g["W1"] = x.T @ dz1; g["b1"] = dz1.sum(0)
        return g

    def _adam(self, g, lr, wd, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, grad in g.items():
            if k.startswith("W") and wd:
                grad = grad + wd * self.p[k]
            self._m[k] = b1 * self._m[k] + (1 - b1) * grad
            self._v[k] = b2 * self._v[k] + (1 - b2) * grad ** 2
            mh = self._m[k] / (1 - b1 ** self._t)
            vh = self._v[k] / (1 - b2 ** self._t)
            self.p[k] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

    def fit(self, X, Y, Xva=None, Yva=None, epochs=15, batch_size=256, lr=8e-4,
            weight_decay=1e-4, weights=None, seed=42, verbose=True, on_epoch=None):
        rng = np.random.default_rng(seed)
        n = len(X)
        history = []
        best = (np.inf, None)
        for ep in range(1, epochs + 1):
            idx = rng.permutation(n)
            tot, seen = 0.0, 0
            for s in range(0, n, batch_size):
                b = idx[s:s + batch_size]
                xb, yb = X[b], Y[b]
                pred, cache = self.forward(xb, cache=True)
                loss = weighted_mse_np(pred, yb, weights)
                g = self._backward(cache, pred, yb, weights)
                self._adam(g, lr, weight_decay)
                tot += loss * len(b); seen += len(b)
            tr = tot / max(seen, 1)
            va = np.nan
            if Xva is not None:
                va = weighted_mse_np(self.predict(Xva), Yva, weights)
                if va < best[0]:
                    best = (va, {k: v.copy() for k, v in self.p.items()})
            history.append({"epoch": ep, "train_loss": float(tr), "val_loss": float(va)})
            if verbose:
                print(f"  epoch {ep:3d}/{epochs}  train {tr:.5f}  val {va:.5f}")
            if on_epoch:
                on_epoch(history[-1])
        if best[1] is not None:
            self.p = best[1]
        return history

    # -- persistence ---------------------------------------------------------
    def state_dict(self):
        return dict(self.p)

    def load_state_dict(self, sd):
        for k in self.p:
            self.p[k] = np.asarray(sd[k], dtype=np.float32)

    def save(self, path, extra: dict | None = None):
        np.savez(path, **self.p, **{f"meta_{k}": v for k, v in (extra or {}).items()},
                 in_dim=self.in_dim, n_depths=self.n_depths, embed_dim=self.embed_dim)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        m = cls(int(z["in_dim"]), int(z["n_depths"]), int(z["embed_dim"]))
        m.load_state_dict({k: z[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4")})
        return m
