"""Load a trained checkpoint and predict, whichever backend produced it.

04_evaluate.py and 05_demo.py both go through Predictor so there is exactly one
place that knows how a checkpoint is laid out.

    p = Predictor.load()
    temp_degC = p.predict(X_raw)        # X_raw is UNSCALED (N,C,P,P)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import load_config, resolve
from .backend import torch_available
from .dataset import apply_scalers, invert_target

__all__ = ["Predictor"]


class Predictor:
    def __init__(self, model, backend: str, xm, xs, ym, ys, depths, meta: dict | None = None):
        self.model = model
        self.backend = backend
        self.xm, self.xs, self.ym, self.ys = xm, xs, ym, ys
        self.depths = np.asarray(depths, dtype=np.float32)
        self.meta = meta or {}

    # -- construction --------------------------------------------------------
    @classmethod
    def load(cls, cfg: dict | None = None, prefer: str | None = None) -> "Predictor":
        """Load the torch checkpoint if present and usable, else the NumPy one."""
        cfg = cfg or load_config()
        pt = resolve(cfg["paths"]["checkpoint"])
        npz = resolve(cfg["paths"]["checkpoint_npy"])

        want_torch = (prefer != "numpy") and pt.exists() and torch_available()
        if want_torch:
            return cls._load_torch(pt, cfg)
        if npz.exists():
            return cls._load_numpy(npz, cfg)
        if pt.exists():
            raise RuntimeError(f"{pt} exists but torch is unusable and no NumPy checkpoint found")
        raise FileNotFoundError("no checkpoint -- run scripts/03_train.py first")

    @classmethod
    def _load_torch(cls, path: Path, cfg: dict) -> "Predictor":
        import torch

        from .model import OceanEmbed

        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = ck.get("cfg_model", cfg["model"])
        model = OceanEmbed(
            in_ch=int(ck["in_ch"]), n_depths=int(ck["n_depths"]),
            encoder=m.get("encoder", "cnn"), embed_dim=int(m.get("embed_dim", 128)),
            head_hidden=int(m.get("head_hidden", 256)),
            depth_attention=bool(m.get("depth_attention", True)),
            patch_size=int(ck.get("patch", cfg["patch"])["size"]), vit=m.get("vit", {}),
            surface_skip=bool(m.get("surface_skip", True)),
        )
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return cls(model, "torch", ck["xm"], ck["xs"], ck["ym"], ck["ys"], ck["depths"],
                   {"encoder": m.get("encoder", "cnn"), "epoch": ck.get("epoch"),
                    "val_loss": ck.get("val_loss"), "path": str(path)})

    @classmethod
    def _load_numpy(cls, path: Path, cfg: dict) -> "Predictor":
        from .model_numpy import NumpyOceanEmbed

        model = NumpyOceanEmbed.load(path)
        sc = np.load(str(path).replace(".npz", "_scalers.npz"))
        return cls(model, "numpy", sc["xm"], sc["xs"], sc["ym"], sc["ys"], sc["depths"],
                   {"encoder": "numpy_mlp", "path": str(path)})

    # -- use -----------------------------------------------------------------
    def predict(self, X_raw: np.ndarray, batch: int = 4096) -> np.ndarray:
        """Unscaled patches (N,C,P,P) -> temperature in degrees Celsius (N,15)."""
        Xn, _ = apply_scalers(np.asarray(X_raw, dtype=np.float32), None,
                              self.xm, self.xs, self.ym, self.ys)
        if self.backend == "torch":
            import torch

            outs = []
            with torch.no_grad():
                for i in range(0, len(Xn), batch):
                    outs.append(self.model(torch.from_numpy(Xn[i:i + batch])).numpy())
            pred = np.concatenate(outs)
        else:
            pred = self.model.predict(Xn, batch=batch)
        return invert_target(pred, self.ym, self.ys)

    def predict_mc(self, X_raw, n: int = 20, p: float = 0.15, seed: int = 0, batch: int = 4096):
        """Monte-Carlo uncertainty: mean and standard deviation over n passes.

        The network has no dropout layers, so we cannot do classic MC-dropout on
        it. Instead we resample the INPUT the way the model was trained: with
        `channel_dropout`, whole variables are blanked 15% of the time, so the
        model is already calibrated to that perturbation. Spread across passes
        answers a question an operator actually asks -- "how much does this
        estimate depend on any single satellite?"

        Returns (mean, std), both (N, 15) in degrees Celsius. Large std means the
        answer hinges on one input, so treat it with caution.
        """
        X = np.asarray(X_raw, dtype=np.float32)
        rng = np.random.default_rng(seed)
        acc = []
        for i in range(n):
            Xi = X.copy()
            drop = rng.random(X.shape[1]) < p
            if drop.all():                       # never blank everything
                drop[rng.integers(X.shape[1])] = False
            for c in np.nonzero(drop)[0]:
                Xi[:, c] = self.xm[0, c]         # the mean is the no-information value
            acc.append(self.predict(Xi, batch=batch))
        A = np.stack(acc)
        return A.mean(axis=0), A.std(axis=0)

    def embed(self, X_raw: np.ndarray) -> np.ndarray:
        """The 128-d latent -- the 'embedding' the architecture is named for."""
        Xn, _ = apply_scalers(np.asarray(X_raw, dtype=np.float32), None,
                              self.xm, self.xs, self.ym, self.ys)
        if self.backend == "torch":
            import torch

            with torch.no_grad():
                return self.model.embed(torch.from_numpy(Xn)).numpy()
        return self.model.embed(Xn)

    def __repr__(self) -> str:
        return f"<Predictor backend={self.backend} encoder={self.meta.get('encoder')} depths={self.depths.size}>"
