"""OceanEmbed model (torch).

    patch (C,P,P) --> Embedding Engine --> z (128-d) --> Reconstruction Head --> T(15)

Two interchangeable encoders (config model.encoder = cnn | vit):

  CNNEncoder     3 conv blocks, then a pooled summary AND a spatially intact
                 reduced feature map. Cheap, strong, sees eddies and fronts.
  ViTTinyEncoder tokenise the P x P window, a few transformer blocks, CLS token.

The head is an MLP plus optional depth attention: each of the 15 depths owns a
learned query vector that attends over the latent, so the model can route
different parts of the embedding to the mixed layer vs the thermocline vs the
deep water instead of forcing one shared linear map. It also receives a skip
connection carrying the raw centre cell (model.surface_skip).

Import guarded: this module raises only if torch is genuinely missing, and
03_train.py checks backend.torch_available() before importing it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CNNEncoder",
    "ViTTinyEncoder",
    "ReconstructionHead",
    "OceanEmbed",
    "weighted_mse",
    "build_model",
]


class CNNEncoder(nn.Module):
    """Convolutional embedding engine over the N x N surface patch."""

    def __init__(self, in_ch: int, embed_dim: int = 128, patch_size: int = 9,
                 spatial_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.GELU(),
            nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        # Global pooling alone throws away WHERE the eddy sits relative to the
        # cell, and a gradient is precisely a where-question. Keep a reduced but
        # spatially intact copy of the feature map alongside the pooled summary.
        self.reduce = nn.Conv2d(128, spatial_dim, 1)
        flat = spatial_dim * patch_size * patch_size
        self.proj = nn.Sequential(nn.Linear(128 * 2 + flat, embed_dim), nn.GELU())
        self.embed_dim = embed_dim

    def forward(self, x):
        h = self.net(x)
        # mean AND centre pixel: the centre is the cell we are predicting, the
        # mean is its mesoscale context
        gap = h.mean(dim=(2, 3))
        cy, cx = h.shape[2] // 2, h.shape[3] // 2
        centre = h[:, :, cy, cx]
        spatial = self.reduce(h).flatten(1)
        return self.proj(torch.cat([gap, centre, spatial], dim=1))


class ViTTinyEncoder(nn.Module):
    """Minimal ViT over the patch: tokenise, add positions, a few blocks, CLS."""

    def __init__(self, in_ch: int, img: int = 9, patch: int = 3, embed_dim: int = 128,
                 depth: int = 4, heads: int = 4):
        super().__init__()
        while img % patch != 0 and patch > 1:
            patch -= 1
        self.tok = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        n_tok = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        t = self.tok(x).flatten(2).transpose(1, 2)                 # (B,n_tok,D)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1) + self.pos
        return self.norm(self.blocks(t))[:, 0]


class ReconstructionHead(nn.Module):
    """Latent z -> temperature at the 15 standard depths."""

    def __init__(self, embed_dim: int, n_depths: int = 15, hidden: int = 256,
                 depth_attention: bool = True):
        super().__init__()
        self.depth_attention = depth_attention
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        if depth_attention:
            # one learned query per depth, attending over the trunk features
            self.queries = nn.Parameter(torch.randn(n_depths, hidden) * 0.02)
            self.key = nn.Linear(hidden, hidden)
            self.value = nn.Linear(hidden, hidden)
            self.out = nn.Linear(hidden, 1)
            self.scale = 1.0 / math.sqrt(hidden)
        else:
            self.out = nn.Linear(hidden, n_depths)
        self.n_depths = n_depths

    def forward(self, z):
        h = self.trunk(z)                                          # (B,H)
        if not self.depth_attention:
            return self.out(h)
        k = self.key(h).unsqueeze(1)                               # (B,1,H)
        v = self.value(h).unsqueeze(1)                             # (B,1,H)
        q = self.queries.unsqueeze(0).expand(h.shape[0], -1, -1)   # (B,Z,H)
        attn = torch.softmax((q * k).sum(-1, keepdim=True) * self.scale, dim=1)
        ctx = attn * v + q                                         # (B,Z,H) residual query
        return self.out(torch.tanh(ctx)).squeeze(-1)               # (B,Z)


class OceanEmbed(nn.Module):
    """Embedding engine + reconstruction head."""

    def __init__(self, in_ch: int, n_depths: int = 15, encoder: str = "cnn",
                 embed_dim: int = 128, head_hidden: int = 256,
                 depth_attention: bool = True, patch_size: int = 9, vit: dict | None = None,
                 surface_skip: bool = True):
        super().__init__()
        vit = vit or {}
        self.surface_skip = surface_skip
        if encoder == "vit":
            self.encoder = ViTTinyEncoder(
                in_ch, img=patch_size, patch=int(vit.get("patch", 3)),
                embed_dim=embed_dim, depth=int(vit.get("depth", 4)),
                heads=int(vit.get("heads", 4)),
            )
        elif encoder == "cnn":
            self.encoder = CNNEncoder(in_ch, embed_dim=embed_dim, patch_size=patch_size)
        else:
            raise ValueError(f"unknown encoder {encoder!r} (use cnn or vit)")
        # Shallow depths are essentially the surface value itself, so give the
        # head an ungated path to the raw centre cell. Without it the pooled
        # embedding blurs the very cell being predicted and a point model wins
        # in the mixed layer.
        head_in = embed_dim + (in_ch if surface_skip else 0)
        self.head = ReconstructionHead(head_in, n_depths, head_hidden, depth_attention)
        self.encoder_name = encoder

    def embed(self, x):
        """Expose the latent -- this is the 'embedding' the pitch talks about."""
        return self.encoder(x)

    def _head_input(self, x):
        z = self.encoder(x)
        if not self.surface_skip:
            return z
        cy, cx = x.shape[2] // 2, x.shape[3] // 2
        return torch.cat([z, x[:, :, cy, cx]], dim=1)

    def forward(self, x):
        return self.head(self._head_input(x))


def weighted_mse(pred, target, weights=None):
    """Per-depth weighted MSE. Weights up-weight the thermocline."""
    err = (pred - target) ** 2
    if weights is None:
        return err.mean()
    w = weights.to(err.device).view(1, -1)
    return (err * w).sum() / (w.sum() * err.shape[0])


def build_model(cfg: dict, in_ch: int, n_depths: int = 15) -> OceanEmbed:
    m = cfg["model"]
    return OceanEmbed(
        in_ch=in_ch,
        n_depths=n_depths,
        encoder=m.get("encoder", "cnn"),
        embed_dim=int(m.get("embed_dim", 128)),
        head_hidden=int(m.get("head_hidden", 256)),
        depth_attention=bool(m.get("depth_attention", True)),
        patch_size=int(cfg["patch"]["size"]),
        vit=m.get("vit", {}),
        surface_skip=bool(m.get("surface_skip", True)),
    )
