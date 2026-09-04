"""OceanEmbed -- subsurface ocean temperature reconstruction from surface fields.

Everything in the package is driven by configs/config.yaml. Import `load_config`
and `ROOT` from here rather than hard-coding paths anywhere else.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

__all__ = ["ROOT", "CONFIG_PATH", "load_config", "resolve", "ensure_dirs", "DEPTHS", "SURFACE_VARS"]

_CACHE: dict | None = None


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load configs/config.yaml (cached for the default path)."""
    global _CACHE
    if path is None:
        if _CACHE is None:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                _CACHE = yaml.safe_load(fh)
        return _CACHE
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel: str) -> Path:
    """Turn a config-relative path into an absolute path under the project root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def ensure_dirs(cfg: dict | None = None) -> None:
    """Create the raw/processed/outputs directories if they are missing."""
    cfg = cfg or load_config()
    for key in ("raw", "processed", "outputs"):
        resolve(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


# Convenience constants -- read once so scripts can import them directly.
DEPTHS = tuple(load_config()["depths"])
SURFACE_VARS = tuple(load_config()["surface_vars"])
