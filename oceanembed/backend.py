"""Backend detection.

`torch_available()` actually runs a tensor op, so a torch that imports but is
broken (bad wheel, missing MKL/DLL) correctly falls back to the NumPy model
instead of blowing up halfway through training.
"""
from __future__ import annotations

import functools

__all__ = ["torch_available", "backend_name", "describe"]

_REASON = "not checked"


@functools.lru_cache(maxsize=1)
def torch_available() -> bool:
    global _REASON
    try:
        import torch

        a = torch.randn(4, 3)
        b = torch.randn(3, 2)
        c = a @ b
        d = torch.nn.Linear(2, 2)(c)
        float(d.detach().sum())
        _REASON = f"torch {torch.__version__}"
        return True
    except Exception as exc:                       # ImportError or a broken runtime
        _REASON = f"{type(exc).__name__}: {exc}"
        return False


def backend_name() -> str:
    return "torch" if torch_available() else "numpy"


def describe() -> str:
    ok = torch_available()
    return f"backend={backend_name()} ({_REASON})" if ok else f"backend=numpy (torch unusable -- {_REASON})"
