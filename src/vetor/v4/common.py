"""Registro de modelos e utilidades comuns da v0.4 (reaproveita o motor da v0.2 sem altera-lo)."""
from __future__ import annotations

import copy

import torch

from ..model_adapter import ROOT, load_config, load_model

MODELS = {
    "135M": {"id": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac"},
    "1.7B": {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "revision": "31b70e2e869a7173562077fd711b654946d38674"},
}
OUT = ROOT / "results" / "v0.4"
PROTECT_FIRST, PROTECT_LAST = 2, 2


def cfg_for(name: str) -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["model"] = {**MODELS[name], "dtype": "float32"}
    return cfg


def load(name: str, device: str, dtype: torch.dtype = torch.float32):
    return load_model(cfg_for(name), device, dtype=dtype)


def eligible(L: int) -> list[int]:
    return list(range(PROTECT_FIRST, L - PROTECT_LAST))


def flops_of(fn, *args) -> int:
    """FLOPs contados pelo PyTorch (FlopCounterMode) numa chamada — nao por formula."""
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc, torch.inference_mode():
        fn(*args)
    return int(fc.get_total_flops())
