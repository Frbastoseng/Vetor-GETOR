"""Utilidades do pipeline da v0.4 no 1.7B: janelas por dominio, compensadores, politicas e avaliacao."""
from __future__ import annotations

import numpy as np
import torch

from ..v2.compensate import make_comps
from ..v2.engine import zero_comps
from ..v2.search import make_windows
from .common import OUT
from .data4 import load_texts

P, NW = 64, 64
DOMAINS = ("en_wiki", "pt_wiki", "chat_en", "chat_pt")
COMPS_FILE = OUT / "comps_1.7B.pt"


def windows(tok, split: str, *, allow_final: bool = False, seed: int = 0, part: str = "all",
            per_domain: int | None = None) -> list[torch.Tensor]:
    """1 janela (P+NW tokens) por documento, nos 4 dominios, em ordem de dominio.
    Validacao: part='calib' = 16 primeiros docs de cada dominio (calibracao de tau); 'eval' = demais."""
    out = []
    for i, dom in enumerate(DOMAINS):
        docs = load_texts(dom, split, allow_final=allow_final)
        if split == "val" and part == "calib":
            docs = docs[:16]
        elif split == "val" and part == "eval":
            docs = docs[16:]
        w = make_windows(tok, docs, P, NW, len(docs), seed + 1000 * i)
        out.extend(w[:per_domain] if per_domain else w)
    return out


def domain_of(tok, split: str, *, allow_final: bool = False, seed: int = 0, part: str = "all") -> list[str]:
    out = []
    for i, dom in enumerate(DOMAINS):
        docs = load_texts(dom, split, allow_final=allow_final)
        if split == "val" and part == "calib":
            docs = docs[:16]
        elif split == "val" and part == "eval":
            docs = docs[16:]
        out.extend([dom] * len(make_windows(tok, docs, P, NW, len(docs), seed + 1000 * i)))
    return out


def load_comps(name: str, L: int, d: int, device):
    if name == "ZERO":
        return zero_comps(L, d, device)
    ck = torch.load(COMPS_FILE, weights_only=False)
    if name == "MEAN":
        return make_comps("mean", ck["mean"], L, d, 0, device)
    r = int(name[2:])
    return make_comps("lr", ck["lr"][r], L, d, r, device)


def random_sets(E, n, k, seed):
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(E, k, replace=False).tolist()) for _ in range(n)]
