"""Salvar/carregar roteadores da v0.4 (mesma classe da v0.2; tau por regime e por k)."""
from __future__ import annotations

import torch

from ..v2.router2 import TraversalRouter


def save_router4(path, r: TraversalRouter, **meta):
    torch.save({"state_dict": r.state_dict(), "d": r.P.in_features, "L": r.L, "dim": r.P.out_features,
                "lagged": r.lagged, "k_max": r.k_max, "n_el": r.n_el, "y_mu": r.y_mu, "y_sd": r.y_sd, **meta}, path)


def load_router4(path, device):
    ck = torch.load(path, weights_only=False)
    r = TraversalRouter(ck["d"], ck["L"], dim=ck["dim"], lagged=ck["lagged"])
    r.load_state_dict(ck["state_dict"])
    r.k_max, r.n_el, r.y_mu, r.y_sd = ck["k_max"], ck["n_el"], ck["y_mu"], ck["y_sd"]
    return r.to(device).eval(), ck
