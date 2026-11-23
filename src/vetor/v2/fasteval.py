"""Avaliacao de qualidade em um forward por sequencia (modo mascarado), em lote.

Janela: ids = prompt(P) + continuacao(n). Entrada do forward = ids[:P+n-1] (T = P+n-1).
Posicoes de decode D = P..T-1; o logit em p preve ids[p+1]. Identico a v0.1 (passos 1..n-1).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .engine import Ctrl, V2MLP, set_ctrl


@torch.inference_mode()
def logprobs_D(model, wraps: list[V2MLP], ids: torch.Tensor, P: int, ctrl: Ctrl | None) -> torch.Tensor:
    """ids [B,T] -> log-probs [B, T-P, V] nas posicoes de decode (float32)."""
    set_ctrl(wraps, ctrl)
    try:
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, P:, :]
        lg = model.lm_head(h).float()
    finally:
        set_ctrl(wraps, None)
    return F.log_softmax(lg, dim=-1)


def kl_rows(lp_ref: torch.Tensor, lp: torch.Tensor) -> torch.Tensor:
    """KL(ref || q) por posicao. lp_ref [1 ou B, D, V], lp [B, D, V] -> [B, D]."""
    return (lp_ref.exp() * (lp_ref - lp)).sum(-1).clamp_min(0)


def nll_rows(lp: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """lp [B,D,V], targets [D] -> NLL [B,D]."""
    return -lp.gather(-1, targets.view(1, -1, 1).expand(lp.shape[0], -1, 1)).squeeze(-1)


def window_io(window: torch.Tensor, P: int) -> tuple[torch.Tensor, torch.Tensor]:
    """window [P+n] -> (entrada [1, P+n-1], alvos [n-1]) — alvos das posicoes de decode D."""
    return window[:-1].view(1, -1), window[P + 1:]


@torch.inference_mode()
def eval_masks(model, wraps, window: torch.Tensor, P: int, masks: torch.Tensor, chunk: int = 32,
               lp_full: torch.Tensor | None = None) -> dict:
    """Avalia mascaras estaticas [M, T, L] (T = P+n-1) na MESMA janela [P+n]. KL/NLL por mascara e posicao."""
    x, tgt = window_io(window, P)
    if lp_full is None:
        lp_full = logprobs_D(model, wraps, x, P, None)
    kls, nlls = [], []
    for i in range(0, masks.shape[0], chunk):
        m = masks[i:i + chunk]
        ctrl = Ctrl(start=P, L=len(wraps), static_mask=m)
        lp = logprobs_D(model, wraps, x.expand(m.shape[0], -1), P, ctrl)
        kls.append(kl_rows(lp_full, lp))
        nlls.append(nll_rows(lp, tgt))
    return {"kl": torch.cat(kls), "nll": torch.cat(nlls), "lp_full": lp_full,
            "nll_full": nll_rows(lp_full, tgt)[0]}


def seq_mask(layers, T: int, P: int, L: int, device) -> torch.Tensor:
    """Mascara [T, L] omitindo `layers` em todas as posicoes de decode (>= P)."""
    m = torch.zeros(T, L, dtype=torch.bool, device=device)
    if len(layers):
        m[P:, list(layers)] = True
    return m
