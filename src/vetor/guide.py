"""Vetor-guia (sidecar) e selecao de mascara com orcamento.

    g[l,t] = tanh(P * LayerNorm(z[l,t]) + e[l])
    s[l,t] = w^T g[l,t] + b[l]

`s` e um logit/score de SEGURANCA da omissao (maior = mais seguro omitir).
No alvo `classify`, sigmoid(s) estima P(safe). No alvo `regress`, -s estima log(KL).
Em ambos os casos a politica usa o *ranking* de s, sob orcamento e camadas protegidas.
"""
from __future__ import annotations

import torch
from torch import nn


class VectorGuide(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, guide_dim: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.norm = nn.LayerNorm(hidden_dim)
        self.P = nn.Linear(hidden_dim, guide_dim, bias=False)
        with torch.no_grad():
            self.P.weight.copy_(torch.randn(guide_dim, hidden_dim, generator=g) / hidden_dim**0.5)
        self.e = nn.Embedding(num_layers, guide_dim)
        nn.init.zeros_(self.e.weight)
        self.w = nn.Linear(guide_dim, 1, bias=False)
        self.b = nn.Embedding(num_layers, 1)
        nn.init.zeros_(self.b.weight)
        self.num_layers = num_layers

    def forward(self, z: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or layer_ids.ndim != 1 or z.shape[0] != layer_ids.shape[0]:
            raise ValueError("z deve ser [N, d] e layer_ids [N]")
        g = torch.tanh(self.P(self.norm(z.float())) + self.e(layer_ids))
        return (self.w(g) + self.b(layer_ids)).squeeze(-1)

    @torch.inference_mode()
    def score_layers(self, Z: torch.Tensor) -> torch.Tensor:
        """Z: [L, d] (uma posicao, todas as camadas) -> scores [L], num unico lote."""
        ids = torch.arange(Z.shape[0], device=Z.device)
        return self.forward(Z, ids)


def eligible_layers(n_layers: int, protected_first: int, protected_last: int) -> list[int]:
    return list(range(protected_first, n_layers - protected_last))


def budget_to_k(budget: float, n_eligible: int) -> int:
    return int(round(budget * n_eligible))


def topk_mask(
    scores: torch.Tensor,
    k: int,
    *,
    protected_first: int,
    protected_last: int,
    min_score: float | None = None,
) -> torch.Tensor:
    """Mascara booleana `active` [L] no dispositivo (True = executar MLP).

    Omite as k camadas elegiveis de maior score; com `min_score`, omite apenas as que
    tambem passam do limiar (taxa variavel <= k). Sem sincronizacao com a CPU.
    """
    L = scores.shape[0]
    active = torch.ones(L, dtype=torch.bool, device=scores.device)
    if k <= 0:
        return active
    s = scores.float().clone()
    s[:protected_first] = float("-inf")
    if protected_last:
        s[L - protected_last:] = float("-inf")
    k = min(k, L - protected_first - protected_last)
    idx = torch.topk(s, k).indices
    if min_score is None:
        active[idx] = False
    else:
        active[idx] = ~(s[idx] >= min_score)
    return active


def validate_mask(active: list[bool], k_max: int, protected_first: int, protected_last: int) -> bool:
    L = len(active)
    if any(not a for a in active[:protected_first]) or any(not a for a in active[L - protected_last:]):
        return False
    if active.count(False) > k_max:
        return False
    return True
