"""Roteador da v0.2 (GUIDED-T e variante atrasada GUIDED-DL).

    g = tanh(P·LN(z) + e[l] + S·s(t,<l) + a·n(t,<l)/|E|)
    yhat = w^T g + b[l]            ~ log(Delta_N + eps)  (risco previsto; menor = mais seguro omitir)
    omitir (t,l)  <=>  l elegivel  e  yhat <= tau  e  n(t,<l) < k_max

GUIDED-T : z = z[l,t]   (antes do MLP da camada atual, no token atual)
GUIDED-DL: z = z[l,t-1] (mesma camada, token anterior) — isola o efeito de decidir com atraso.
Ambos usam s(t,<l) e n(t,<l): decisoes ja tomadas no token atual (causais).
"""
from __future__ import annotations

import torch
from torch import nn


class TraversalRouter(nn.Module):
    def __init__(self, d: int, L: int, dim: int = 16, lagged: bool = False, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.L, self.lagged = L, lagged
        self.norm = nn.LayerNorm(d)
        self.P = nn.Linear(d, dim, bias=False)
        with torch.no_grad():
            self.P.weight.copy_(torch.randn(dim, d, generator=g) / d ** 0.5)
        self.e = nn.Embedding(L, dim)
        nn.init.zeros_(self.e.weight)
        self.S = nn.Linear(L, dim, bias=False)
        nn.init.zeros_(self.S.weight)
        self.a = nn.Parameter(torch.zeros(dim))
        self.w = nn.Linear(dim, 1, bias=False)
        self.b = nn.Embedding(L, 1)
        nn.init.zeros_(self.b.weight)
        # regra de decisao (calibrada na validacao)
        self.register_buffer("tau", torch.tensor(0.0))
        self.register_buffer("eligible", torch.zeros(L, dtype=torch.bool))
        self.k_max = 0
        self.n_el = 1
        self.y_mu, self.y_sd = 0.0, 1.0  # padronizacao do alvo (treino)

    def forward(self, z: torch.Tensor, lid: torch.Tensor, bits: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """z [N,d], lid [N], bits [N,L] (so camadas < l), n [N] -> yhat padronizado [N]."""
        h = self.P(self.norm(z.float())) + self.e(lid) + self.S(bits.float()) + self.a * (n.float() / self.n_el).unsqueeze(-1)
        return (self.w(torch.tanh(h)) + self.b(lid)).squeeze(-1)

    def configure(self, eligible: list[int], k_max: int, tau: float | None = None):
        self.eligible.zero_()
        self.eligible[eligible] = True
        self.n_el = len(eligible)
        self.k_max = k_max
        if tau is not None:
            self.tau.fill_(tau)

    # ---- modo mascarado: chamado pelo V2MLP com z [B,T,d] e o Ctrl
    @torch.inference_mode()
    def masked(self, z: torch.Tensor, l: int, ctrl) -> torch.Tensor:
        B, T, d = z.shape
        if self.lagged:
            z = torch.cat([z[:, :1], z[:, :-1]], dim=1)  # posicao t usa z[l, t-1]
        bits = ctrl.bits.clone()
        bits[:, :, l:] = False
        lid = torch.full((B * T,), l, device=z.device, dtype=torch.long)
        y = self.forward(z.reshape(-1, d), lid, bits.view(-1, self.L), ctrl.count.view(-1)).view(B, T)
        return (y <= self.tau) & (ctrl.count < self.k_max)

    # ---- execucao real (1 token): estado por token, 1 sincronizacao por camada elegivel
    def reset_state(self, device):
        self._bits = torch.zeros(1, self.L, dtype=torch.bool, device=device)
        self._n = 0
        self._zprev = {}

    @torch.inference_mode()
    def decide_real(self, x: torch.Tensor, l: int) -> bool:
        zc = x[:, -1, :]
        z = self._zprev.get(l, zc) if self.lagged else zc
        if self.lagged:
            self._zprev[l] = zc
        if l == 0:
            self._bits.zero_()
            self._n = 0
        if not bool(self.eligible[l]) or self._n >= self.k_max:
            return False
        bits = self._bits.clone()
        bits[:, l:] = False
        lid = torch.tensor([l], device=x.device)
        y = self.forward(z, lid, bits, torch.tensor([self._n], device=x.device))
        skip = bool((y <= self.tau).item())  # sincronizacao GPU->CPU (1 por camada elegivel)
        if skip:
            self._bits[0, l] = True
            self._n += 1
        return skip
