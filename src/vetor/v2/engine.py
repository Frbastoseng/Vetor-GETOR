"""Motor da v0.2: compensadores, wrapper de MLP com dois caminhos e controlador por forward.

Caminhos do wrapper (V2MLP):
1. EXECUCAO REAL (velocidade): `active=False` em entrada de 1 token -> devolve C_l(z) SEM chamar o MLP.
   Com `router` definido (GUIDED-T real), decide antes do MLP e sincroniza 1x por camada.
2. MASCARADO EM LOTE (qualidade, diagnostico): com `ctrl` ativo, calcula MLP(z) para todas as posicoes
   e substitui por C_l(z) onde o controlador decide omitir. Matematicamente identico ao decode com KV
   (atencao causal); NAO serve para medir velocidade. Equivalencia verificada em teste.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


# --------------------------------------------------------------------------- compensadores
class Compensator(nn.Module):
    """C_l(z). kind: zero | mean | lr. Para lr: C(z) = ((z - zbar) @ V) @ U + ybar."""

    def __init__(self, kind: str, d: int, rank: int = 0):
        super().__init__()
        self.kind = kind
        self.rank = rank
        self.register_buffer("ybar", torch.zeros(d))
        self.register_buffer("zbar", torch.zeros(d))
        self.register_buffer("V", torch.zeros(d, max(rank, 1)))
        self.register_buffer("U", torch.zeros(max(rank, 1), d))
        self._zero = None

    @property
    def name(self) -> str:
        return {"zero": "ZERO", "mean": "MEAN"}.get(self.kind, f"LR{self.rank}")

    def flops(self) -> int:
        d = self.ybar.numel()
        return {"zero": 0, "mean": d}.get(self.kind, 2 * d * self.rank + 2 * d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.kind == "zero":
            zz = self._zero
            if zz is None or zz.shape != z.shape or zz.device != z.device or zz.dtype != z.dtype:
                zz = self._zero = torch.zeros_like(z)
            return zz
        if self.kind == "mean":
            return self.ybar.expand_as(z)
        return torch.addmm(self.ybar, ((z.reshape(-1, z.shape[-1]) - self.zbar) @ self.V), self.U).view_as(z)


# --------------------------------------------------------------------------- controlador (modo mascarado)
@dataclass
class Ctrl:
    """Estado de UM forward mascarado [B,T]. `start` = 1a posicao de decode (P)."""
    start: int
    L: int
    static_mask: torch.Tensor | None = None      # [B,T,L] bool, True = omitir
    router: object | None = None                 # callable(z, l, ctrl) -> bool [B,T]
    eligible: torch.Tensor | None = None         # [L] bool
    record: bool = True
    capture_layers: tuple = ()                   # camadas cujo z[B,T,d] deve ser guardado
    bits: torch.Tensor | None = None             # [B,T,L] decisoes tomadas (True = omitido)
    count: torch.Tensor | None = None            # [B,T] omissoes ja feitas no token
    z_store: dict = field(default_factory=dict)
    lag_z: dict = field(default_factory=dict)    # z da camada (para roteador atrasado)

    def init(self, B: int, T: int, device):
        self.bits = torch.zeros(B, T, self.L, dtype=torch.bool, device=device)
        self.count = torch.zeros(B, T, dtype=torch.int32, device=device)
        self.z_store = {}
        self.pos_ok = (torch.arange(T, device=device) >= self.start).view(1, T)


class V2MLP(nn.Module):
    def __init__(self, inner: nn.Module, layer_idx: int, comp: Compensator):
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self.comp = comp
        self.active = True
        self.ctrl: Ctrl | None = None
        self.capture = False
        self.captured = None
        self.calls_inner = 0
        self.calls_skipped = 0
        self.rt_decider = None  # GUIDED-T real: callable(z[1,1,d], l) -> bool (sincroniza)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.ctrl
        if c is None:  # caminho de execucao real
            if self.capture:
                self.captured = x[:, -1, :]
            if x.shape[1] == 1:
                if self.rt_decider is not None:
                    skip = self.rt_decider(x, self.layer_idx)
                else:
                    skip = not self.active
                if skip:
                    self.calls_skipped += 1
                    return self.comp(x)
            self.calls_inner += 1
            return self.inner(x)
        # caminho mascarado (qualidade)
        l = self.layer_idx
        if c.bits is None or c.bits.shape[:2] != x.shape[:2]:
            c.init(x.shape[0], x.shape[1], x.device)
        if l in c.capture_layers:
            c.z_store[l] = x.detach()
        y = self.inner(x)
        if c.static_mask is not None:
            s = c.static_mask[:, :, l] & c.pos_ok
        elif c.router is not None and (c.eligible is None or bool(c.eligible[l])):
            s = c.router(x, l, c) & c.pos_ok
        else:
            return y
        c.bits[:, :, l] = s
        c.count += s.to(torch.int32)
        return torch.where(s.unsqueeze(-1), self.comp(x), y)


def patch_v2(model, comps: list[Compensator]) -> list[V2MLP]:
    out = []
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, V2MLP):
            raise TypeError("ja instrumentado")
        w = V2MLP(layer.mlp, i, comps[i])
        layer.mlp = w
        out.append(w)
    return out


def unpatch_v2(model) -> None:
    for layer in model.model.layers:
        if isinstance(layer.mlp, V2MLP):
            layer.mlp = layer.mlp.inner


def set_comps(wraps: list[V2MLP], comps: list[Compensator]) -> None:
    for w, c in zip(wraps, comps):
        w.comp = c


def set_ctrl(wraps: list[V2MLP], ctrl: Ctrl | None) -> None:
    for w in wraps:
        w.ctrl = ctrl


def zero_comps(L: int, d: int, device) -> list[Compensator]:
    return [Compensator("zero", d).to(device) for _ in range(L)]
