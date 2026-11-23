"""Modulos da v0.3. O LLM original fica congelado; so estes modulos sao treinaveis.

    h_{l+1} = u_l + m_l * F_l(z) + (1 - m_l) * C_l(z) + a_{l,t} * G_l(z)

- Correction G_l(z) = B silu(A^T z), B iniciado em zero (comeca identico ao FULL).
- Guide: g = tanh(P LN(z) + e_l); a = sigmoid(w^T g + b_l)  (portao continuo, NO DISPOSITIVO, sem sincronizacao).
- A0 (controle): a = 1 constante.
- m_l estatico (realocacao B): MLP substituido por C_l = MEAN (vetor medio da saida, ja ajustado na v0.2).
- LoRA (controle): W + (alpha/r) B A em Linear escolhidos; fundivel nos pesos (custo zero de latencia).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Correction(nn.Module):
    def __init__(self, d: int, r: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.A = nn.Parameter(torch.randn(d, r, generator=g) / d ** 0.5)
        self.B = nn.Parameter(torch.zeros(r, d))

    def forward(self, z):
        return F.silu(z @ self.A) @ self.B


class Guide(nn.Module):
    """Vetor-guia compartilhado entre as camadas de S (projecao P compartilhada, embedding por camada)."""

    def __init__(self, d: int, layers: list[int], dim: int = 16, gate_bias: float = 0.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.idx = {l: i for i, l in enumerate(layers)}
        self.norm = nn.LayerNorm(d)
        self.P = nn.Linear(d, dim, bias=False)
        with torch.no_grad():
            self.P.weight.copy_(torch.randn(dim, d, generator=g) / d ** 0.5)
        self.e = nn.Parameter(torch.zeros(len(layers), dim))
        self.w = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.full((len(layers),), gate_bias))

    def gate(self, z, l: int):
        i = self.idx[l]
        gv = torch.tanh(self.P(self.norm(z)) + self.e[i])
        return torch.sigmoid(gv @ self.w + self.b[i]).unsqueeze(-1)


class V3MLP(nn.Module):
    def __init__(self, inner: nn.Module, layer_idx: int):
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self.corr: Correction | None = None
        self.guide: Guide | None = None       # None -> portao constante 1 (A0)
        self.replace_with: torch.Tensor | None = None  # MEAN (realocacao B): substitui o MLP
        self.gate_stats = None                # para regularizacao/analise (media do portao no forward)

    def forward(self, z):
        if getattr(self, "_disabled", False):  # modelo FULL (referencia de estabilidade)
            return self.inner(z)
        y = self.replace_with.expand_as(z) if self.replace_with is not None else self.inner(z)
        if self.corr is not None:
            c = self.corr(z)
            if self.guide is not None:
                a = self.guide.gate(z, self.layer_idx)
                self.gate_stats = a
                c = a * c
            y = y + c
        return y


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, generator=g) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin = lin.to(self.base.weight.device, self.base.weight.dtype)
        lin.weight.copy_(self.base.weight + self.scale * (self.B @ self.A))
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        lin.requires_grad_(False)
        lin._vetor_base = self.base  # permite restaurar o Linear original (detach_all)
        return lin


# --------------------------------------------------------------------------- montagem
def attach_vetor(model, layers: list[int], r: int, *, guided: bool, replace: dict[int, torch.Tensor] | None = None,
                 gate_bias: float = 0.0, seed: int = 0) -> dict:
    """Instala correcoes em `layers` (A ou A0) e substituicoes MEAN em `replace` (B). Retorna modulos treinaveis."""
    d = model.config.hidden_size
    dev = next(model.parameters()).device
    guide = Guide(d, layers, gate_bias=gate_bias, seed=seed).to(dev) if guided else None
    corrs = nn.ModuleDict()
    for l, layer in enumerate(model.model.layers):
        w = layer.mlp if isinstance(layer.mlp, V3MLP) else V3MLP(layer.mlp, l)
        layer.mlp = w
        if l in layers:
            w.corr = Correction(d, r, seed=seed + l).to(dev)
            w.guide = guide
            corrs[str(l)] = w.corr
        if replace and l in replace:
            w.replace_with = replace[l].to(dev)
    return {"corrections": corrs, "guide": guide}


def attach_lora(model, r: int = 1, targets=("o_proj", "down_proj"), seed: int = 0) -> nn.ModuleList:
    mods = nn.ModuleList()
    for l, layer in enumerate(model.model.layers):
        for parent in (layer.self_attn, layer.mlp.inner if isinstance(layer.mlp, V3MLP) else layer.mlp):
            for name in targets:
                lin = getattr(parent, name, None)
                if isinstance(lin, nn.Linear):
                    lo = LoRALinear(lin, r, seed=seed + l).to(lin.weight.device)
                    setattr(parent, name, lo)
                    mods.append(lo)
    return mods


def merge_lora(model) -> int:
    n = 0
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp.inner if isinstance(layer.mlp, V3MLP) else layer.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, LoRALinear):
                    setattr(parent, name, child.merged())
                    n += 1
    return n


def detach_all(model) -> None:
    """Remove modulos da v0.3 (volta ao modelo original)."""
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp.inner if isinstance(layer.mlp, V3MLP) else layer.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, LoRALinear):
                    setattr(parent, name, child.base)
                elif hasattr(child, "_vetor_base"):  # LoRA ja fundido
                    setattr(parent, name, child._vetor_base)
        if isinstance(layer.mlp, V3MLP):
            layer.mlp = layer.mlp.inner


def n_params(*mods) -> int:
    return sum(p.numel() for m in mods if m is not None for p in m.parameters() if p.requires_grad)
