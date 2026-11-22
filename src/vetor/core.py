"""Primitivas testaveis antes de integrar modelos Hugging Face.

Atencao: a operacao `skip_decode` e potencialmente incorreta do ponto de vista
linguistico; so se torna valida como aproximacao apos a avaliacao experimental.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class InstrumentedMLP(nn.Module):
    """Wrapper reversivel, acionado pelo chamador.

    Captura o ultimo vetor de entrada e implementa dois modos distintos:
    - `skip_decode`: omite o MLP APENAS para um token de decodificacao.
    - `ablate_last`: abla somente o ultimo token de uma sequencia (diagnostico,
      NAO e aceleracao; o MLP original ainda e calculado neste modo).

    O wrapper **nao** toma decisoes probabilisticas: a politica deve ser externa.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.skip_decode = False
        self.ablate_last = False
        self.capture = False
        self.last_input: torch.Tensor | None = None
        self.forward_calls = 0
        self.skipped_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch,time,dim], got {tuple(x.shape)}")
        self.forward_calls += 1
        if self.capture:
            self.last_input = x[:, -1, :].detach()
        if self.skip_decode and x.shape[1] == 1:
            self.skipped_calls += 1
            return torch.zeros_like(x)
        out = self.inner(x)
        if self.ablate_last:
            # Preserve earlier positions for a causal teacher-forcing ablation.
            out = torch.cat((out[:, :-1, :], torch.zeros_like(out[:, -1:, :])), dim=1)
        return out


class VectorGuide(nn.Module):
    """Pequeno guia treinavel que recebe z[l,t] e o indice da camada.

    Retorna logit de PROBABILIDADE DE OMISSAO SEGURA, nunca uma decisao final.
    Uma politica externa impõe limites de risco/orcamento/posicoes protegidas.
    """

    def __init__(self, hidden_dim: int, num_layers: int, guide_dim: int = 16):
        super().__init__()
        self.normalizer = nn.LayerNorm(hidden_dim)
        self.project = nn.Linear(hidden_dim, guide_dim, bias=False)
        self.layer_embedding = nn.Embedding(num_layers, guide_dim)
        self.head = nn.Linear(guide_dim, 1, bias=False)
        self.layer_bias = nn.Embedding(num_layers, 1)

    def forward(self, z: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or layer_ids.ndim != 1 or z.shape[0] != layer_ids.shape[0]:
            raise ValueError("z must be [N, hidden_dim]; layer_ids must be [N]")
        g = torch.tanh(self.project(self.normalizer(z)) + self.layer_embedding(layer_ids))
        return (self.head(g) + self.layer_bias(layer_ids)).squeeze(-1)


def choose_mask(
    safe_probs: list[float],
    *,
    max_skips: int,
    threshold: float = 0.9,
    protected_first: int = 2,
    protected_last: int = 2,
) -> list[bool]:
    """Return `active[l]` — True means execute MLP, False means skip.

    A politica seleciona os candidatos com maior seguranca esperada,
    protegendo bordas e nunca ultrapassando max_skips.
    """
    n = len(safe_probs)
    if max_skips < 0 or not (0 <= threshold <= 1):
        raise ValueError("Invalid routing budget/threshold")
    if any(not (0 <= p <= 1) or p != p for p in safe_probs):
        raise ValueError("Probabilities must be finite and in [0,1]")
    active = [True] * n
    candidates = [(p, i) for i, p in enumerate(safe_probs)
                  if protected_first <= i < n - protected_last and p >= threshold]
    candidates.sort(key=lambda x: (-x[0], x[1]))
    for _, i in candidates[:max_skips]:
        active[i] = False
    return active


def patch_llama_like_model(model: nn.Module) -> list[InstrumentedMLP]:
    """Patch only Llama-like decoder MLPs; preserves original in `.inner`.

    Must be called once, before collecting baseline data using wrappers.
    Does not claim compatibility with all Transformers versions.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("Expected model.model.layers; verify selected architecture")
    wrapped: list[InstrumentedMLP] = []
    for layer in layers:
        original = getattr(layer, "mlp", None)
        if original is None or isinstance(original, InstrumentedMLP):
            raise TypeError("Missing MLP or already patched")
        wrapper = InstrumentedMLP(original)
        layer.mlp = wrapper
        wrapped.append(wrapper)
    return wrapped


def unpatch_llama_like_model(model: nn.Module) -> None:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("Expected model.model.layers")
    for layer in layers:
        if isinstance(layer.mlp, InstrumentedMLP):
            layer.mlp = layer.mlp.inner
