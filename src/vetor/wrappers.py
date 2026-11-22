"""Wrapper reversivel dos MLPs: execucao, skip real, ablacao diagnostica e captura.

Contrato:
- `active=True`  -> chama o MLP original.
- `active=False` -> em entradas de UM token ([B,1,d], decode) retorna zero SEM chamar
  o MLP original. Em prefill (T>1) o MLP e sempre executado (v0.1 nunca pula prefill).
- `ablate_rows`  -> diagnostico: calcula o MLP inteiro e zera a saida do ULTIMO token
  nas linhas do lote indicadas. Nao serve para medir velocidade.
- `capture=True` -> guarda uma *referencia* a z[l,t] do ultimo token (sem clone,
  sem sincronizacao), inclusive quando o MLP e pulado.
"""
from __future__ import annotations

import torch
from torch import nn


class RoutedMLP(nn.Module):
    def __init__(self, inner: nn.Module, layer_idx: int):
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self.active = True
        self.capture = False
        self.captured: torch.Tensor | None = None
        self.ablate_rows: torch.Tensor | None = None
        self.calls_inner = 0
        self.calls_skipped = 0
        self._zero: torch.Tensor | None = None

    def _zeros(self, x: torch.Tensor) -> torch.Tensor:
        z = self._zero
        if z is None or z.shape != x.shape or z.dtype != x.dtype or z.device != x.device:
            z = torch.zeros_like(x)
            self._zero = z
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Esperado [batch,time,dim], recebido {tuple(x.shape)}")
        if self.capture:
            self.captured = x[:, -1, :]
        if not self.active and x.shape[1] == 1:
            self.calls_skipped += 1
            # Buffer reutilizado: o decoder faz `residual + out` (nao in-place).
            return self._zeros(x)
        self.calls_inner += 1
        out = self.inner(x)
        if self.ablate_rows is not None:
            out = out.clone()
            out[self.ablate_rows, -1, :] = 0
        return out


def _layers(model: nn.Module):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("Esperado model.model.layers (arquitetura Llama-like)")
    return layers


def patch_model(model: nn.Module) -> list[RoutedMLP]:
    """Substitui layer.mlp por RoutedMLP, guardando o original em `.inner`."""
    wraps = []
    for i, layer in enumerate(_layers(model)):
        orig = getattr(layer, "mlp", None)
        if orig is None or isinstance(orig, RoutedMLP):
            raise TypeError(f"Camada {i}: MLP ausente ou ja instrumentado")
        w = RoutedMLP(orig, i)
        layer.mlp = w
        wraps.append(w)
    return wraps


def unpatch_model(model: nn.Module) -> None:
    for layer in _layers(model):
        if isinstance(layer.mlp, RoutedMLP):
            layer.mlp = layer.mlp.inner


def reset(wraps: list[RoutedMLP]) -> None:
    for w in wraps:
        w.active = True
        w.capture = False
        w.captured = None
        w.ablate_rows = None


def apply_mask(wraps: list[RoutedMLP], active: list[bool]) -> None:
    if len(active) != len(wraps):
        raise ValueError("Mascara com tamanho diferente do numero de camadas")
    for w, a in zip(wraps, active):
        w.active = bool(a)


def set_capture(wraps: list[RoutedMLP], on: bool) -> None:
    for w in wraps:
        w.capture = on
        if not on:
            w.captured = None


def stack_captured(wraps: list[RoutedMLP], row: int = 0) -> torch.Tensor:
    """[L, d] com z[l, ultimo token] da linha `row` do lote. Nao sincroniza."""
    if any(w.captured is None for w in wraps):
        raise RuntimeError("Captura ausente em alguma camada")
    return torch.stack([w.captured[row] for w in wraps])


def counters(wraps: list[RoutedMLP]) -> tuple[int, int]:
    return sum(w.calls_inner for w in wraps), sum(w.calls_skipped for w in wraps)


def reset_counters(wraps: list[RoutedMLP]) -> None:
    for w in wraps:
        w.calls_inner = 0
        w.calls_skipped = 0
