"""Ajuste dos compensadores (MEAN e LR-r) por regressao de posto reduzido com ridge, forma fechada.

So os compensadores sao ajustados; o LLM fica congelado. Estatisticas suficientes em float64.
"""
from __future__ import annotations

import numpy as np
import torch

from .engine import Compensator


class StatsAccumulator:
    """Acumula, por camada, somas de z, y, z z^T, z y^T, y y^T (float64, CPU)."""

    def __init__(self, L: int, d: int):
        self.L, self.d, self.n = L, d, 0
        z = lambda *s: torch.zeros(*s, dtype=torch.float64)
        self.sz, self.sy = z(L, d), z(L, d)
        self.szz, self.szy, self.syy = z(L, d, d), z(L, d, d), z(L, d)

    def add(self, l: int, zf: torch.Tensor, yf: torch.Tensor):
        """zf, yf [N, d] (no dispositivo, float32)."""
        self.sz[l] += zf.sum(0).cpu().double()
        self.sy[l] += yf.sum(0).cpu().double()
        self.szz[l] += (zf.T @ zf).cpu().double()
        self.szy[l] += (zf.T @ yf).cpu().double()
        self.syy[l] += (yf * yf).sum(0).cpu().double()

    def moments(self, l: int, n: int):
        zb, yb = self.sz[l] / n, self.sy[l] / n
        czz = self.szz[l] / n - torch.outer(zb, zb)
        czy = self.szy[l] / n - torch.outer(zb, yb)
        vy = self.syy[l] / n - yb * yb
        return zb, yb, czz, czy, vy


def collect_stats(model, wraps, batches, device, positions_from: int = 8) -> tuple[StatsAccumulator, int]:
    """batches: iteravel de tensores [B,T] de ids. Usa posicoes >= positions_from."""
    L, d = len(wraps), model.config.hidden_size
    acc = StatsAccumulator(L, d)
    hooks, n_tok = [], 0

    def mk(l):
        def h(mod, inp, out):
            z = inp[0][:, positions_from:].reshape(-1, d).float()
            y = out[:, positions_from:].reshape(-1, d).float()
            acc.add(l, z, y)
        return h

    for w in wraps:
        hooks.append(w.inner.register_forward_hook(mk(w.layer_idx)))
    try:
        with torch.inference_mode():
            for ids in batches:
                model.model(input_ids=ids.to(device), use_cache=False)
                n_tok += ids.shape[0] * (ids.shape[1] - positions_from)
    finally:
        for h in hooks:
            h.remove()
    acc.n = n_tok
    return acc, n_tok


def fit_layer(acc: StatsAccumulator, l: int, rank: int, alpha: float) -> dict:
    """Regressao de posto reduzido: W_ols ridge -> projecao nos top-r autovetores de Cov(yhat)."""
    zb, yb, czz, czy, _ = acc.moments(l, acc.n)
    d = czz.shape[0]
    lam = alpha * torch.trace(czz) / d
    W = torch.linalg.solve(czz + lam * torch.eye(d, dtype=czz.dtype), czy)  # [d,d]
    cyy_hat = W.T @ czz @ W
    evals, evecs = torch.linalg.eigh(cyy_hat)
    Vr = evecs[:, -rank:]  # [d, r] (espaco de saida)
    return {"zbar": zb, "ybar": yb, "V": W @ Vr, "U": Vr.T}  # C(z) = ((z-zb) @ V) @ U + yb


def r2_layer(acc_eval: StatsAccumulator, l: int, fit: dict | None) -> float:
    """R^2 do compensador nos dados de acc_eval (MEAN se fit tem so ybar; ZERO se fit=None)."""
    n = acc_eval.n
    zb, yb, czz, czy, vy = acc_eval.moments(l, n)
    tot = vy.sum() + 0.0  # variancia total de y (em torno da media de avaliacao)
    if fit is None:  # ZERO: erro = E||y||^2
        err = (acc_eval.syy[l] / n).sum()
        return float(1 - err / tot)
    a = fit["ybar"] - yb  # vies de media
    if "V" not in fit:  # MEAN
        return float(1 - (vy.sum() + (a * a).sum()) / tot)
    A = fit["V"] @ fit["U"]  # [d,d], pred = (z - zbar_fit) A + ybar_fit
    # E||y - pred||^2 = tr(Cyy) - 2 tr(A^T Czy) + tr(A^T Czz A) + ||E[y] - E[pred]||^2
    mean_pred = (zb - fit["zbar"]) @ A + fit["ybar"]
    err = vy.sum() - 2 * torch.trace(A.T @ czy) + torch.trace(A.T @ czz @ A) + ((yb - mean_pred) ** 2).sum()
    return float(1 - err / tot)


def make_comps(kind: str, fits: list[dict] | None, L: int, d: int, rank: int, device) -> list[Compensator]:
    out = []
    for l in range(L):
        c = Compensator(kind, d, rank=rank if kind == "lr" else 0)
        if kind in ("mean", "lr"):
            c.ybar.copy_(fits[l]["ybar"].float())
        if kind == "lr":
            c.zbar.copy_(fits[l]["zbar"].float())
            c.V.copy_(fits[l]["V"].float())
            c.U.copy_(fits[l]["U"].float())
        out.append(c.to(device))
    return out


def token_batches(docs_ids: list[list[int]], T: int, B: int, max_batches: int, seed: int):
    """Janelas aleatorias de T tokens dos documentos, em lotes [B,T]."""
    rng = np.random.default_rng(seed)
    pool = [ids for ids in docs_ids if len(ids) > T]
    for _ in range(max_batches):
        rows = []
        for _ in range(B):
            ids = pool[int(rng.integers(len(pool)))]
            s = int(rng.integers(0, len(ids) - T))
            rows.append(ids[s:s + T])
        yield torch.tensor(rows)
