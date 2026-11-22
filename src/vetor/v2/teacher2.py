"""Professor da v0.2: efeito marginal ACUMULADO de cada decisao (t,l) sob uma mascara-base.

    Delta_N(t,l | M_b) = sum_{p=t}^{t+N-1} KL(q_exec(p) || q_skip(p))
onde q_exec/q_skip = M_b com (t,l) executado/omitido, demais decisoes congeladas (uma delas e a propria base).
Inclui acumulacao (efeito via KV em p > t) e interacao (medido sob as omissoes reais de M_b).
Features guardadas: z[l,t] e z[l,t-1] sob M_b, s(t,<l), n(t,<l) — exatamente o que o roteador ve.
"""
from __future__ import annotations

import numpy as np
import torch

from .engine import Ctrl
from .fasteval import kl_rows, logprobs_D, nll_rows

NS = (8, 16, 32)


@torch.inference_mode()
def label_window(model, wraps, window: torch.Tensor, P: int, base: torch.Tensor, positions: list[int],
                 layers: list[int], device, chunk: int = 64) -> dict:
    """window [P+n]; base [T,L] bool (True = omitir). Retorna features e rotulos por celula (t,l)."""
    L = len(wraps)
    w = window.to(device)
    x, tgt = w[:-1].view(1, -1), w[P + 1:]
    base = base.to(device)
    ctrl = Ctrl(start=P, L=L, static_mask=base.unsqueeze(0), capture_layers=tuple(range(L)))
    lp_b = logprobs_D(model, wraps, x, P, ctrl)                     # [1,D,V]
    nll_b = nll_rows(lp_b, tgt)[0]                                    # [D]
    Z = torch.stack([ctrl.z_store[l][0] for l in range(L)], 1)        # [T,L,d]
    cells = [(t, l) for t in positions for l in layers]
    masks = base.unsqueeze(0).repeat(len(cells), 1, 1)
    was = torch.tensor([bool(base[t, l]) for t, l in cells])
    for i, (t, l) in enumerate(cells):
        masks[i, t, l] = not was[i]  # vira a decisao da celula
    D = x.shape[1] - P
    out = {f"d{N}": [] for N in NS} | {f"dnll{N}": [] for N in NS}
    for j in range(0, len(cells), chunk):
        m = masks[j:j + chunk]
        lp = logprobs_D(model, wraps, x.expand(m.shape[0], -1), P, Ctrl(start=P, L=L, static_mask=m))
        nll = nll_rows(lp, tgt)
        for r in range(m.shape[0]):
            i = j + r
            t, _ = cells[i]
            p0 = t - P
            if was[i]:   # base omitia: linha = executar
                kl = kl_rows(lp[r:r + 1], lp_b)[0]
                dn = nll_b - nll[r]
            else:        # base executava: linha = omitir
                kl = kl_rows(lp_b, lp[r:r + 1])[0]
                dn = nll[r] - nll_b
            for N in NS:
                sl = slice(p0, min(p0 + N, D))
                out[f"d{N}"].append(float(kl[sl].sum()))
                out[f"dnll{N}"].append(float(dn[sl].sum()))
    ts = torch.tensor([t for t, _ in cells])
    ls = torch.tensor([l for _, l in cells])
    bits = torch.stack([base[t].clone() for t, _ in cells]).cpu()
    for i, (t, l) in enumerate(cells):
        bits[i, l:] = False
    return {"z": Z[ts, ls].half().cpu(), "z_lag": Z[ts - 1, ls].half().cpu(), "lid": ls, "bits": bits,
            "n": bits.sum(1), "was_skipped": was, **{k: torch.tensor(v) for k, v in out.items()}}


def sample_positions(P: int, n: int, k: int, rng, horizon: int = 32) -> list[int]:
    last = P + n - 2 - (horizon - 1)  # garante janela completa de N=32
    return sorted(rng.choice(np.arange(P, last + 1), size=k, replace=False).tolist())


def concat(parts: list[dict]) -> dict:
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
