"""Estatistica da v0.4: bootstrap pareado da diferenca de dNLL relativo (razao de somas) entre politicas."""
from __future__ import annotations

import numpy as np


def rel(nll, full):
    return 100 * (np.sum(nll) / np.sum(full) - 1)


def boot_rel(nll, full, n_boot=10000, seed=0):
    nll, full = np.asarray(nll, float), np.asarray(full, float)
    idx = np.random.default_rng(seed).integers(0, len(nll), (n_boot, len(nll)))
    b = 100 * (nll[idx].sum(1) / full[idx].sum(1) - 1)
    return {"rel_dnll_pct": float(rel(nll, full)), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def paired_diff(nll_a, nll_b, full, n_boot=10000, seed=0):
    """dNLL%(a) - dNLL%(b) nas MESMAS janelas reamostradas. Negativo = a melhor."""
    a, b, f = (np.asarray(x, float) for x in (nll_a, nll_b, full))
    idx = np.random.default_rng(seed).integers(0, len(a), (n_boot, len(a)))
    fs = f[idx].sum(1)
    d = 100 * (a[idx].sum(1) - b[idx].sum(1)) / fs
    return {"diff_pp": float(100 * (a.sum() - b.sum()) / f.sum()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def rep4(tokens) -> float:
    """Fracao de 4-gramas repetidos numa sequencia de ids."""
    g = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)
