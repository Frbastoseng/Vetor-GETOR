"""Estatistica pareada da v0.3: diferenca macro (p.p.) entre dois resultados nos MESMOS exemplos."""
from __future__ import annotations

import numpy as np


def _score(vals):
    """Media do escore por exemplo; para pares (gabarito, predicao) calcula F1 macro."""
    if vals and isinstance(vals[0], (list, tuple)):
        gold = np.array([v[0] for v in vals]); pred = np.array([v[1] for v in vals])
        f1s = []
        for c in np.unique(np.concatenate([gold, pred])):
            tp = np.sum((pred == c) & (gold == c)); fp = np.sum((pred == c) & (gold != c)); fn = np.sum((pred != c) & (gold == c))
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        return float(np.mean(f1s))
    return float(np.mean(vals))


def aligned(a: dict, b: dict, task: str):
    da = dict((d, v) for d, v in a["tasks"][task]["per_example"])
    db = dict((d, v) for d, v in b["tasks"][task]["per_example"])
    ids = sorted(set(da) & set(db))
    return [da[i] for i in ids], [db[i] for i in ids]


def paired_macro(a: dict, b: dict, tasks: list[str], n_boot: int = 10000, seed: int = 0) -> dict:
    """Diferenca macro (a - b) em p.p., IC95 por bootstrap pareado (reamostragem por tarefa)."""
    rng = np.random.default_rng(seed)
    per, boots = {}, np.zeros(n_boot)
    for t in tasks:
        va, vb = aligned(a, b, t)
        n = len(va)
        per[t] = {"a": 100 * _score(va), "b": 100 * _score(vb), "diff": 100 * (_score(va) - _score(vb)), "n": n}
        idx = rng.integers(0, n, (n_boot, n))
        if va and isinstance(va[0], (list, tuple)):
            d = np.array([_score([va[i] for i in row]) - _score([vb[i] for i in row]) for row in idx])
        else:
            A, B = np.array(va, float), np.array(vb, float)
            d = A[idx].mean(1) - B[idx].mean(1)
        boots += 100 * d / len(tasks)
    diff = float(np.mean([per[t]["diff"] for t in tasks]))
    return {"macro_diff_pp": diff, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "macro_a": float(np.mean([per[t]["a"] for t in tasks])),
            "macro_b": float(np.mean([per[t]["b"] for t in tasks])), "per_task": per}


def by_category(res: dict, cats: dict[str, str]) -> dict:
    out = {}
    for t, v in res["per_task"].items():
        out.setdefault(cats[t], []).append(v["diff"])
    return {c: float(np.mean(v)) for c, v in out.items()}
