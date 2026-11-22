"""Janelas, avaliacao em lote de politicas e busca gulosa de conjuntos (oraculo e STATIC-GREEDY)."""
from __future__ import annotations

import numpy as np
import torch

from .engine import Ctrl
from .fasteval import kl_rows, logprobs_D, nll_rows


def make_windows(tok, docs: list[dict], P: int, n: int, max_docs: int, seed: int, per_doc: int = 1) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    out = []
    order = rng.permutation(len(docs))
    for i in order:
        ids = tok(docs[int(i)]["text"], add_special_tokens=False).input_ids
        if len(ids) < P + n:
            continue
        for _ in range(per_doc):
            s = int(rng.integers(0, len(ids) - (P + n) + 1))
            out.append(torch.tensor(ids[s:s + P + n]))
        if len(out) >= max_docs * per_doc:
            break
    return out


@torch.inference_mode()
def eval_batch(model, wraps, windows: list[torch.Tensor], P: int, device, *, masks=None, router=None,
               eligible=None, chunk: int = 32) -> dict:
    """Avalia uma politica (mascaras [W,T,L] ou roteador) em W janelas. Metricas por janela."""
    L = len(wraps)
    res = {"nll": [], "nll_full": [], "kl": [], "top1": [], "skips": []}
    for i in range(0, len(windows), chunk):
        win = torch.stack(windows[i:i + chunk]).to(device)
        x, tgt = win[:, :-1], win[:, P + 1:]
        lp_f = logprobs_D(model, wraps, x, P, None)
        if masks is not None:
            ctrl = Ctrl(start=P, L=L, static_mask=masks[i:i + chunk].to(device))
        elif router is not None:
            ctrl = Ctrl(start=P, L=L, router=router, eligible=eligible)
        else:
            ctrl = None
        lp = logprobs_D(model, wraps, x, P, ctrl) if ctrl is not None else lp_f
        g = lambda a: a.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        res["nll"].append((-g(lp)).sum(1).cpu())
        res["nll_full"].append((-g(lp_f)).sum(1).cpu())
        res["kl"].append(kl_rows(lp_f, lp).cpu())
        res["top1"].append((lp.argmax(-1) == lp_f.argmax(-1)).float().mean(1).cpu())
        sk = ctrl.bits[:, P:].sum(-1).float() if ctrl is not None and ctrl.bits is not None else torch.zeros(win.shape[0], x.shape[1] - P)
        res["skips"].append(sk.cpu())
    return {k: torch.cat(v) for k, v in res.items()}


def summarize(r: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    nll, nf = r["nll"].numpy(), r["nll_full"].numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(nll), (n_boot, len(nll)))
    boot = nll[idx].sum(1) / nf[idx].sum(1) - 1
    kl = r["kl"].numpy().ravel()
    sk = r["skips"].numpy()
    return {"rel_dnll_pct": float(100 * (nll.sum() / nf.sum() - 1)),
            "ci95": [float(100 * np.percentile(boot, 2.5)), float(100 * np.percentile(boot, 97.5))],
            "kl_mean": float(kl.mean()), "kl_p90": float(np.percentile(kl, 90)),
            "top1": float(r["top1"].mean()), "skips_per_token": float(sk.mean()),
            "skips_per_token_max": float(sk.max()), "n_windows": int(len(nll))}


def masks_from_sets(sets: list[list[int]], T: int, P: int, L: int) -> torch.Tensor:
    m = torch.zeros(len(sets), T, L, dtype=torch.bool)
    for i, s in enumerate(sets):
        if s:
            m[i, P:, s] = True
    return m


@torch.inference_mode()
def greedy_sets(model, wraps, windows: list[torch.Tensor], P: int, device, candidates: list[int], k_max: int,
                chunk_rows: int = 64) -> tuple[list[int], list[float]]:
    """Busca gulosa: adiciona a camada que minimiza o KL medio (por posicao) somado sobre as janelas.
    Com 1 janela = ORACLE-GREEDY-SEQ (diagnostico); com janelas de treino = STATIC-GREEDY."""
    L = len(wraps)
    T = windows[0].shape[0] - 1
    fulls = []
    for w in windows:
        x = w.to(device)[:-1].view(1, -1)
        fulls.append(logprobs_D(model, wraps, x, P, None))
    chosen, traj = [], []
    for _ in range(k_max):
        rest = [c for c in candidates if c not in chosen]
        score = torch.zeros(len(rest))
        masks = masks_from_sets([chosen + [c] for c in rest], T, P, L).to(device)
        for w, lpf in zip(windows, fulls):
            x = w.to(device)[:-1].view(1, -1)
            for j in range(0, len(rest), chunk_rows):
                m = masks[j:j + chunk_rows]
                lp = logprobs_D(model, wraps, x.expand(m.shape[0], -1), P, Ctrl(start=P, L=L, static_mask=m))
                score[j:j + m.shape[0]] += kl_rows(lpf, lp).mean(1).cpu()
        best = int(score.argmin())
        chosen.append(rest[best])
        traj.append(float(score[best] / len(windows)))
    return chosen, traj
