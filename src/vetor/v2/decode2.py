"""Decode com execucao condicional REAL (v0.2): mascara estatica com compensador, ou GUIDED-T real.

Usado para (a) medir velocidade e (b) provar a equivalencia com o modo mascarado em lote.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from ..model_adapter import Throttle, sync
from .engine import V2MLP


def _set(wraps: list[V2MLP], skip: set[int], router=None):
    for w in wraps:
        w.active = w.layer_idx not in skip
        w.rt_decider = router.decide_real if router is not None else None
        w.calls_inner = w.calls_skipped = 0


def _clear(wraps):
    for w in wraps:
        w.active, w.rt_decider = True, None


@torch.inference_mode()
def decode_real(model, wraps, prompt: torch.Tensor, n: int, *, skip: set[int] = frozenset(), router=None,
                forced: torch.Tensor | None = None, collect_logp: bool = False, timed: bool = False) -> dict:
    """prompt [1,P]. Prefill completo; depois n-1 forwards de decode com omissoes reais."""
    device = prompt.device.type
    _clear(wraps)
    lag = router is not None and router.lagged
    if router is not None:
        router.reset_state(prompt.device)
    for w in wraps:
        w.capture = lag
    sync(device) if timed else None
    t0 = time.perf_counter()
    out = model(input_ids=prompt, use_cache=True)
    if lag:  # GUIDED-DL: o 1o token de decode usa z[l, ultimo token do prompt]
        router._zprev = {w.layer_idx: w.captured for w in wraps}
        for w in wraps:
            w.capture, w.captured = False, None
    sync(device) if timed else None
    t1 = time.perf_counter()
    cache, logits = out.cache, out.logits[:, -1, :]
    _set(wraps, set(skip), router)
    th = Throttle(device)
    toks, lps = [], []
    for i in range(n):
        if collect_logp:
            lps.append(F.log_softmax(logits.float(), -1)[0])
        nxt = forced[i].view(1, 1) if forced is not None else logits.argmax(-1, keepdim=True)
        toks.append(nxt.view(-1))
        if i == n - 1:
            break
        out = model(input_ids=nxt, cache=cache, use_cache=True)
        cache, logits = out.cache, out.logits[:, -1, :]
        th.step()
    sync(device) if timed else None
    t2 = time.perf_counter()
    inner = sum(w.calls_inner for w in wraps)
    skipped = sum(w.calls_skipped for w in wraps)
    _clear(wraps)
    return {"tokens": torch.cat(toks), "logp": torch.stack(lps) if collect_logp else None,
            "prefill_s": t1 - t0, "decode_s": t2 - t1, "forwards": n - 1,
            "mlp_skipped": skipped, "mlp_inner_decode": inner}
