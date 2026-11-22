"""Medicao de tempo: prefill e decode separados, sincronizacao apenas nas bordas.

Para cada repeticao: sync -> prefill -> sync (t_prefill) -> decisao da politica + decode
de N forwards -> sync (t_decode). O custo do guia (inclusive sua sincronizacao) entra no
tempo de decode. Ordem das politicas rotacionada a cada repeticao.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .model_adapter import Throttle, sync
from .policies import Policy
from .wrappers import RoutedMLP, apply_mask, counters, reset, reset_counters, set_capture, stack_captured


@torch.inference_mode()
def timed_decode(model, wraps: list[RoutedMLP], prompt_ids: torch.Tensor, n_forwards: int,
                 policy: Policy, device: str, ctx: dict | None = None, throttle_lag: int = 2) -> dict:
    reset(wraps)
    reset_counters(wraps)
    set_capture(wraps, True)
    sync(device)
    t0 = time.perf_counter()
    out = model(input_ids=prompt_ids, use_cache=True)
    sync(device)
    t1 = time.perf_counter()
    cache = out.cache
    active = policy.start(stack_captured(wraps), ctx or {})
    set_capture(wraps, policy.capture_decode)
    apply_mask(wraps, active)
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    prefill_inner, _ = counters(wraps)
    throttle = Throttle(device, throttle_lag)
    for i in range(n_forwards):
        new = policy.update(i, wraps)
        if new is not None:
            apply_mask(wraps, new)
        out = model(input_ids=nxt, cache=cache, use_cache=True)
        cache = out.cache
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        throttle.step()
    sync(device)
    t2 = time.perf_counter()
    inner, skipped = counters(wraps)
    reset(wraps)
    dec_inner = inner - prefill_inner
    return {
        "prefill_s": t1 - t0,
        "decode_s": t2 - t1,
        "decode_tok_s": n_forwards / (t2 - t1),
        "mlp_skipped": skipped,
        "mlp_executed_decode": dec_inner,
        "skip_rate": skipped / max(1, skipped + dec_inner),
    }


def memory_bytes(device: str) -> dict:
    if device == "mps":
        return {"mps_current_allocated": torch.mps.current_allocated_memory(),
                "mps_driver_allocated": torch.mps.driver_allocated_memory()}
    return {}


def run_benchmark(model, wraps, prompts: list[torch.Tensor], policies: list[Policy], device: str,
                  n_forwards: int, warmup: int, repeats: int, ctx_fn=None, throttle_lag: int = 2) -> dict:
    """prompts: lista de [1,T]. Cada repeticao usa um prompt (ciclico) e todas as politicas."""
    for _ in range(warmup):
        for p in policies:
            timed_decode(model, wraps, prompts[0], n_forwards, p, device,
                         ctx_fn(p, 0) if ctx_fn else None, throttle_lag)
    raw = {p.name: [] for p in policies}
    for r in range(repeats):
        order = policies[r % len(policies):] + policies[: r % len(policies)]
        pi = r % len(prompts)
        for p in order:
            res = timed_decode(model, wraps, prompts[pi], n_forwards, p, device,
                               ctx_fn(p, pi) if ctx_fn else None, throttle_lag)
            res["repeat"] = r
            res["prompt_idx"] = pi
            raw[p.name].append(res)
    summary = {name: summarize(rows) for name, rows in raw.items()}
    return {"summary": summary, "raw": raw, "memory": memory_bytes(device)}


def summarize(rows: list[dict]) -> dict:
    tps = np.array([r["decode_tok_s"] for r in rows])
    dec_ms = np.array([r["decode_s"] for r in rows]) * 1e3
    pre_ms = np.array([r["prefill_s"] for r in rows]) * 1e3
    return {
        "n": len(rows),
        "decode_tok_s_median": float(np.median(tps)),
        "decode_tok_s_p05": float(np.percentile(tps, 5)),
        "decode_ms_median": float(np.median(dec_ms)),
        "decode_ms_p95": float(np.percentile(dec_ms, 95)),
        "prefill_ms_median": float(np.median(pre_ms)),
        "prefill_ms_p95": float(np.percentile(pre_ms, 95)),
        "skip_rate": float(np.mean([r["skip_rate"] for r in rows])),
    }


def paired_speedup(raw_a: list[dict], raw_b: list[dict], n_boot: int = 5000, seed: int = 0) -> dict:
    """Razao tok/s B/A pareada por repeticao, com IC bootstrap 95% da mediana."""
    a = np.array([r["decode_tok_s"] for r in raw_a])
    b = np.array([r["decode_tok_s"] for r in raw_b])
    ratio = b / a
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(ratio, size=(n_boot, len(ratio)), replace=True), axis=1)
    return {"median_ratio": float(np.median(ratio)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "frac_faster": float(np.mean(ratio > 1))}
