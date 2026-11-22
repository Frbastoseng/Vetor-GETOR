"""Diagnostico M2: de onde vem o tempo do decode e qual o teto do skip de MLP.

Mede ms/token para mascaras estaticas (0, 26 elegiveis, 30 = diagnostico) e separa o tempo de
CPU para *enfileirar* (sem esperar a GPU) do tempo total sincronizado. Tambem cronometra um
MLP e uma atencao isolados por camada (com sincronizacao, so para diagnostico).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from vetor.model_adapter import ROOT, Throttle, load_config, load_model, pick_device, sync, write_json
from vetor.wrappers import apply_mask, patch_model, reset


@torch.inference_mode()
def decode_ms(model, wraps, prompt, active, device, n=64, throttle=True, reps=7):
    out_tot, out_cpu = [], []
    for r in range(reps + 2):
        reset(wraps)
        o = model(input_ids=prompt, use_cache=True)
        c, nxt = o.cache, o.logits[:, -1].argmax(-1, keepdim=True)
        apply_mask(wraps, active)
        th = Throttle(device, 2 if throttle else 0)
        sync(device)
        t0 = time.perf_counter()
        cpu = 0.0
        for _ in range(n):
            a = time.perf_counter()
            o = model(input_ids=nxt, cache=c, use_cache=True)
            c, nxt = o.cache, o.logits[:, -1].argmax(-1, keepdim=True)
            cpu += time.perf_counter() - a
            th.step()
        sync(device)
        if r >= 2:
            out_tot.append((time.perf_counter() - t0) / n * 1e3)
            out_cpu.append(cpu / n * 1e3)
    reset(wraps)
    return float(np.median(out_tot)), float(np.median(out_cpu))


@torch.inference_mode()
def module_us(mod, x, device, reps=200):
    for _ in range(10):
        mod(x)
    sync(device)
    t = time.perf_counter()
    for _ in range(reps):
        mod(x)
    sync(device)
    return (time.perf_counter() - t) / reps * 1e6


def main():
    cfg = load_config()
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    wraps = patch_model(model)
    L = len(wraps)
    prompt = torch.randint(100, 20000, (1, 64), device=device)
    masks = {"full": [True] * L,
             "skip_26_eligible": [l < 2 or l >= L - 2 for l in range(L)],
             "skip_all_30_diag": [False] * L}
    res = {}
    for name, m in masks.items():
        tot, cpu = decode_ms(model, wraps, prompt, m, device)
        tot_nt, cpu_nt = decode_ms(model, wraps, prompt, m, device, throttle=False)
        res[name] = {"ms_per_token": tot, "cpu_enqueue_ms_per_token": cpu,
                     "ms_per_token_no_throttle": tot_nt, "cpu_enqueue_ms_no_throttle": cpu_nt}
        print(name, res[name])
    x = torch.randn(1, 1, model.config.hidden_size, device=device)
    res["isolated_mlp_us_per_call"] = module_us(wraps[10].inner, x, device)
    res["isolated_rmsnorm_us_per_call"] = module_us(model.model.layers[10].post_attention_layernorm, x, device)
    res["mlp_saving_us_per_layer_measured"] = (res["full"]["ms_per_token"] - res["skip_26_eligible"]["ms_per_token"]) / 26 * 1e3
    print({k: v for k, v in res.items() if not isinstance(v, dict)})
    write_json(ROOT / "results" / "diag_dispatch.json", res)


if __name__ == "__main__":
    main()
