"""v0.3 / Fase 5: desempenho real (decode tok/s, TTFT, memoria) de FULL e das execucoes treinadas.

Mesmo dispositivo, precisao (float32) e backend; throttle de eventos (como v0.1-v0.2); ordem rotacionada
por repeticao; razoes pareadas contra FULL com IC95 bootstrap. LoRA medido JA FUNDIDO.
  python scripts/v3_05_perf.py --tags A0_r8_task_g1 A_r8_task_g1 LORA_r1_task_g1_lr1e3 --repeats 20
Saida: results/v0.3/perf.json
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from vetor.model_adapter import ROOT, Throttle, env_info, load_config, load_model, pick_device, seed_all, sync, write_json
from vetor.v3.modules import detach_all

sys.path.insert(0, str(ROOT / "scripts"))
from v3_04_eval_run import load_run  # noqa: E402

P, N = 64, 64


@torch.inference_mode()
def timed(model, prompt, device):
    sync(device)
    t0 = time.perf_counter()
    o = model(input_ids=prompt, use_cache=True)
    nxt = o.logits[:, -1].argmax(-1, keepdim=True)
    sync(device)
    t1 = time.perf_counter()  # TTFT = prefill + 1o token escolhido
    c, th = o.cache, Throttle(device)
    for _ in range(N - 1):
        o = model(input_ids=nxt, cache=c, use_cache=True)
        c, nxt = o.cache, o.logits[:, -1].argmax(-1, keepdim=True)
        th.step()
    sync(device)
    t2 = time.perf_counter()
    return t1 - t0, (N - 1) / (t2 - t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    prompts = [torch.randint(100, 20000, (1, P), generator=torch.Generator().manual_seed(i)).to(device) for i in range(4)]
    confs = ["FULL"] + args.tags
    mem = {}

    def setup(tag):
        detach_all(model)
        if tag != "FULL":
            load_run(model, tag)
        model.eval()

    for tag in confs:  # aquecimento + memoria
        setup(tag)
        for _ in range(3):
            timed(model, prompts[0], device)
        if device == "mps":
            mem[tag] = torch.mps.driver_allocated_memory()
    raw = {t: [] for t in confs}
    for r in range(args.repeats):
        for tag in confs[r % len(confs):] + confs[: r % len(confs)]:
            setup(tag)
            timed(model, prompts[r % 4], device)  # 1 passada de aquecimento apos trocar de configuracao
            ttft, tps = timed(model, prompts[r % 4], device)
            raw[tag].append({"ttft_s": ttft, "tok_s": tps, "repeat": r})
    detach_all(model)
    full_t = np.array([x["tok_s"] for x in raw["FULL"]])
    full_f = np.array([x["ttft_s"] for x in raw["FULL"]])
    out = {"env": env_info(device), "prompt_len": P, "decode_tokens": N - 1, "repeats": args.repeats, "configs": {}}
    rng = np.random.default_rng(0)
    for tag in confs:
        t = np.array([x["tok_s"] for x in raw[tag]])
        f = np.array([x["ttft_s"] for x in raw[tag]])
        ratio = t / full_t
        boot = np.median(rng.choice(ratio, (5000, len(ratio))), axis=1)
        out["configs"][tag] = {
            "decode_tok_s_median": float(np.median(t)), "throughput_ratio_median": float(np.median(ratio)),
            "throughput_ratio_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "ttft_ms_median": float(np.median(f) * 1e3), "ttft_ratio_median": float(np.median(f / full_f)),
            "mps_driver_bytes": mem.get(tag), "mem_ratio": (mem.get(tag) / mem["FULL"]) if mem.get("FULL") else None}
        v = out["configs"][tag]
        print(f"{tag:28s} {v['decode_tok_s_median']:6.1f} tok/s  throughput {100 * v['throughput_ratio_median']:5.1f}% "
              f"[{100 * v['throughput_ratio_ci95'][0]:.1f}, {100 * v['throughput_ratio_ci95'][1]:.1f}]  "
              f"TTFT {v['ttft_ms_median']:.1f} ms ({100 * v['ttft_ratio_median']:.0f}%)", flush=True)
    out["raw"] = raw
    write_json(ROOT / "results" / "v0.3" / "perf.json", out)


if __name__ == "__main__":
    main()
