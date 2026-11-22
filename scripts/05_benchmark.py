"""Benchmark de tempo real (decode tok/s, prefill, p95, memoria) por politica e orcamento.

Prompts de TESTE (sem ajuste algum feito sobre eles). Ordem rotacionada por repeticao.
Speedup pareado por repeticao contra FULL com IC bootstrap 95%.

  python scripts/05_benchmark.py --kinds STATIC RANDOM          # gate M2 (sem guia)
  python scripts/05_benchmark.py --kinds STATIC RANDOM GUIDED-S GUIDED-D --guide results/guide.pt
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from vetor.benchmark import paired_speedup, run_benchmark
from vetor.collect import eval_windows, load_articles, tokenize_docs
from vetor.guide import budget_to_k, eligible_layers
from vetor.model_adapter import (ROOT, env_info, load_config, load_model, mlp_flops_per_token, new_run_id,
                                 pick_device, seed_all, write_json)
from vetor.policies import build_policies
from vetor.wrappers import patch_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["STATIC", "RANDOM"])
    ap.add_argument("--guide", default=None)
    ap.add_argument("--Ks", nargs="+", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--dtype", default=None, help="float32 | float16")
    ap.add_argument("--tag", default="bench")
    args = ap.parse_args()
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    dtype = getattr(torch, args.dtype) if args.dtype else None
    run_id = new_run_id(args.tag)
    model, tok = load_model(cfg, device, dtype)
    wraps = patch_model(model)
    L = len(wraps)
    rc, bc = cfg["routing"], cfg["benchmark"]
    n_el = len(eligible_layers(L, rc["protected_first"], rc["protected_last"]))
    ks = [budget_to_k(b, n_el) for b in rc["budgets"]]

    rank = torch.load(ROOT / "data" / "teacher_train.pt", weights_only=False)["kl"].mean(0).argsort().tolist()
    guide = None
    if args.guide:
        ck = torch.load(args.guide, weights_only=False)
        from vetor.guide import VectorGuide
        guide = VectorGuide(model.config.hidden_size, L, ck["dim"])
        guide.load_state_dict(ck["state_dict"])
        guide = guide.to(device).eval()
        rank = ck.get("static_rank", rank)
    pols = build_policies(L, ks, kinds=args.kinds, static_rank=rank, guide=guide, Ks=tuple(args.Ks),
                          seed=cfg["seed"], protected_first=rc["protected_first"],
                          protected_last=rc["protected_last"])
    names = [p.name + (f"@k{p.k}" if p.k else "") for p in pols]
    for p, n in zip(pols, names):
        p.name = n

    docs = tokenize_docs(tok, load_articles(cfg, "test"), cfg["data"]["min_doc_tokens"])
    wins = eval_windows(docs, bc["prompt_len"], 0, args.n_prompts, cfg["seed"] + 99)
    prompts = [torch.tensor([w["prompt"]], device=device) for w in wins]
    repeats = args.repeats or bc["repeats"]
    print(f"{len(pols)} politicas x {repeats} repeticoes, {bc['decode_tokens']} forwards de decode")
    res = run_benchmark(model, wraps, prompts, pols, device, bc["decode_tokens"], bc["warmup"], repeats)

    guide_overhead = None
    if guide is not None:  # custo isolado de UMA decisao: score [L] + top-k + leitura no host
        import time
        from vetor.guide import topk_mask
        from vetor.model_adapter import sync
        Z = torch.randn(L, model.config.hidden_size, device=device)
        ts = []
        for i in range(220):
            sync(device)
            t = time.perf_counter()
            topk_mask(guide.score_layers(Z), ks[-1], protected_first=rc["protected_first"],
                      protected_last=rc["protected_last"]).tolist()
            ts.append(time.perf_counter() - t)
        import numpy as np
        guide_overhead = {"us_per_decision_median": float(np.median(ts[20:]) * 1e6),
                          "us_per_decision_p95": float(np.percentile(ts[20:], 95) * 1e6),
                          "n_params": sum(p.numel() for p in guide.parameters()),
                          "flops_per_decision_approx": 2 * L * model.config.hidden_size * ck["dim"]}
        print("Overhead do guia:", guide_overhead)

    flops = mlp_flops_per_token(model)
    rows = []
    for p in pols:
        s = res["summary"][p.name]
        sp = paired_speedup(res["raw"]["FULL"], res["raw"][p.name])
        rows.append({"policy": p.name, "k": p.k, **s,
                     "theoretical_mlp_flops_saved_per_token": p.k * flops,
                     "theoretical_mlp_flops_saved_frac": p.k / L,
                     "speedup_vs_full_median": sp["median_ratio"], "speedup_ci95_lo": sp["ci95"][0],
                     "speedup_ci95_hi": sp["ci95"][1], "frac_repeats_faster": sp["frac_faster"]})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / f"{run_id}_bench.csv", index=False)
    write_json(ROOT / "results" / f"{run_id}.json",
               {"run_id": run_id, "env": env_info(device), "dtype": str(next(model.parameters()).dtype),
                "revision": cfg["model"]["revision"], "kinds": args.kinds, "guide": args.guide,
                "prompt_len": bc["prompt_len"], "decode_forwards": bc["decode_tokens"],
                "repeats": repeats, "warmup": bc["warmup"], "memory": res["memory"],
                "guide_overhead": guide_overhead,
                "summary": rows, "raw": res["raw"]})
    cols = ["policy", "skip_rate", "decode_tok_s_median", "decode_ms_p95", "prefill_ms_median",
            "speedup_vs_full_median", "speedup_ci95_lo", "speedup_ci95_hi"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Salvo:", run_id)


if __name__ == "__main__":
    main()
