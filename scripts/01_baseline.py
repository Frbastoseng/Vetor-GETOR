"""Etapa 1: baseline FULL reproduzivel — equivalencia com generate, CPU x MPS, tempos.

Usa janelas do split de VALIDACAO (o teste fica intocado ate a avaliacao final).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from vetor.benchmark import run_benchmark
from vetor.collect import eval_windows, load_articles, tokenize_docs
from vetor.decode import run_decode
from vetor.model_adapter import (ROOT, env_info, load_config, load_model, model_info, new_run_id,
                                 pick_device, seed_all, sha256_json, sync, write_json)
from vetor.policies import FullPolicy
from vetor.wrappers import patch_model, unpatch_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--n-new", type=int, default=32)
    args = ap.parse_args()
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    run_id = new_run_id("baseline")
    model, tok = load_model(cfg, device)
    L = model.config.num_hidden_layers

    docs = tokenize_docs(tok, load_articles(cfg, "val"), cfg["data"]["min_doc_tokens"])
    wins = eval_windows(docs, cfg["data"]["eval"]["prompt_len"], args.n_new, args.n_prompts, cfg["seed"])
    prompts = [torch.tensor([w["prompt"]], device=device) for w in wins]
    report = {"run_id": run_id, "env": env_info(device), "model": model_info(model),
              "revision": cfg["model"]["revision"], "seed": cfg["seed"],
              "data_hash": sha256_json([(w["doc_id"], w["start"]) for w in wins]),
              "n_prompts": len(prompts), "n_new": args.n_new}

    # (a) referencia sem patch: generate greedy + logits do prefill
    ref_tokens, ref_logits = [], []
    with torch.inference_mode():
        for p in prompts:
            g = model.generate(p, max_new_tokens=args.n_new, do_sample=False, min_new_tokens=args.n_new)
            ref_tokens.append(g[0, p.shape[1]:].tolist())
            ref_logits.append(model(input_ids=p).logits[0, -1].float().cpu())

    wraps = patch_model(model)
    full = FullPolicy(L)
    ours, max_dlogit = [], 0.0
    with torch.inference_mode():
        for p, rl in zip(prompts, ref_logits):
            r = run_decode(model, wraps, p, args.n_new, full)
            ours.append(r["tokens"].tolist())
            lg = model(input_ids=p).logits[0, -1].float().cpu()
            max_dlogit = max(max_dlogit, float((lg - rl).abs().max()))
    match = [a == b for a, b in zip(ours, ref_tokens)]
    report["equivalence_vs_generate"] = {"all_sequences_identical": all(match),
                                         "n_identical": int(sum(match)),
                                         "max_abs_prefill_logit_diff_patched_vs_unpatched": max_dlogit}

    # (c) determinismo MPS: repetir
    again = [run_decode(model, wraps, p, args.n_new, full)["tokens"].tolist() for p in prompts]
    report["mps_repeat_identical"] = again == ours

    # (c) CPU x MPS (modelo sem patch na CPU)
    cpu_model, _ = load_model(cfg, "cpu")
    agree, dl = [], []
    with torch.inference_mode():
        for p, toks in zip(prompts[:6], ours[:6]):
            pc = p.cpu()
            g = cpu_model.generate(pc, max_new_tokens=args.n_new, do_sample=False, min_new_tokens=args.n_new)
            ct = g[0, pc.shape[1]:].tolist()
            agree.append(float(np.mean([a == b for a, b in zip(ct, toks)])))
            lc = cpu_model(input_ids=pc).logits[0, -1].float()
            lm = model(input_ids=p).logits[0, -1].float().cpu()
            dl.append(float((lc - lm).abs().max()))
    del cpu_model
    report["cpu_vs_mps"] = {"token_agreement_mean": float(np.mean(agree)),
                            "max_abs_prefill_logit_diff": float(np.max(dl))}

    # (d) tempos FULL
    bcfg = cfg["benchmark"]
    bench = run_benchmark(model, wraps, prompts, [full], device, bcfg["decode_tokens"],
                          bcfg["warmup"], bcfg["repeats"])
    report["timing_full"] = bench["summary"]["FULL"]
    report["memory"] = bench["memory"]
    nothrottle = run_benchmark(model, wraps, prompts, [full], device, bcfg["decode_tokens"],
                               bcfg["warmup"], bcfg["repeats"], throttle_lag=0)
    report["timing_full_without_throttle"] = nothrottle["summary"]["FULL"]
    # generate() como referencia externa de velocidade
    ts = []
    with torch.inference_mode():
        for r in range(bcfg["repeats"]):
            p = prompts[r % len(prompts)]
            sync(device); t = time.perf_counter()
            model.generate(p, max_new_tokens=bcfg["decode_tokens"] + 1, min_new_tokens=bcfg["decode_tokens"] + 1,
                           do_sample=False)
            sync(device); ts.append(time.perf_counter() - t)
    report["timing_hf_generate_total_ms_median"] = float(np.median(ts) * 1e3)
    unpatch_model(model)

    path = write_json(ROOT / "results" / f"{run_id}.json", report)
    import json
    print(json.dumps({k: v for k, v in report.items() if k not in ("env", "model")}, indent=2))
    print("Salvo:", path)


if __name__ == "__main__":
    main()
