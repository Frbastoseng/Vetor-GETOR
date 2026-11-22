"""Etapas 6-7 (qualidade): FULL, RANDOM, STATIC, ORACLE-DIAG, GUIDED-S, GUIDED-D{1,4,8,16}.

Teacher forcing autoregressivo com KV e geracao greedy, omissoes CONJUNTAS reais no decode.
  python scripts/04_evaluate.py --split val    # checagem (decisoes ja congeladas)
  python scripts/04_evaluate.py --split test   # avaliacao final, rodar UMA vez
"""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from vetor.collect import eval_windows, load_articles, tokenize_docs
from vetor.evaluate import evaluate, summarize
from vetor.guide import VectorGuide, budget_to_k, eligible_layers
from vetor.model_adapter import (ROOT, env_info, load_config, load_model, new_run_id, pick_device,
                                 seed_all, sha256_json, write_json)
from vetor.policies import build_policies
from vetor.wrappers import patch_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--guide", default=str(ROOT / "results" / "guide.pt"))
    ap.add_argument("--no-greedy", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    run_id = new_run_id(f"eval_{args.split}")
    model, tok = load_model(cfg, device)
    wraps = patch_model(model)
    L = len(wraps)
    rc, ec = cfg["routing"], cfg["data"]["eval"]
    n_el = len(eligible_layers(L, rc["protected_first"], rc["protected_last"]))
    ks = [budget_to_k(b, n_el) for b in rc["budgets"]]

    ck = torch.load(args.guide, weights_only=False)
    guide = VectorGuide(model.config.hidden_size, L, ck["dim"])
    guide.load_state_dict(ck["state_dict"])
    guide = guide.to(device).eval()
    pols = build_policies(L, ks, kinds=["RANDOM", "STATIC", "ORACLE", "GUIDED-S", "GUIDED-D"],
                          static_rank=ck["static_rank"], guide=guide, Ks=(1, 4, 8, 16), seed=cfg["seed"],
                          protected_first=rc["protected_first"], protected_last=rc["protected_last"],
                          pool_extra=ck.get("pool_extra"))
    pols = pols[1:]  # FULL e a referencia interna do avaliador

    docs = tokenize_docs(tok, load_articles(cfg, args.split), cfg["data"]["min_doc_tokens"])
    wins = eval_windows(docs, ec["prompt_len"], ec["continuation_len"], args.max_docs or ec["max_docs"],
                        cfg["seed"] + 7)
    print(f"{len(wins)} janelas ({args.split}), {len(pols)} politicas, ks={ks}")
    df, klt = evaluate(model, wraps, wins, pols, device, greedy=not args.no_greedy, oracle=True,
                       progress=lambda x: tqdm(x, desc="amostras"))
    summ = summarize(df, klt)
    df.to_csv(ROOT / "results" / f"{run_id}_samples.csv", index=False)
    summ.to_csv(ROOT / "results" / f"{run_id}_summary.csv", index=False)
    write_json(ROOT / "results" / f"{run_id}.json",
               {"run_id": run_id, "split": args.split, "env": env_info(device),
                "revision": cfg["model"]["revision"], "guide": {k: v for k, v in ck.items() if k != "state_dict"},
                "ks": ks, "n_windows": len(wins), "prompt_len": ec["prompt_len"],
                "continuation_len": ec["continuation_len"],
                "data_hash": sha256_json([(w["doc_id"], w["start"]) for w in wins]),
                "summary": summ.to_dict(orient="records")})
    cols = ["policy", "k", "skip_rate_all", "rel_delta_nll_pct", "rel_delta_nll_ci95_lo",
            "rel_delta_nll_ci95_hi", "kl_mean", "kl_p90", "top1_agree", "greedy_match"]
    cols = [c for c in cols if c in summ]
    print(summ[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Salvo:", run_id)


if __name__ == "__main__":
    main()
