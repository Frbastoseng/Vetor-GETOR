"""v0.2 / etapa 6: latencia e velocidade REAIS (execucao condicional, sem calcular-e-zerar).

Mede separadamente: custo de cada compensador (us/chamada), overhead do roteador (us/decisao),
decode tok/s de FULL, STATIC-GREEDY com cada compensador e GUIDED-T / GUIDED-DL reais (k alvo).
Prompts de VALIDACAO (o conjunto final nao e usado para tempo de desenvolvimento).
  python scripts/v2_06_benchmark.py --routers results/v0.2/router_LR32_r1_T.pt results/v0.2/router_LR32_r1_DL.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from vetor.model_adapter import ROOT, env_info, load_config, load_model, pick_device, seed_all, sync, write_json
from vetor.v2.data import load_split
from vetor.v2.decode2 import decode_real
from vetor.v2.engine import patch_v2, set_comps, zero_comps
from vetor.v2.search import make_windows

sys.path.insert(0, str(ROOT / "scripts"))
from v2_03_teacher import load_comps, load_router  # noqa: E402

OUT = ROOT / "results" / "v0.2"
P, NF = 64, 64


def us_per_call(fn, device, reps=300):
    for _ in range(20):
        fn()
    sync(device)
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(device)
    return (time.perf_counter() - t) / reps * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", nargs="*", default=[])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--split", default="val")
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args()
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    E = list(range(2, L - 2))
    orc = json.loads((OUT / "oracle_val.json").read_text())
    docs = load_split(args.split, allow_final=(args.split == "final"))
    prompts = [w[:P].view(1, -1).to(device) for w in make_windows(tok, docs, P, 1, 8, 31)]
    comps = {c: load_comps(c, L, d, device) for c in ("ZERO", "MEAN", "LR8", "LR16", "LR32")}

    rep = {"env": env_info(device), "k": args.k, "decode_forwards": NF - 1, "split": args.split,
           "comp_us": {}, "comp_flops": {}, "router": {}, "decode": {}}
    x = torch.randn(1, 1, d, device=device)
    rep["mlp_us"] = us_per_call(lambda: wraps[10].inner(x), device)
    rep["mlp_flops"] = 6 * d * model.config.intermediate_size
    for c, cs in comps.items():
        rep["comp_us"][c] = us_per_call(lambda: cs[10](x), device)
        rep["comp_flops"][c] = cs[10].flops()

    # politicas: (nome, compensador, skip-set, roteador)
    pols = [("FULL", "ZERO", set(), None)]
    for c in comps:
        sg = orc["comps"][c]["static_greedy_order"][: args.k]
        pols.append((f"STATIC-GREEDY-{c}@k{args.k}", c, set(sg), None))
    for path in args.routers:
        r, ck = load_router(path, device)
        r.configure(E, k_max=int(-(-1.5 * args.k // 1)), tau=ck["tau_by_k"][str(float(args.k))])
        nm = ("GUIDED-DL" if ck["lagged"] else "GUIDED-T") + f"-{ck['comp']}@k{args.k}"
        pols.append((nm, ck["comp"], set(), r))
        zz = torch.randn(1, 1, d, device=device)
        r.reset_state(device)
        rep["router"][nm] = {"us_per_decision_with_sync": us_per_call(lambda: (r.reset_state(device), r.decide_real(zz, 10)), device),
                             "flops_per_decision": 2 * d * 16 + 2 * L * 16 + 2 * 16,
                             "params": sum(p.numel() for p in r.parameters()), "syncs_per_token": len(E)}
    raw = {p[0]: [] for p in pols}
    for i in range(3):  # aquecimento
        for nm, c, sk, r in pols:
            set_comps(wraps, comps[c])
            decode_real(model, wraps, prompts[0], NF, skip=sk, router=r, timed=True)
    for rep_i in range(args.repeats):
        order = pols[rep_i % len(pols):] + pols[: rep_i % len(pols)]
        pr = prompts[rep_i % len(prompts)]
        for nm, c, sk, r in order:
            set_comps(wraps, comps[c])
            o = decode_real(model, wraps, pr, NF, skip=sk, router=r, timed=True)
            raw[nm].append({"decode_s": o["decode_s"], "prefill_s": o["prefill_s"],
                            "tok_s": o["forwards"] / o["decode_s"],
                            "skips_per_token": o["mlp_skipped"] / o["forwards"], "repeat": rep_i})
    full = np.array([x["tok_s"] for x in raw["FULL"]])
    for nm, rows in raw.items():
        tps = np.array([x["tok_s"] for x in rows])
        ratio = tps / full
        boot = np.median(np.random.default_rng(0).choice(ratio, (4000, len(ratio))), axis=1)
        ms = np.array([x["decode_s"] for x in rows]) / (NF - 1) * 1e3
        rep["decode"][nm] = {"tok_s_median": float(np.median(tps)), "ms_per_token_median": float(np.median(ms)),
                             "ms_per_token_p95": float(np.percentile(ms, 95)),
                             "speedup_median": float(np.median(ratio)),
                             "speedup_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                             "skips_per_token": float(np.mean([x["skips_per_token"] for x in rows]))}
        v = rep["decode"][nm]
        print(f"{nm:28s} {v['tok_s_median']:7.1f} tok/s  speedup {v['speedup_median']:.3f} "
              f"[{v['speedup_ci95'][0]:.3f},{v['speedup_ci95'][1]:.3f}]  omissoes/token {v['skips_per_token']:.2f}")
    print("us/chamada:", {k: round(v, 1) for k, v in rep["comp_us"].items()}, "MLP:", round(rep["mlp_us"], 1))
    print("roteador:", rep["router"])
    rep["raw"] = raw
    write_json(OUT / f"benchmark_{args.tag}.json", rep)


if __name__ == "__main__":
    main()
