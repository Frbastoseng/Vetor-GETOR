"""v0.4 / Fase 2: GUIDED-T e GUIDED-DL no 1.7B (treino da v0.2), selecao na VALIDACAO, 3 sementes.

1) Grade da v0.2 (N in 8/16/32 x {mse, pinball q=0.3}) com semente 0; escolha pelo dNLL real em k=5 (val/eval).
2) Melhor configuracao treinada com sementes 0,1,2. Para cada semente e cada k: tau por regime
   - tau_tf : texto natural (64 janelas val/calib)
   - tau_gen: continuacoes greedy do FULL a partir dos mesmos prompts (politica de orcamento, plano sec. 3)
  python scripts/v4_14_router.py --rounds 0          (depois --rounds 0 1, --rounds 0 1 2)
Saida: results/v0.4/routers/{T|DL}_r{max}_s{seed}.pt, results/v0.4/router_r{max}.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import torch

from vetor.model_adapter import ROOT, pick_device, seed_all, write_json
from vetor.v2.engine import patch_v2, set_comps, zero_comps
from vetor.v2.search import summarize
from vetor.v2.teacher2 import concat
from vetor.v2.train2 import calibrate_tau, eval_router, train_router
from vetor.v4.common import OUT, eligible, load
from vetor.v4.pipe import P, NW, load_comps, windows
from vetor.v4.routers import save_router4

KS = (3, 4, 5, 6, 8)
K = 5
SEEDS = (0, 1, 2)


@torch.inference_mode()
def full_greedy_windows(model, wins, device):
    """prompt (P) + continuacao greedy do FULL (NW tokens) — texto gerado pelo proprio modelo."""
    out = []
    for i in range(0, len(wins), 16):
        x = torch.stack([w[:P] for w in wins[i:i + 16]]).to(device)
        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=NW, min_new_tokens=NW,
                           do_sample=False, pad_token_id=2)
        out.extend(list(g.cpu()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", type=int, default=[0])
    ap.add_argument("--comp", default=None, help="padrao: escolhido na validacao; ZERO = controle sem compensacao")
    args = ap.parse_args()
    seed_all(1234)
    device = pick_device("auto")
    model, tok = load("1.7B", device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    E = eligible(L)
    comp = args.comp or json.loads((OUT / "static_1.7B.json").read_text())["chosen_comp"]
    sfx = "" if args.comp is None else f"_{args.comp}"
    wraps = patch_v2(model, zero_comps(L, d, device))
    set_comps(wraps, load_comps(comp, L, d, device))
    data = concat([{k: v for k, v in torch.load(ROOT / "data" / f"v4_teacher_r{r}{sfx}.pt", weights_only=False).items()
                    if torch.is_tensor(v)} for r in args.rounds])
    calib = windows(tok, "val", part="calib", seed=12)
    va = windows(tok, "val", part="eval", seed=12)
    calib_gen = full_greedy_windows(model, calib, device)
    rtag = max(args.rounds)
    (OUT / "routers").mkdir(parents=True, exist_ok=True)
    rep = {"comp": comp, "rounds": args.rounds, "n_cells": int(data["lid"].shape[0]), "search": [], "best": {}}
    for variant, lagged in (("T", False), ("DL", True)):
        best = None
        for N, (loss, q) in itertools.product((8, 16, 32), (("mse", 0.5), ("pinball", 0.3))):
            t0 = time.time()
            r = train_router(data, d=d, L=L, N=N, lagged=lagged, loss=loss, q=q, device=device, seed=0, n_el=len(E))
            tau, _ = calibrate_tau(model, wraps, r, calib, P, device, float(K), E)
            s = eval_router(model, wraps, r, va, P, device)
            rep["search"].append({"variant": variant, "N": N, "loss": loss, "q": q, "tau": tau, **s})
            print(f"{variant} N={N:2d} {loss:7s}: dNLL={s['rel_dnll_pct']:6.2f}% skips={s['skips_per_token']:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if best is None or s["rel_dnll_pct"] < best[0]:
                best = (s["rel_dnll_pct"], N, loss, q)
        _, N, loss, q = best
        rep["best"][variant] = {"N": N, "loss": loss, "q": q, "seeds": {}}
        for seed in SEEDS:
            r = train_router(data, d=d, L=L, N=N, lagged=lagged, loss=loss, q=q, device=device, seed=seed, n_el=len(E))
            tau_tf, tau_gen, curve = {}, {}, {}
            for k in (KS if seed == 0 else (4, K)):
                tau_tf[str(float(k))], _ = calibrate_tau(model, wraps, r, calib, P, device, float(k), E)
                tau_gen[str(float(k))], _ = calibrate_tau(model, wraps, r, calib_gen, P, device, float(k), E)
                r.configure(E, k_max=int(-(-1.5 * k // 1)), tau=tau_tf[str(float(k))])
                curve[str(k)] = eval_router(model, wraps, r, va, P, device)
            r.configure(E, k_max=int(-(-1.5 * K // 1)), tau=tau_tf[str(float(K))])
            path = OUT / "routers" / f"{variant}{sfx}_r{rtag}_s{seed}.pt"
            save_router4(path, r, N=N, loss=loss, q=q, comp=comp, rounds=args.rounds, seed=seed,
                         tau_tf=tau_tf, tau_gen=tau_gen, selected_on="val")
            rep["best"][variant]["seeds"][str(seed)] = {"path": str(path.relative_to(ROOT)), "tau_tf": tau_tf,
                                                        "tau_gen": tau_gen, "curve_val_tf": curve}
            print(f"== {variant} s{seed} N={N} {loss}: " + "  ".join(
                f"k{k}={v['rel_dnll_pct']:.2f}%({v['skips_per_token']:.2f})" for k, v in curve.items()), flush=True)
        write_json(OUT / f"router_r{rtag}{sfx}.json", rep)


if __name__ == "__main__":
    main()
