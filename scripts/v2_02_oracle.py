"""v0.2 / etapa 2: oraculo guloso por sequencia, STATIC-GREEDY e controles, para cada compensador.

STATIC-GREEDY: ajustado em janelas de TREINO. Avaliacao: janelas de VALIDACAO. Oraculo: por janela de val.
Saida: results/v0.2/oracle_val.json
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from vetor.model_adapter import ROOT, load_config, load_model, pick_device, seed_all, write_json
from vetor.v2.compensate import make_comps
from vetor.v2.data import load_split
from vetor.v2.engine import patch_v2, set_comps, zero_comps
from vetor.v2.search import eval_batch, greedy_sets, make_windows, masks_from_sets, summarize

OUT = ROOT / "results" / "v0.2"
KS = (3, 5, 6, 8, 10)
P, N = 64, 64


def comp_sets(L, d, device):
    ck = torch.load(OUT / "compensators.pt", weights_only=False)
    out = {"ZERO": zero_comps(L, d, device), "MEAN": make_comps("mean", ck["mean"], L, d, 0, device)}
    for r in (8, 16, 32):
        out[f"LR{r}"] = make_comps("lr", ck["lr"][r], L, d, r, device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=96)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-oracle", type=int, default=48)
    args = ap.parse_args()
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    E = list(range(2, L - 2))
    tr = make_windows(tok, load_split("train"), P, N, args.n_train, 11)
    va = make_windows(tok, load_split("val"), P, N, args.n_val, 12)
    T = P + N - 1
    v01_rank = [l for l in torch.load(ROOT / "results" / "guide.pt", weights_only=False)["static_rank"] if l in E]
    rng = np.random.default_rng(0)
    rand_sets = {k: [sorted(rng.choice(E, k, replace=False).tolist()) for _ in va] for k in KS}
    rep = {"P": P, "N": N, "n_train_windows": len(tr), "n_val_windows": len(va), "comps": {}}
    for cname, comps in comp_sets(L, d, device).items():
        set_comps(wraps, comps)
        t0 = time.time()
        sg, traj = greedy_sets(model, wraps, tr, P, device, E, max(KS))
        res = {"static_greedy_order": sg, "static_greedy_train_kl": traj, "policies": {}}
        for k in KS:
            pols = {"STATIC-GREEDY": [sg[:k]] * len(va), "STATIC": [v01_rank[:k]] * len(va), "RANDOM": rand_sets[k]}
            for pname, sets in pols.items():
                r = eval_batch(model, wraps, va, P, device, masks=masks_from_sets(sets, T, P, L))
                res["policies"][f"{pname}@k{k}"] = summarize(r)
        # oraculo por janela (diagnostico, usa a propria continuacao)
        orc = {k: [] for k in KS}
        for w in va[: args.n_oracle]:
            o, _ = greedy_sets(model, wraps, [w], P, device, E, max(KS))
            for k in KS:
                orc[k].append(o[:k])
        for k in KS:
            r = eval_batch(model, wraps, va[: args.n_oracle], P, device, masks=masks_from_sets(orc[k], T, P, L))
            res["policies"][f"ORACLE-GREEDY-SEQ@k{k}"] = summarize(r)
            # controle no MESMO subconjunto do oraculo
            r = eval_batch(model, wraps, va[: args.n_oracle], P, device,
                           masks=masks_from_sets([sg[:k]] * args.n_oracle, T, P, L))
            res["policies"][f"STATIC-GREEDY(oracle-subset)@k{k}"] = summarize(r)
        rep["comps"][cname] = res
        print(f"\n== {cname} ({time.time() - t0:.0f}s)  STATIC-GREEDY ordem: {sg}")
        for k in KS:
            row = "  ".join(f"{p}={res['policies'][f'{p}@k{k}']['rel_dnll_pct']:6.2f}%"
                            for p in ("STATIC-GREEDY", "STATIC", "RANDOM", "ORACLE-GREEDY-SEQ"))
            print(f"  k={k:2d}  {row}")
        write_json(OUT / "oracle_val.json", rep)


if __name__ == "__main__":
    main()
