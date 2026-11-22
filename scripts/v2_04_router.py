"""v0.2 / etapa 4: treina GUIDED-T (z atual) e GUIDED-DL (z do token anterior) com o professor novo.

Selecao SO na validacao, pela dNLL REAL (avaliacao em lote) na taxa-alvo k=6 (criterio).
Mesmo espaco de busca para T e DL (comparacao justa, Q4). Depois calibra tau para cada k.
  python scripts/v2_04_router.py --comp LR32 --rounds 0        (ou --rounds 0 1)
Saidas: results/v0.2/router_{comp}_r{rmax}_{T|DL}.pt, results/v0.2/router_{comp}_r{rmax}.json
"""
from __future__ import annotations

import argparse
import itertools
import time

import torch

from vetor.model_adapter import ROOT, load_config, load_model, pick_device, seed_all, write_json
from vetor.v2.data import load_split
from vetor.v2.engine import patch_v2, set_comps, zero_comps
from vetor.v2.search import make_windows
from vetor.v2.teacher2 import concat
from vetor.v2.train2 import calibrate_tau, eval_router, train_router

OUT = ROOT / "results" / "v0.2"
P, NW = 64, 64
KS = (3, 5, 6, 8, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--rounds", nargs="+", type=int, default=[0])
    ap.add_argument("--n-val", type=int, default=200)
    args = ap.parse_args()
    from v2_03_teacher import load_comps  # noqa
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    set_comps(wraps, load_comps(args.comp, L, d, device))
    E = list(range(2, L - 2))
    data = concat([{k: v for k, v in torch.load(ROOT / "data" / f"v2_teacher_{args.comp}_r{r}_train.pt",
                                                 weights_only=False).items() if torch.is_tensor(v)}
                   for r in args.rounds])
    va = make_windows(tok, load_split("val"), P, NW, args.n_val, 12)
    calib = va[:64]
    grid = list(itertools.product((8, 16, 32), (("mse", 0.5), ("pinball", 0.3))))
    rep = {"comp": args.comp, "rounds": args.rounds, "n_cells": int(data["lid"].shape[0]), "search": [], "best": {}}
    rtag = max(args.rounds)
    for variant, lagged in (("T", False), ("DL", True)):
        best = None
        for N, (loss, q) in grid:
            t0 = time.time()
            r = train_router(data, d=d, L=L, N=N, lagged=lagged, loss=loss, q=q, device=device)
            tau, rate = calibrate_tau(model, wraps, r, calib, P, device, 6.0, E)
            s = eval_router(model, wraps, r, va, P, device)
            row = {"variant": variant, "N": N, "loss": loss, "q": q, "tau": tau, **s}
            rep["search"].append(row)
            print(f"{variant} N={N:2d} {loss:7s}: dNLL={s['rel_dnll_pct']:6.2f}% [{s['ci95'][0]:.2f},{s['ci95'][1]:.2f}]"
                  f" skips/token={s['skips_per_token']:.2f} ({time.time() - t0:.0f}s)")
            if best is None or s["rel_dnll_pct"] < best[0]:
                best = (s["rel_dnll_pct"], r, row)
        _, r, row = best
        tau_by_k, curve = {}, {}
        for k in KS:
            tau, _ = calibrate_tau(model, wraps, r, calib, P, device, float(k), E)
            tau_by_k[str(float(k))] = tau
            curve[str(k)] = eval_router(model, wraps, r, va, P, device)
        rep["best"][variant] = {"config": row, "curve_val": curve}
        torch.save({"state_dict": r.state_dict(), "d": d, "L": L, "dim": 16, "lagged": lagged,
                    "k_max": r.k_max, "n_el": r.n_el, "y_mu": r.y_mu, "y_sd": r.y_sd, "N": row["N"],
                    "loss": row["loss"], "q": row["q"], "tau_by_k": tau_by_k, "comp": args.comp,
                    "rounds": args.rounds, "selected_on": "val"},
                   OUT / f"router_{args.comp}_r{rtag}_{variant}.pt")
        print(f"== melhor {variant}: N={row['N']} {row['loss']}  curva val: " +
              "  ".join(f"k{k}={curve[str(k)]['rel_dnll_pct']:.2f}%({curve[str(k)]['skips_per_token']:.2f})" for k in KS))
    write_json(OUT / f"router_{args.comp}_r{rtag}.json", rep)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
