"""v0.2 / etapa 1: ajusta compensadores MEAN e LR-r (r=8,16,32) em TREINO; escolhe alpha na VALIDACAO.

Saidas: results/v0.2/compensators.pt (ajustes), results/v0.2/compensators.json (R^2 por camada).
"""
from __future__ import annotations

import time

import torch

from vetor.model_adapter import ROOT, load_config, load_model, pick_device, seed_all, write_json
from vetor.v2.compensate import collect_stats, fit_layer, r2_layer, token_batches
from vetor.v2.data import load_split
from vetor.v2.engine import patch_v2, zero_comps

OUT = ROOT / "results" / "v0.2"
RANKS = (8, 16, 32)
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1)


def main():
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    ids = {s: [tok(x["text"], add_special_tokens=False).input_ids for x in load_split(s)] for s in ("train", "val")}
    t0 = time.time()
    acc_tr, n_tr = collect_stats(model, wraps, token_batches(ids["train"], 256, 16, 50, 1), device)
    acc_va, n_va = collect_stats(model, wraps, token_batches(ids["val"], 256, 16, 12, 2), device)
    print(f"tokens treino={n_tr} val={n_va} ({time.time() - t0:.0f}s)")

    rep = {"n_tokens_train": n_tr, "n_tokens_val": n_va, "alpha_search": {}, "r2_val": {}}
    # alpha escolhido na validacao (R^2 medio, r=32)
    best_a, best = None, -1e9
    for a in ALPHAS:
        r2 = sum(r2_layer(acc_va, l, fit_layer(acc_tr, l, 32, a)) for l in range(L)) / L
        rep["alpha_search"][str(a)] = r2
        print(f"alpha={a:g}: R2 val medio (r=32) = {r2:.4f}")
        if r2 > best:
            best_a, best = a, r2
    rep["alpha"] = best_a
    fits = {}
    mean_fits = [{"ybar": acc_tr.moments(l, acc_tr.n)[1]} for l in range(L)]
    rep["r2_val"]["ZERO"] = [r2_layer(acc_va, l, None) for l in range(L)]
    rep["r2_val"]["MEAN"] = [r2_layer(acc_va, l, mean_fits[l]) for l in range(L)]
    for r in RANKS:
        fits[r] = [fit_layer(acc_tr, l, r, best_a) for l in range(L)]
        rep["r2_val"][f"LR{r}"] = [r2_layer(acc_va, l, fits[r][l]) for l in range(L)]
    full = [fit_layer(acc_tr, l, d, best_a) for l in range(L)]
    rep["r2_val"]["LINEAR_FULL_RANK_ref"] = [r2_layer(acc_va, l, full[l]) for l in range(L)]
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"alpha": best_a, "mean": mean_fits, "lr": fits}, OUT / "compensators.pt")
    write_json(OUT / "compensators.json", rep)
    print("R2 val (media camadas 2-27):")
    for k, v in rep["r2_val"].items():
        print(f"  {k:22s} {sum(v[2:28]) / 26:.4f}   camadas 2/14/27: {v[2]:.3f} {v[14]:.3f} {v[27]:.3f}")


if __name__ == "__main__":
    main()
