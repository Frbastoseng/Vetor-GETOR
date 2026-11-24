"""v0.4 / Fase 2: compensadores do 1.7B (MEAN, LR16/32/64) por regressao de posto reduzido (codigo da v0.2).
Estatisticas em textos de TREINO dos 4 dominios; alpha escolhido no R2 de VALIDACAO (r=32), como na v0.2.
Saida: results/v0.4/comps_1.7B.pt (float32 contiguo, compacto), results/v0.4/comps_1.7B.json
"""
from __future__ import annotations

import time

import torch

from vetor.model_adapter import pick_device, seed_all, write_json
from vetor.v2.compensate import collect_stats, fit_layer, r2_layer, token_batches
from vetor.v2.engine import patch_v2, zero_comps
from vetor.v4.common import OUT, load
from vetor.v4.data4 import load_texts
from vetor.v4.pipe import COMPS_FILE, DOMAINS

RANKS = (16, 32, 64)
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1)


def main():
    seed_all(1234)
    device = pick_device("auto")
    model, tok = load("1.7B", device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    ids = {s: [tok(x["text"], add_special_tokens=False).input_ids for dom in DOMAINS for x in load_texts(dom, s)]
           for s in ("train", "val")}
    t0 = time.time()
    acc_tr, n_tr = collect_stats(model, wraps, token_batches(ids["train"], 256, 16, 120, 1), device)
    acc_va, n_va = collect_stats(model, wraps, token_batches(ids["val"], 256, 16, 24, 2), device)
    print(f"tokens treino={n_tr} val={n_va} ({time.time() - t0:.0f}s)", flush=True)
    rep = {"n_tokens_train": n_tr, "n_tokens_val": n_va, "alpha_search": {}, "r2_val": {}}
    best_a, best = None, -1e9
    for a in ALPHAS:
        r2 = sum(r2_layer(acc_va, l, fit_layer(acc_tr, l, 32, a)) for l in range(L)) / L
        rep["alpha_search"][str(a)] = r2
        print(f"alpha={a:g}: R2 val medio (r=32) = {r2:.4f}", flush=True)
        if r2 > best:
            best_a, best = a, r2
    rep["alpha"] = best_a
    c = lambda t: t.float().contiguous().clone()
    mean_fits = [{"ybar": c(acc_tr.moments(l, acc_tr.n)[1])} for l in range(L)]
    rep["r2_val"]["ZERO"] = [r2_layer(acc_va, l, None) for l in range(L)]
    rep["r2_val"]["MEAN"] = [r2_layer(acc_va, l, {"ybar": mean_fits[l]["ybar"].double()}) for l in range(L)]
    fits = {}
    for r in RANKS:
        raw = [fit_layer(acc_tr, l, r, best_a) for l in range(L)]
        rep["r2_val"][f"LR{r}"] = [r2_layer(acc_va, l, f) for l, f in enumerate(raw)]
        fits[r] = [{k: c(v) for k, v in f.items()} for f in raw]
    torch.save({"alpha": best_a, "mean": mean_fits, "lr": fits}, COMPS_FILE)
    write_json(OUT / "comps_1.7B.json", rep)
    print("R2 val (media das camadas elegiveis 2-21):")
    for k, v in rep["r2_val"].items():
        print(f"  {k:6s} {sum(v[2:22]) / 20:.4f}   camadas 2/12/21: {v[2]:.3f} {v[12]:.3f} {v[21]:.3f}")


if __name__ == "__main__":
    main()
