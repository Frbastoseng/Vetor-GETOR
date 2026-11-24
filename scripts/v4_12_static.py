"""v0.4 / Fase 2: STATIC-GREEDY (ordem gulosa em janelas de TREINO), RANDOM e ORACLE-GREEDY-SEQ (diagnostico)
para cada compensador no 1.7B; escolha do compensador pelo dNLL do STATIC-GREEDY em k=5 na VALIDACAO (como v0.2).
Saida: results/v0.4/static_1.7B.json
"""
from __future__ import annotations

import time

import torch

from vetor.model_adapter import pick_device, seed_all, write_json
from vetor.v2.engine import patch_v2, set_comps, zero_comps
from vetor.v2.search import eval_batch, greedy_sets, masks_from_sets, summarize
from vetor.v4.common import OUT, eligible, load
from vetor.v4.pipe import NW, P, load_comps, random_sets, windows

KS = (3, 4, 5, 6, 8)
K_MAIN = 5
COMPS = ("ZERO", "MEAN", "LR16", "LR32", "LR64")


def main():
    seed_all(1234)
    device = pick_device("auto")
    model, tok = load("1.7B", device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    E = eligible(L)
    wraps = patch_v2(model, zero_comps(L, d, device))
    tr = windows(tok, "train", seed=11, per_domain=24)  # 24 janelas de treino por dominio
    va = windows(tok, "val", part="eval", seed=12)
    T = P + NW - 1
    rep = {"n_train_windows": len(tr), "n_val_windows": len(va), "ks": KS, "comps": {}}
    for cname in COMPS:
        set_comps(wraps, load_comps(cname, L, d, device))
        t0 = time.time()
        sg, traj = greedy_sets(model, wraps, tr, P, device, E, max(KS), chunk_rows=20)
        res = {"static_greedy_order": sg, "static_greedy_train_kl": traj, "policies": {}}
        for k in KS:
            for pname, sets in (("STATIC-GREEDY", [sg[:k]] * len(va)), ("RANDOM", random_sets(E, len(va), k, 7))):
                r = eval_batch(model, wraps, va, P, device, masks=masks_from_sets(sets, T, P, L), chunk=16)
                res["policies"][f"{pname}@k{k}"] = summarize(r)
        rep["comps"][cname] = res
        print(f"== {cname} ({time.time() - t0:.0f}s) ordem {sg}: " + "  ".join(
            f"k{k}: SG={res['policies'][f'STATIC-GREEDY@k{k}']['rel_dnll_pct']:.2f}% RND={res['policies'][f'RANDOM@k{k}']['rel_dnll_pct']:.2f}%"
            for k in KS), flush=True)
        write_json(OUT / "static_1.7B.json", rep)
    best = min(COMPS, key=lambda c: rep["comps"][c]["policies"][f"STATIC-GREEDY@k{K_MAIN}"]["rel_dnll_pct"])
    rep["chosen_comp"] = best
    rep["rule"] = "menor dNLL do STATIC-GREEDY em k=5 na validacao (regra da v0.2)"
    # oraculo por sequencia (diagnostico) em 16 janelas de validacao (4 por dominio), compensador escolhido
    set_comps(wraps, load_comps(best, L, d, device))
    sub = windows(tok, "val", part="eval", seed=12, per_domain=4)
    orc = {k: [] for k in KS}
    for w in sub:
        o, _ = greedy_sets(model, wraps, [w], P, device, E, max(KS), chunk_rows=20)
        for k in KS:
            orc[k].append(o[:k])
    sg = rep["comps"][best]["static_greedy_order"]
    for k in KS:
        rep.setdefault("oracle_seq", {})[f"k{k}"] = {
            "ORACLE-GREEDY-SEQ": summarize(eval_batch(model, wraps, sub, P, device, masks=masks_from_sets(orc[k], T, P, L))),
            "STATIC-GREEDY(same subset)": summarize(eval_batch(model, wraps, sub, P, device,
                                                               masks=masks_from_sets([sg[:k]] * len(sub), T, P, L)))}
    write_json(OUT / "static_1.7B.json", rep)
    print("compensador escolhido:", best, {k: v["ORACLE-GREEDY-SEQ"]["rel_dnll_pct"] for k, v in rep["oracle_seq"].items()})


if __name__ == "__main__":
    main()
