"""v0.2 / Q3: o professor novo ajuda? Treina o MESMO GUIDED-T com os rotulos da v0.1
(KL isolado de 1 camada, no proximo token, prefixo intacto; data/teacher_train.pt) e avalia igual
(mesmo compensador, tau calibrado na validacao para k=6, mesmas janelas de validacao).
"""
from __future__ import annotations

import argparse
import sys

import torch

from vetor.model_adapter import ROOT, load_config, load_model, pick_device, seed_all, write_json
from vetor.v2.data import load_split
from vetor.v2.engine import patch_v2, set_comps, zero_comps
from vetor.v2.search import make_windows
from vetor.v2.train2 import calibrate_tau, eval_router, train_router

sys.path.insert(0, str(ROOT / "scripts"))
from v2_03_teacher import load_comps  # noqa: E402

P = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="LR32")
    args = ap.parse_args()
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    set_comps(wraps, load_comps(args.comp, L, d, device))
    E = list(range(2, L - 2))
    t = torch.load(ROOT / "data" / "teacher_train.pt", weights_only=False)
    N = t["z"].shape[0]
    lid = torch.tensor(E).repeat(N)
    data = {"z": t["z"][:, E].reshape(-1, d), "lid": lid, "bits": torch.zeros(lid.shape[0], L, dtype=torch.bool),
            "n": torch.zeros(lid.shape[0], dtype=torch.long), "d1": t["kl"][:, E].reshape(-1)}
    data["z_lag"] = data["z"]
    va = make_windows(tok, load_split("val"), P, 64, 200, 12)
    rep = {"comp": args.comp, "label": "v0.1 (KL isolado, proximo token)", "n_cells": int(lid.shape[0]), "runs": []}
    for loss, q in (("mse", 0.5), ("pinball", 0.3)):
        r = train_router(data, d=d, L=L, N=1, lagged=False, loss=loss, q=q, device=device)
        calibrate_tau(model, wraps, r, va[:64], P, device, 6.0, E)
        s = eval_router(model, wraps, r, va, P, device)
        rep["runs"].append({"loss": loss, **s})
        print(f"GUIDED-T com rotulo v0.1 ({loss}): dNLL={s['rel_dnll_pct']:.2f}% {s['ci95']} skips/token={s['skips_per_token']:.2f}")
    write_json(ROOT / "results" / "v0.2" / f"q3_old_teacher_{args.comp}.json", rep)


if __name__ == "__main__":
    main()
