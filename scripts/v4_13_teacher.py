"""v0.4 / Fase 2: professor acumulado Delta_N (N=8,16,32) no 1.7B (codigo da v0.2: label_window).

Rodada 0: base = FULL. Rodada r>0: base = decisoes do GUIDED-T (semente 0) da rodada anterior em k=5 (DAgger).
  python scripts/v4_13_teacher.py --round 0
  python scripts/v4_13_teacher.py --round 1 --base-router results/v0.4/routers/T_r0_s0.pt
Saida: data/v4_teacher_r{round}.pt  (360 janelas de treino, 90 por dominio, 12 posicoes, 20 camadas)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from vetor.model_adapter import ROOT, pick_device, seed_all
from vetor.v2.engine import Ctrl, patch_v2, set_comps, zero_comps
from vetor.v2.fasteval import logprobs_D
from vetor.v2.teacher2 import concat, label_window, sample_positions
from vetor.v4.common import OUT, eligible, load
from vetor.v4.pipe import NW, P, load_comps, windows
from vetor.v4.routers import load_router4

K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--base-router", default=None)
    ap.add_argument("--per-domain", type=int, default=90)
    ap.add_argument("--positions", type=int, default=12)
    ap.add_argument("--comp", default=None, help="padrao: compensador escolhido na validacao; ZERO = controle sem compensacao")
    args = ap.parse_args()
    seed_all(1234 + args.round)
    device = pick_device("auto")
    model, tok = load("1.7B", device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    E = eligible(L)
    comp = args.comp or json.loads((OUT / "static_1.7B.json").read_text())["chosen_comp"]
    wraps = patch_v2(model, zero_comps(L, d, device))
    set_comps(wraps, load_comps(comp, L, d, device))
    router = None
    if args.round > 0:
        router, ck = load_router4(args.base_router, device)
        router.configure(E, k_max=int(-(-1.5 * K // 1)), tau=ck["tau_tf"][str(float(K))])
    rng = np.random.default_rng(200 + args.round)
    wins = windows(tok, "train", seed=21 + 10 * args.round, per_domain=args.per_domain)
    parts, t0 = [], time.time()
    for i, w in enumerate(wins):
        T = w.shape[0] - 1
        if router is None:
            base = torch.zeros(T, L, dtype=torch.bool)
        else:
            c = Ctrl(start=P, L=L, router=router.masked, eligible=router.eligible)
            logprobs_D(model, wraps, w.to(device)[:-1].view(1, -1), P, c)
            base = c.bits[0].cpu()
        parts.append(label_window(model, wraps, w, P, base, sample_positions(P, NW, args.positions, rng), E, device))
        if i % 20 == 0:
            print(f"r{args.round}: {i}/{len(wins)} ({time.time() - t0:.0f}s)", flush=True)
    data = concat(parts)
    data.update({"comp": comp, "round": args.round, "n_windows": len(wins),
                 "base_skip_rate": float(data["was_skipped"].float().mean())})
    sfx = "" if args.comp is None else f"_{args.comp}"
    torch.save(data, ROOT / "data" / f"v4_teacher_r{args.round}{sfx}.pt")
    print(f"{len(wins)} janelas, {data['lid'].shape[0]} celulas, {time.time() - t0:.0f}s, "
          f"Delta16 mediana={data['d16'].median():.4f} p90={data['d16'].quantile(0.9):.4f}")


if __name__ == "__main__":
    main()
