"""v0.2 / etapa 3: professor acumulado Delta_N (N=8,16,32) sob uma mascara-base.

Rodada 0: base = FULL. Rodada r>0: base = decisoes do roteador GUIDED-T da rodada anterior (DAgger).
  python scripts/v2_03_teacher.py --comp LR32 --round 0
  python scripts/v2_03_teacher.py --comp LR32 --round 1 --base-router results/v0.2/router_LR32_r0_T.pt --k 6
Saida: data/v2_teacher_{comp}_r{round}_{split}.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from vetor.model_adapter import ROOT, load_config, load_model, pick_device, seed_all
from vetor.v2.data import load_split
from vetor.v2.engine import Ctrl, patch_v2, set_comps, zero_comps
from vetor.v2.fasteval import logprobs_D
from vetor.v2.router2 import TraversalRouter
from vetor.v2.search import make_windows
from vetor.v2.teacher2 import concat, label_window, sample_positions

P, N = 64, 64


def load_comps(name, L, d, device):
    from vetor.v2.compensate import make_comps
    if name == "ZERO":
        return zero_comps(L, d, device)
    f = ROOT / "results" / "v0.2" / "compensators.pt"
    if not f.exists():  # copia compacta publicada (v0.4); mesmos valores em float32
        f = ROOT / "results" / "v0.2" / "compensators_f32.pt"
    ck = torch.load(f, weights_only=False)
    if name == "MEAN":
        return make_comps("mean", ck["mean"], L, d, 0, device)
    r = int(name[2:])
    return make_comps("lr", ck["lr"][r], L, d, r, device)


def load_router(path, device):
    ck = torch.load(path, weights_only=False)
    r = TraversalRouter(ck["d"], ck["L"], dim=ck["dim"], lagged=ck["lagged"])
    r.load_state_dict(ck["state_dict"])
    r.k_max, r.n_el, r.y_mu, r.y_sd = ck["k_max"], ck["n_el"], ck["y_mu"], ck["y_sd"]
    return r.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--base-router", default=None)
    ap.add_argument("--k", type=float, default=6)
    ap.add_argument("--n-train", type=int, default=480)
    ap.add_argument("--n-val", type=int, default=120)
    ap.add_argument("--positions", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config()
    seed_all(cfg["seed"] + args.round)
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    set_comps(wraps, load_comps(args.comp, L, d, device))
    E = list(range(2, L - 2))
    router = None
    if args.round > 0:
        router, rck = load_router(args.base_router, device)
        tau = rck["tau_by_k"][str(float(args.k))]
        router.configure(E, k_max=int(-(-1.5 * args.k // 1)), tau=tau)
    rng = np.random.default_rng(100 + args.round)
    for split, nwin, seed in (("train", args.n_train, 21), ("val", args.n_val, 22)):
        wins = make_windows(tok, load_split(split), P, N, nwin, seed + 10 * args.round)
        parts, t0 = [], time.time()
        for w in tqdm(wins, desc=f"{args.comp} r{args.round} {split}"):
            T = w.shape[0] - 1
            if router is None:
                base = torch.zeros(T, L, dtype=torch.bool)
            else:  # decisoes realizadas pelo roteador nesta janela
                ctrl = Ctrl(start=P, L=L, router=router.masked, eligible=router.eligible)
                logprobs_D(model, wraps, w.to(device)[:-1].view(1, -1), P, ctrl)
                base = ctrl.bits[0].cpu()
            pos = sample_positions(P, N, args.positions, rng)
            parts.append(label_window(model, wraps, w, P, base, pos, E, device))
        data = concat(parts)
        data.update({"comp": args.comp, "round": args.round, "split": split, "n_windows": len(wins),
                     "base_skip_rate": float(data["was_skipped"].float().mean())})
        torch.save(data, ROOT / "data" / f"v2_teacher_{args.comp}_r{args.round}_{split}.pt")
        print(f"{split}: {len(wins)} janelas, {data['lid'].shape[0]} celulas, {time.time() - t0:.0f}s, "
              f"Delta16 mediana={data['d16'].median():.4f} p90={data['d16'].quantile(0.9):.4f}")


if __name__ == "__main__":
    main()
