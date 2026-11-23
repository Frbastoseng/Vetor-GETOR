"""v0.2 / etapa 2b: ORACLE-GREEDY-TOKEN (diagnostico, NAO implementavel; usa a propria continuacao).

Para cada token de decode t, em ordem, com as decisoes de tokens anteriores fixadas: escolhe de forma
gulosa k camadas que minimizam sum_{p=t}^{t+H-1} KL(q_FULL(p) || q_M(p)). Responde: decisoes POR TOKEN
conseguem, em principio, dNLL <= 5% com k = 6?
Saida: results/v0.2/oracle_token_val.json
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from vetor.model_adapter import ROOT, load_config, load_model, pick_device, seed_all, write_json
from vetor.v2.data import load_split
from vetor.v2.engine import Ctrl, patch_v2, set_comps, zero_comps
from vetor.v2.fasteval import kl_rows, logprobs_D
from vetor.v2.search import eval_batch, make_windows, masks_from_sets, summarize

sys.path.insert(0, str(ROOT / "scripts"))
from v2_03_teacher import load_comps  # noqa: E402

P, NW = 64, 64


@torch.inference_mode()
def oracle_token_mask(model, wraps, window, k, E, H, device):
    L = len(wraps)
    w = window.to(device)
    x = w[:-1].view(1, -1)
    T = x.shape[1]
    lpf = logprobs_D(model, wraps, x, P, None)  # [1,D,V]
    mask = torch.zeros(T, L, dtype=torch.bool, device=device)
    for t in range(P, T):
        end = min(t + H, T)  # posicoes [t, end)
        xt = x[:, :end]
        chosen = []
        for _ in range(k):
            rest = [c for c in E if c not in chosen]
            m = mask[:end].unsqueeze(0).repeat(len(rest), 1, 1)
            for i, c in enumerate(rest):
                m[i, t, chosen + [c]] = True
            lp = logprobs_D(model, wraps, xt.expand(len(rest), -1), P, Ctrl(start=P, L=L, static_mask=m))
            score = kl_rows(lpf[:, t - P:end - P], lp[:, t - P:end - P]).sum(1)
            chosen.append(rest[int(score.argmin())])
        mask[t, chosen] = True
    return mask.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", nargs="+", default=["LR32", "ZERO"])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    E = list(range(2, L - 2))
    import json
    orc = json.loads((ROOT / "results" / "v0.2" / "oracle_val.json").read_text())
    va = make_windows(tok, load_split("val"), P, NW, 200, 12)[: args.n]  # mesmas 1as janelas do oraculo seq
    rep = {"k": args.k, "H": args.H, "n_windows": len(va), "comps": {}}
    T = P + NW - 1
    for c in args.comps:
        set_comps(wraps, load_comps(c, L, d, device))
        t0 = time.time()
        masks = torch.stack([oracle_token_mask(model, wraps, w, args.k, E, args.H, device) for w in va])
        tok_r = summarize(eval_batch(model, wraps, va, P, device, masks=masks))
        sg = orc["comps"][c]["static_greedy_order"][: args.k]
        sg_r = summarize(eval_batch(model, wraps, va, P, device, masks=masks_from_sets([sg] * len(va), T, P, L)))
        rep["comps"][c] = {"ORACLE-GREEDY-TOKEN": tok_r, "STATIC-GREEDY(same windows)": sg_r,
                           "layer_usage": masks[:, P:].float().mean((0, 1)).tolist(), "seconds": time.time() - t0}
        print(f"{c}: ORACLE-GREEDY-TOKEN k={args.k}: {tok_r['rel_dnll_pct']:.2f}% {tok_r['ci95']} | "
              f"STATIC-GREEDY mesmas janelas: {sg_r['rel_dnll_pct']:.2f}% ({time.time() - t0:.0f}s)")
        write_json(ROOT / "results" / "v0.2" / "oracle_token_val.json", rep)


if __name__ == "__main__":
    main()
