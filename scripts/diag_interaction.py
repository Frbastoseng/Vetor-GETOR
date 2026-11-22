"""Diagnostico (VALIDACAO): interacao entre camadas x acumulacao no KV cache.

Para cada prefixo e conjunto S de camadas omitidas (STATIC, RANDOM, e STATIC espalhado), mede na
MESMA posicao (ultimo token, prefixo intacto):
  aditivo  = soma_l KL_l      (o que o professor/guia assumem)
  conjunto = KL(S inteiro)    (interacao, ainda sem acumulacao)
O KL acumulado do decode vem de results/*_eval_val.json (mesmas politicas).
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from vetor.collect import load_articles, teacher_windows, tokenize_docs
from vetor.model_adapter import ROOT, load_config, load_model, pick_device, write_json
from vetor.report import latest
from vetor.wrappers import patch_model, reset


@torch.inference_mode()
def joint_kl(model, wraps, prefix, sets, device):
    """Linha 0 = FULL; linha i+1 = todas as camadas de sets[i] zeradas no ultimo token."""
    ids = torch.tensor(prefix, device=device).view(1, -1).expand(len(sets) + 1, -1).contiguous()
    rows = {l: [] for l in range(len(wraps))}
    for i, S in enumerate(sets):
        for l in S:
            rows[l].append(i + 1)
    reset(wraps)
    for l, w in enumerate(wraps):
        if rows[l]:
            w.ablate_rows = torch.tensor(rows[l], device=device)
    try:
        lg = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    finally:
        reset(wraps)
    lp = F.log_softmax(lg, -1)
    return (lp[0].exp() * (lp[0] - lp[1:])).sum(-1).clamp_min(0).cpu().numpy()


def main():
    cfg = load_config()
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    wraps = patch_model(model)
    L = len(wraps)
    ck = torch.load(ROOT / "results" / "guide.pt", weights_only=False)
    rank = [l for l in ck["static_rank"] if 2 <= l < L - 2]
    rng = np.random.default_rng(0)
    sets, names = [], []
    for k in (3, 5, 8, 10):
        st = sorted(rank[:k])
        sets.append(st); names.append(f"STATIC@k{k}")
        rd = sorted(rng.choice(range(2, L - 2), k, replace=False).tolist())
        sets.append(rd); names.append(f"RANDOM@k{k}")
        # STATIC espalhado: das 2k mais seguras, pega uma sim, uma nao (evita blocos contiguos)
        sp = sorted(sorted(rank[:2 * k])[::2])[:k]
        sets.append(sp); names.append(f"STATIC-SPREAD@k{k}")
    singles = [[l] for l in range(2, L - 2)]
    docs = tokenize_docs(tok, load_articles(cfg, "val"), cfg["data"]["min_doc_tokens"])
    wins = teacher_windows(docs, 2, tuple(cfg["data"]["teacher"]["prefix_len"]), 555)
    add, joint = [], []
    for w in wins:
        s = joint_kl(model, wraps, w["prefix"], singles, device)
        j = joint_kl(model, wraps, w["prefix"], sets, device)
        add.append([s[[l - 2 for l in S]].sum() for S in sets])
        joint.append(j)
    add, joint = np.array(add), np.array(joint)
    ev = json.loads(latest("*_eval_val.json").read_text())
    acc = {(r["policy"], r["k"]): r["kl_mean"] for r in ev["summary"]}
    out = []
    for i, (n, S) in enumerate(zip(names, sets)):
        pol, k = n.split("@k")
        r = {"policy": n, "layers": S, "additive_kl_mean": float(add[:, i].mean()),
             "joint_kl_mean": float(joint[:, i].mean()),
             "joint_over_additive": float(joint[:, i].mean() / add[:, i].mean()),
             "decode_accumulated_kl_mean": acc.get((pol, int(k)))}
        out.append(r)
        print(f"{n:20s} {str(S):42s} aditivo {r['additive_kl_mean']:.3f}  conjunto {r['joint_kl_mean']:.3f} "
              f"(x{r['joint_over_additive']:.2f})  decode acumulado {r['decode_accumulated_kl_mean']}")
    write_json(ROOT / "results" / "diag_interaction_val.json", {"n_prefixes": len(wins), "rows": out})


if __name__ == "__main__":
    main()
