"""Etapas 2-3: professor por ablacao isolada do MLP no ultimo token + mapa de sensibilidade.

Saidas:
  data/teacher_{train,val,test}.pt  — z [N,L,d], kl [N,L], delta_nll [N,L], metadados
  results/{run_id}.json, results/{run_id}_layers.csv, results/sensitivity.png
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vetor.collect import ablation_teacher, load_articles, teacher_windows, tokenize_docs
from vetor.model_adapter import (ROOT, env_info, load_config, load_model, new_run_id, pick_device,
                                 seed_all, sha256_json, write_json)
from vetor.wrappers import patch_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--max-per-split", type=int, default=None)
    ap.add_argument("--per-doc", nargs="+", type=int, default=None,
                    help="prefixos por documento para cada split (mesma ordem de --splits)")
    args = ap.parse_args()
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    run_id = new_run_id("teacher")
    model, tok = load_model(cfg, device)
    wraps = patch_model(model)
    L = len(wraps)
    tc, lab = cfg["data"]["teacher"], cfg["labels"]
    meta = {"run_id": run_id, "env": env_info(device), "revision": cfg["model"]["revision"],
            "labels": lab, "teacher_cfg": tc, "splits": {}}
    layer_rows = []
    for si, split in enumerate(args.splits):
        docs = tokenize_docs(tok, load_articles(cfg, split), cfg["data"]["min_doc_tokens"])
        ppd = tc["prefixes_per_doc"]
        per_doc = args.per_doc[si] if args.per_doc else (ppd[split] if isinstance(ppd, dict) else ppd)
        wins = teacher_windows(docs, per_doc, tuple(tc["prefix_len"]), cfg["seed"] + si)
        if args.max_per_split:
            wins = wins[: args.max_per_split]
        Z, KL, DN, NLL = [], [], [], []
        t0 = time.time()
        for w in tqdm(wins, desc=split):
            r = ablation_teacher(model, wraps, w["prefix"], w["target"], device)
            Z.append(r["z"].cpu())
            KL.append(r["kl"].cpu())
            DN.append(r["delta_nll"].cpu())
            NLL.append(r["nll_full"].cpu())
        Z, KL, DN, NLL = torch.stack(Z), torch.stack(KL), torch.stack(DN), torch.stack(NLL)
        safe = (KL <= lab["kl_max"]) & (DN <= lab["delta_nll_max"])
        info = {"n_samples": len(wins), "n_docs": len({w["doc_id"] for w in wins}),
                "seconds": time.time() - t0,
                "data_hash": sha256_json([(w["doc_id"], w["start"], len(w["prefix"])) for w in wins]),
                "safe_rate": float(safe.float().mean())}
        meta["splits"][split] = info
        torch.save({"z": Z.half(), "kl": KL, "delta_nll": DN, "nll_full": NLL, "safe": safe,
                    "doc_id": [w["doc_id"] for w in wins], "start": [w["start"] for w in wins],
                    "prefix_len": [len(w["prefix"]) for w in wins], "labels": lab,
                    "revision": cfg["model"]["revision"]},
                   ROOT / "data" / f"teacher_{split}.pt")
        for l in range(L):
            k = KL[:, l].numpy()
            layer_rows.append({"split": split, "layer": l, "kl_mean": k.mean(), "kl_median": np.median(k),
                               "kl_p90": np.percentile(k, 90), "delta_nll_mean": DN[:, l].mean().item(),
                               "safe_rate": safe[:, l].float().mean().item()})
        print(split, info)

    df = pd.DataFrame(layer_rows)
    df.to_csv(ROOT / "results" / f"{run_id}_layers.csv", index=False)
    meta["layers_csv"] = f"{run_id}_layers.csv"
    write_json(ROOT / "results" / f"{run_id}.json", meta)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = df[df.split == "train"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].semilogy(tr.layer, tr.kl_median, "o-", label="mediana")
    ax[0].semilogy(tr.layer, tr.kl_mean, "s--", label="media")
    ax[0].semilogy(tr.layer, tr.kl_p90, "^:", label="p90")
    ax[0].axhline(cfg["labels"]["kl_max"], color="gray", lw=0.8)
    ax[0].set(xlabel="camada", ylabel="KL(p_full || p_ablado) [nats]", title="Sensibilidade por camada (treino)")
    ax[0].legend()
    ax[1].bar(tr.layer, tr.safe_rate)
    ax[1].set(xlabel="camada", ylabel="fracao safe", title="Rotulo safe (KL<=0.02 e dNLL<=0.05)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "sensitivity.png", dpi=120)
    print(tr[["layer", "kl_median", "kl_mean", "kl_p90", "safe_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
