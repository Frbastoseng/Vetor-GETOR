"""Etapa 5: treina o vetor-guia (P, e, w, b) nos dados do professor. Selecao SO na validacao.

Saidas: results/guide.pt (melhor alvo na validacao), results/{run_id}.json com metricas offline.
"""
from __future__ import annotations

import argparse
import json

import torch

from vetor.guide import budget_to_k, eligible_layers
from vetor.model_adapter import ROOT, load_config, new_run_id, seed_all, write_json
from vetor.train_router import load_teacher, offline_metrics, static_ranking, train_guide


def offline_metrics_static(t, k, rank, pf, pl):
    L = t["kl"].shape[1]
    chosen = [l for l in rank if pf <= l < L - pl][:k]
    return float(t["kl"][:, chosen].sum(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["regress", "quantile:0.1", "quantile:0.25", "classify"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--pools", nargs="+", default=["none", "2", "4", "6"],
                    help="m extra do pool (k+m camadas mais seguras do treino); none = sem pool")
    args = ap.parse_args()
    cfg = load_config()
    seed_all(cfg["seed"])
    run_id = new_run_id("train")
    gc, rc = cfg["guide"], cfg["routing"]
    tr = load_teacher(ROOT / "data" / "teacher_train.pt")
    va = load_teacher(ROOT / "data" / "teacher_val.pt")
    L, d = tr["z"].shape[1], tr["z"].shape[2]
    pf, pl = rc["protected_first"], rc["protected_last"]
    n_el = len(eligible_layers(L, pf, pl))
    ks = [budget_to_k(b, n_el) for b in rc["budgets"]]
    rank = static_ranking(tr)  # STATIC usa SO o treino (simetria com o guia)

    report = {"run_id": run_id, "static_rank_train": rank, "ks": ks, "candidates": []}
    nk = sum(1 for k in ks if k)
    st_crit = sum(offline_metrics_static(va, k, rank, pf, pl) for k in ks if k) / nk
    print(f"Referencia STATIC (val): crit={st_crit:.4f}")
    for tspec in args.targets:
        target, q = (tspec.split(":") + [None])[:2]
        q = float(q) if q else 0.2
        for seed in args.seeds:
            g, info = train_guide(tr, va, n_layers=L, hidden=d, pf=pf, pl=pl, target=target,
                                  dim=gc["dim"], epochs=gc["epochs"], lr=gc["lr"], q=q,
                                  weight_decay=gc["weight_decay"], batch_size=gc["batch_size"], seed=seed,
                                  log=lambda *_: None)
            m_tr = offline_metrics(g, tr, n_layers=L, pf=pf, pl=pl, ks=ks, static_rank=rank)
            for pool in args.pools:
                pe = None if pool == "none" else int(pool)
                m_val = offline_metrics(g, va, n_layers=L, pf=pf, pl=pl, ks=ks, static_rank=rank, pool_extra=pe)
                # criterio (validacao): soma media de KL isolado das k camadas omitidas, media nos k
                crit = sum(m_val[f"k{k}"]["guided_sumKL"] for k in ks if k) / nk
                cand = {"target": tspec, "seed": seed, "pool_extra": pe, "n_params": info["n_params"],
                        "best_val_loss": info["best_val_loss"], "val": m_val,
                        "train_auc_within_layer": m_tr["auc_safe_within_layer_mean"],
                        "val_criterion_mean_sumKL": crit}
                report["candidates"].append(cand)
                cand["_guide"] = g
                print(f"{tspec:14s} seed={seed} pool={pool:4s}: AUC intra-camada val="
                      f"{m_val['auc_safe_within_layer_mean']:.3f} (treino {m_tr['auc_safe_within_layer_mean']:.3f})"
                      f" crit={crit:.4f}")

    # Regra anti-vies de selecao: escolhe a FAMILIA (alvo, pool) pela media entre sementes e usa a
    # PRIMEIRA semente dessa familia (regra pre-declarada; nunca a melhor semente individual).
    fam = {}
    for c in report["candidates"]:
        fam.setdefault((c["target"], c["pool_extra"]), []).append(c)
    fam_mean = {k: sum(c["val_criterion_mean_sumKL"] for c in v) / len(v) for k, v in fam.items()}
    best_fam = min(fam_mean, key=fam_mean.get)
    cand = sorted(fam[best_fam], key=lambda c: c["seed"])[0]
    g, crit = cand["_guide"], cand["val_criterion_mean_sumKL"]
    for c in report["candidates"]:
        c.pop("_guide", None)
    report["family_mean_criterion"] = [{"target": k[0], "pool_extra": k[1], "mean_crit": v}
                                       for k, v in sorted(fam_mean.items(), key=lambda x: x[1])]
    print("\nFamilias (media entre sementes, validacao):")
    for r in report["family_mean_criterion"]:
        print(f"  {r['target']:14s} pool={r['pool_extra']}: {r['mean_crit']:.4f}")
    torch.save({"state_dict": g.state_dict(), "dim": gc["dim"], "target": cand["target"],
                "seed": cand["seed"], "pool_extra": cand["pool_extra"], "static_rank": rank,
                "labels": cfg["labels"], "selected_on": "val", "val_criterion": crit},
               ROOT / "results" / "guide.pt")
    report["static_val_criterion"] = st_crit
    report["selected"] = {"target": cand["target"], "seed": cand["seed"], "pool_extra": cand["pool_extra"],
                          "criterion": crit}
    write_json(ROOT / "results" / f"{run_id}.json", report)
    print("\nSelecionado (validacao):", report["selected"])
    print("Validacao — soma de KL isolado das k camadas omitidas (menor = melhor):")
    for k in ks:
        if k:
            v = cand["val"][f"k{k}"]
            print(f"  k={k:2d}  GUIDED {v['guided_sumKL']:.4f}  STATIC {v['static_sumKL']:.4f}  "
                  f"RANDOM {v['random_sumKL']:.4f}  ORACLE {v['oracle_sumKL']:.4f}  "
                  f"guided<static em {100 * v['guided_beats_static_frac']:.0f}% das amostras")


if __name__ == "__main__":
    main()
