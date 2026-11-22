"""Relatorio final: junta qualidade (teste) + tempo (benchmark) e gera tabela e graficos de Pareto.

Mostra SEMPRE os dados crus, inclusive quando GUIDED perde para FULL/STATIC/RANDOM.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .model_adapter import ROOT

RES = ROOT / "results"


def latest(pattern: str) -> Path | None:
    files = sorted(RES.glob(pattern))
    return files[-1] if files else None


def load_json(p: Path | None) -> dict | None:
    return json.loads(p.read_text()) if p else None


def merged_table(eval_json: dict, bench_json: dict | None) -> pd.DataFrame:
    q = pd.DataFrame(eval_json["summary"])
    if bench_json:
        b = pd.DataFrame(bench_json["summary"])
        b["base"] = b.policy.str.split("@").str[0]
        b = b.rename(columns={"policy": "bench_name"})
        q = q.merge(b[["base", "k", "decode_tok_s_median", "decode_ms_p95", "prefill_ms_median",
                       "speedup_vs_full_median", "speedup_ci95_lo", "speedup_ci95_hi", "skip_rate"]],
                    left_on=["policy", "k"], right_on=["base", "k"], how="left").drop(columns=["base"])
    return q


def pareto_plots(df: pd.DataFrame, out: Path, title_suffix: str = "") -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    style = {"RANDOM": ("tab:gray", "x"), "STATIC": ("tab:blue", "s"), "ORACLE-DIAG": ("tab:green", "*"),
             "GUIDED-S": ("tab:red", "o"), "GUIDED-D1": ("tab:orange", "^"), "GUIDED-D4": ("tab:purple", "v"),
             "GUIDED-D8": ("tab:brown", "<"), "GUIDED-D16": ("tab:pink", ">")}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, g in df.groupby("policy", sort=False):
        c, m = style.get(name, ("k", "."))
        g = g.sort_values("k")
        ax[0].plot(100 * g.skip_rate_all, g.rel_delta_nll_pct, marker=m, color=c, label=name)
        ax[1].plot(100 * g.skip_rate_all, g.kl_mean, marker=m, color=c, label=name)
    ax[0].axhline(5, color="gray", ls=":", lw=0.8)
    ax[0].set(xlabel="MLPs omitidos no decode (% de 30)", ylabel="aumento relativo de NLL (%)",
              title="Qualidade x economia" + title_suffix)
    ax[1].set(xlabel="MLPs omitidos no decode (% de 30)", ylabel="KL medio vs FULL (nats/token)",
              yscale="log", title="Divergencia x economia" + title_suffix)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    p = out / "pareto_quality.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    files.append(p.name)

    if "speedup_vs_full_median" in df and df.speedup_vs_full_median.notna().any():
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for name, g in df.dropna(subset=["speedup_vs_full_median"]).groupby("policy", sort=False):
            c, m = style.get(name, ("k", "."))
            g = g.sort_values("k")
            sp = 100 * (g.speedup_vs_full_median - 1)
            ax.errorbar(sp, g.rel_delta_nll_pct,
                        xerr=[sp - 100 * (g.speedup_ci95_lo - 1), 100 * (g.speedup_ci95_hi - 1) - sp],
                        marker=m, color=c, label=name, capsize=2)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(5, color="gray", ls=":", lw=0.8)
        ax.axvline(5, color="gray", ls=":", lw=0.8)
        ax.set(xlabel="ganho de decode tok/s vs FULL (%, IC95)", ylabel="aumento relativo de NLL (%)",
               title="Pareto velocidade x qualidade (teste)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out / "pareto_speed.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        files.append(p.name)
    return files
