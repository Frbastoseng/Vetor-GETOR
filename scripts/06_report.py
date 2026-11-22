"""Gera results/REPORT.md e graficos a partir dos JSON mais recentes (nenhum numero manual).

  python scripts/06_report.py [--eval results/<run>_eval_test.json] [--bench results/<run>.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from vetor.report import RES, latest, load_json, merged_table, pareto_plots


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=None)
    ap.add_argument("--bench", default=None)
    args = ap.parse_args()
    ev_p = Path(args.eval) if args.eval else latest("*_eval_test.json")
    if ev_p is None:
        raise SystemExit("Nenhuma avaliacao de teste encontrada (rode scripts/04_evaluate.py --split test)")
    ev = load_json(ev_p)
    bp = Path(args.bench) if args.bench else latest("*_final_bench.json")
    bench = load_json(bp)
    base = load_json(latest("*_baseline.json"))
    diag = load_json(RES / "diag_dispatch.json") if (RES / "diag_dispatch.json").exists() else None
    train = load_json(latest("*_train.json"))
    df = merged_table(ev, bench)
    plots = pareto_plots(df, RES)

    L = []
    L.append("# VETOR — relatório de resultados (gerado automaticamente)\n")
    L.append(f"- Avaliação de qualidade: `{ev_p.name}` ({ev['n_windows']} janelas de **teste**, "
             f"prompt {ev['prompt_len']} + continuação {ev['continuation_len']} tokens)")
    if bench:
        L.append(f"- Benchmark: `{bp.name}` ({bench['repeats']} repetições pareadas, "
                 f"{bench['decode_forwards']} forwards de decode, prompt {bench['prompt_len']}, {bench['dtype']})")
    L.append(f"- Guia: alvo `{ev['guide'].get('target')}`, semente {ev['guide'].get('seed')}, "
             f"pool `{ev['guide'].get('pool_extra')}` — selecionado na validação")
    env = ev["env"]
    L.append(f"- Ambiente: {env.get('cpu')}, macOS {env.get('macos')}, torch {env.get('torch')}, "
             f"transformers {env.get('transformers')}, device `{env.get('device')}`\n")

    if base:
        t = base["timing_full"]
        L.append("## Baseline FULL\n")
        L.append(f"- Equivalência com `generate`: {base['equivalence_vs_generate']['n_identical']}/"
                 f"{base['n_prompts']} sequências idênticas; diferença de logits com wrappers: "
                 f"{base['equivalence_vs_generate']['max_abs_prefill_logit_diff_patched_vs_unpatched']}")
        L.append(f"- Decode FULL: **{t['decode_tok_s_median']:.1f} tok/s** (mediana), prefill {t['prefill_ms_median']:.1f} ms")
        if "timing_full_without_throttle" in base:
            L.append(f"- Sem throttle de eventos MPS: {base['timing_full_without_throttle']['decode_tok_s_median']:.1f} tok/s\n")

    L.append("## Tabela principal (teste)\n")
    L.append("Omissões = fração dos 30 MLPs pulados por token de decode. ΔNLL% = aumento relativo da NLL "
             "do texto verdadeiro (teacher forcing com KV, política ativa). KL contra FULL no mesmo prefixo. "
             "Speedup pareado contra FULL (IC95 bootstrap). ORACLE-DIAG usa informação do próprio exemplo "
             "e **não é implementável**: é só um limite superior.\n")
    cols = ["policy", "k", "skip_rate_all", "rel_delta_nll_pct", "rel_delta_nll_ci95_lo", "rel_delta_nll_ci95_hi",
            "kl_mean", "kl_p90", "top1_agree", "greedy_match", "decode_tok_s_median", "speedup_vs_full_median",
            "speedup_ci95_lo", "speedup_ci95_hi", "decode_ms_p95"]
    hdr = ["política", "k", "omissões", "ΔNLL %", "IC95", "KL médio", "KL p90", "top-1 =", "greedy =",
           "tok/s", "speedup", "IC95", "p95 ms"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for _, r in df.sort_values(["k", "policy"]).iterrows():
        g = r.get
        L.append("| " + " | ".join([
            str(g("policy")), str(int(g("k"))), f"{100 * g('skip_rate_all'):.1f}%",
            fmt(g("rel_delta_nll_pct"), 2), f"[{fmt(g('rel_delta_nll_ci95_lo'), 2)}, {fmt(g('rel_delta_nll_ci95_hi'), 2)}]",
            fmt(g("kl_mean"), 4), fmt(g("kl_p90"), 4), fmt(g("top1_agree"), 3), fmt(g("greedy_match"), 3),
            fmt(g("decode_tok_s_median"), 1), fmt(g("speedup_vs_full_median"), 3),
            f"[{fmt(g('speedup_ci95_lo'), 3)}, {fmt(g('speedup_ci95_hi'), 3)}]", fmt(g("decode_ms_p95"), 1)]) + " |")
    L.append("")

    # comparacao direta GUIDED vs controles no mesmo k
    L.append("## GUIDED contra controles no mesmo orçamento (ΔNLL %, teste)\n")
    piv = df.pivot_table(index="k", columns="policy", values="rel_delta_nll_pct")
    L.append(piv.to_markdown(floatfmt=".2f"))
    L.append("")
    if bench and bench.get("guide_overhead"):
        o = bench["guide_overhead"]
        L.append(f"Custo do guia: {o['n_params']} parâmetros, ~{o['flops_per_decision_approx']:,} FLOPs por "
                 f"decisão; {o['us_per_decision_median']:.0f} µs por decisão (mediana, com sincronização).\n")
    if diag:
        L.append("## Diagnóstico de despacho (por que o ganho de tempo é pequeno)\n")
        L.append(f"- FULL: {diag['full']['ms_per_token']:.2f} ms/token, dos quais "
                 f"{diag['full']['cpu_enqueue_ms_per_token']:.2f} ms são CPU enfileirando (decode limitado pela CPU).")
        L.append(f"- Teto prático (26 MLPs elegíveis pulados): {diag['skip_26_eligible']['ms_per_token']:.2f} ms/token "
                 f"(−{100 * (1 - diag['skip_26_eligible']['ms_per_token'] / diag['full']['ms_per_token']):.1f}%).")
        L.append(f"- Economia medida por MLP pulado: {diag['mlp_saving_us_per_layer_measured']:.1f} µs.\n")
    if train:
        L.append("## Seleção do guia (validação)\n")
        L.append(f"Critério STATIC (val): {train.get('static_val_criterion', float('nan')):.4f}\n")
        L.append("| alvo | pool | critério médio (3 sementes) |\n|---|---|---|")
        for r in train.get("family_mean_criterion", []):
            L.append(f"| {r['target']} | {r['pool_extra']} | {r['mean_crit']:.4f} |")
        L.append("")
    L.append("## Gráficos\n")
    for p in plots + ["sensitivity.png"]:
        L.append(f"![{p}]({p})")
    (RES / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Gerado results/REPORT.md;", ", ".join(plots))


if __name__ == "__main__":
    main()
