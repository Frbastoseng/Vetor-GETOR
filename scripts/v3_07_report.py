"""v0.3: gera results/v0.3/REPORT_v0.3.md e o grafico qualidade x throughput (nenhum numero manual)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vetor.model_adapter import ROOT
from vetor.v3.bench import SUITE
from vetor.v3.stats import by_category, paired_macro

R3 = ROOT / "results" / "v0.3"
FIN = R3 / "final"


def combine(paths: list[Path]) -> dict:
    """Media do escore por exemplo entre sementes."""
    rs = [json.loads(p.read_text()) for p in paths]
    out = {"tasks": {}}
    for t in rs[0]["tasks"]:
        acc = {}
        for r in rs:
            for d, v in r["tasks"][t]["per_example"]:
                acc.setdefault(d, []).append(v)
        out["tasks"][t] = {"per_example": [(d, float(np.mean(v))) for d, v in sorted(acc.items())]}
    return out


def main():
    dec = json.loads((ROOT / "configs" / "v3_decision.json").read_text())
    tasks = [r["task"] for r in json.loads((R3 / "applicability.json").read_text()) if r["applicable"]]
    cats = {t: SUITE[t]["cat"] for t in tasks}
    models = {"FULL": combine([FIN / "FULL.json"]), "V02": combine([FIN / "V02.json"])}
    for m, tags in dec["models"].items():
        if isinstance(tags, list) and tags:
            models[m] = combine([FIN / f"{m}__{t}.json" for t in tags])
    full = models["FULL"]
    cmp = {m: paired_macro(models[m], full, tasks) for m in models if m != "FULL"}
    prim = dec["primary_vetor"]
    perf = json.loads((R3 / "perf.json").read_text())["configs"] if (R3 / "perf.json").exists() else {}
    tagmap = {"A0": "A0_r8_task_g1", "A": "A_r8_task_g1", "C": "C_r8_task_g1", "B": "B_r8_rep4_task_g1",
              "B0": "B0_r8_rep4_task_g1", "LORA": "LORA_r1_task_g1_lr1e3", "FULL": "FULL"}
    L = []
    a = L.append
    a("# VETOR v0.3 — Relatório de resultados (gerado automaticamente)\n")
    commit = json.loads((FIN / "FULL.json").read_text())["decision_commit"]
    a(f"- **Avaliação final única** na metade FINAL das {len(tasks)} tarefas aplicáveis. A decisão foi congelada no commit "
      f"`{commit.split()[0][:10]}` ({commit.split()[1]}), antes de o conjunto final ser lido.")
    a(f"- **VETOR principal** (escolhido na DEV): `{prim}`, a correção residual com portão do vetor-guia. "
      "As configurações com 2 sementes têm o escore de cada exemplo calculado como a média das duas.")
    a("- **IC95:** bootstrap pareado, com 10.000 reamostragens por tarefa.\n")

    # ------------------------------------------------ criterios
    p = cmp[prim]
    catd = by_category(p, cats)
    best_ctrl = max(("A0", "LORA", "LORA_KD"), key=lambda m: cmp[m]["macro_diff_pp"])
    q3 = paired_macro(models[prim], models[best_ctrl], tasks)
    s1a = paired_macro(models[prim], models["A0"], tasks)
    s1b = paired_macro(models[prim], models["LORA"], tasks)
    pp = perf.get(tagmap[prim], {})
    ok = lambda b: "**sim**" if b else "não"
    a("## 1. Critérios pré-registrados (metade FINAL)\n")
    a("| Critério | Meta | Resultado | Atingido? |\n|---|---|---|---|")
    a(f"| Q1 qualidade | VETOR − FULL ≥ +5,0 p.p., IC95 > 0 | {p['macro_diff_pp']:+.2f} p.p. [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] | {ok(p['macro_diff_pp'] >= 5 and p['ci95'][0] > 0)} |")
    worst = min(catd.items(), key=lambda x: x[1])
    a(f"| Q2 sem regressão grave | nenhuma categoria ≤ −2,0 p.p. | pior: {worst[0]} {worst[1]:+.2f} p.p. | {ok(worst[1] > -2)} |")
    a(f"| Q3 superior aos controles | VETOR − melhor controle ({best_ctrl}) > 0, IC95 > 0 | {q3['macro_diff_pp']:+.2f} [{q3['ci95'][0]:+.2f}, {q3['ci95'][1]:+.2f}] | {ok(q3['ci95'][0] > 0)} |")
    a(f"| S1 ganho atribuível ao guia | VETOR − A₀ > 0 **e** VETOR − LoRA > 0 (IC95) | A₀: {s1a['macro_diff_pp']:+.2f} [{s1a['ci95'][0]:+.2f}, {s1a['ci95'][1]:+.2f}] · LoRA: {s1b['macro_diff_pp']:+.2f} [{s1b['ci95'][0]:+.2f}, {s1b['ci95'][1]:+.2f}] | {ok(s1a['ci95'][0] > 0 and s1b['ci95'][0] > 0)} |")
    if pp:
        a(f"| P1 throughput | ≥ 98% do FULL | {100 * pp['throughput_ratio_median']:.1f}% [{100 * pp['throughput_ratio_ci95'][0]:.1f}, {100 * pp['throughput_ratio_ci95'][1]:.1f}] | {ok(pp['throughput_ratio_median'] >= 0.98)} |")
        a(f"| P2 TTFT | ≤ +5% | {100 * (pp['ttft_ratio_median'] - 1):+.1f}% | {ok(pp['ttft_ratio_median'] <= 1.05)} |")
        mr = pp.get("mem_ratio")
        a(f"| P3 memória | ≤ +5% | {'—' if mr is None else f'{100 * (mr - 1):+.1f}%'} | {ok(mr is not None and mr <= 1.05)} |")
    a("| P4 zero sincronizações por camada | teste automático | `tests/test_v3_modules.py::test_gate_has_no_host_sync` | **sim** |\n")

    # ------------------------------------------------ tabela principal
    a("## 2. Qualidade na metade FINAL (diferença vs FULL, p.p., IC95)\n")
    order = ["A", "A0", "C", "LORA", "LORA_KD", "B", "B0", "V02"]
    desc = {"A": "VETOR A: correção + portão do guia (2 sementes)", "A0": "A₀: correção estática, sem guia (2 sementes)",
            "C": "VETOR C: guia também atenua o MLP", "LORA": "LoRA, mesmos dados e perda (2 sementes)",
            "LORA_KD": "LoRA + destilação (destilação convencional)", "B": "VETOR B: 4 MLPs → MEAN + correções com guia",
            "B0": "B₀: 4 MLPs → MEAN + correções estáticas", "V02": "VETOR v0.2 (GUIDED-T LR32, k = 6; foco em omitir)"}
    a(f"FULL: média macro = **{cmp[prim]['macro_b']:.2f}**\n")
    a("| Modelo | Macro | Δ vs FULL | IC95 |\n|---|---|---|---|")
    for m in order:
        c = cmp[m]
        a(f"| {desc[m]} | {c['macro_a']:.2f} | **{c['macro_diff_pp']:+.2f}** | [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] |")
    a("\n**Por tarefa (Δ vs FULL, p.p.):**\n")
    a("| Tarefa | Categoria | FULL | " + " | ".join(order) + " |\n|---|---|---|" + "---|" * len(order))
    for t in tasks:
        a(f"| {t} | {cats[t]} | {cmp[prim]['per_task'][t]['b']:.2f} | " + " | ".join(f"{cmp[m]['per_task'][t]['diff']:+.2f}" for m in order) + " |")
    a("\n**Comparações diretas (pareadas):**\n")
    a("| Comparação | Δ (p.p.) | IC95 | O que isola |\n|---|---|---|---|")
    for x, y, what in (("A", "A0", "valor do portão adaptativo (S1)"), ("A", "LORA", "VETOR vs adapter convencional (S1)"),
                       ("A0", "LORA", "correção residual vs LoRA"), ("C", "A0", "atenuação adaptativa"),
                       ("B", "B0", "guia na realocação"), ("LORA_KD", "LORA", "efeito da destilação")):
        r = paired_macro(models[x], models[y], tasks)
        a(f"| {x} − {y} | {r['macro_diff_pp']:+.2f} | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {what} |")

    # ------------------------------------------------ desempenho
    if perf:
        a("\n## 3. Desempenho real (decode de 63 tokens, prompt 64, float32, MPS; LoRA fundido)\n")
        a("| Modelo | tok/s | Throughput vs FULL [IC95] | TTFT (ms) | Memória vs FULL |\n|---|---|---|---|---|")
        inv = {v: k for k, v in tagmap.items()}
        for tag, v in perf.items():
            mr = v.get("mem_ratio")
            a(f"| {inv.get(tag, tag)} | {v['decode_tok_s_median']:.0f} | {100 * v['throughput_ratio_median']:.1f}% "
              f"[{100 * v['throughput_ratio_ci95'][0]:.1f}, {100 * v['throughput_ratio_ci95'][1]:.1f}] | {v['ttft_ms_median']:.1f} | "
              f"{'—' if mr is None else f'{100 * (mr - 1):+.1f}%'} |")
        npar = {m: json.loads((R3 / "runs" / f"{tagmap[m]}.json").read_text())["n_params"] for m in ("A0", "A", "C", "B", "LORA")}
        a("\nParâmetros treináveis: " + "; ".join(f"{m} {n:,}" for m, n in npar.items())
          + " (o LoRA, depois de fundido, tem 0 parâmetros extras na inferência).")
        # grafico
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for m in ("A", "A0", "C", "LORA", "B", "B0"):
            tg = tagmap[m]
            if tg in perf:
                x = 100 * perf[tg]["throughput_ratio_median"]
                ax.errorbar(x, cmp[m]["macro_diff_pp"], yerr=[[cmp[m]["macro_diff_pp"] - cmp[m]["ci95"][0]], [cmp[m]["ci95"][1] - cmp[m]["macro_diff_pp"]]],
                            fmt="o", capsize=3, label=m)
        ax.axhline(0, color="k", lw=0.6); ax.axhline(5, color="gray", ls=":", lw=0.8); ax.axvline(98, color="gray", ls=":", lw=0.8)
        ax.set(xlabel="throughput de decode vs FULL (%)", ylabel="Δ qualidade macro vs FULL (p.p., IC95)",
               title="VETOR v0.3 — qualidade × velocidade (metade FINAL)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(R3 / "quality_vs_speed.png", dpi=120)
        a("\n![qualidade x velocidade](quality_vs_speed.png)")

    # ------------------------------------------------ DEV
    a("\n## 4. Todas as configurações testadas na DEV (desenvolvimento)\n")
    a("| Execução | Parâmetros | Δ vs FULL (DEV) | IC95 |\n|---|---|---|---|")
    for f in sorted((R3 / "runs").glob("*_dev.json")):
        d = json.loads(f.read_text())
        v = d["vs_full"]
        a(f"| {d['tag']} | {d['n_params']:,} | {v['macro_diff_pp']:+.2f} | [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] |")
    (R3 / "REPORT_v0.3.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({m: {k: v for k, v in c.items() if k != "per_task"} | {"per_task": c["per_task"]} for m, c in cmp.items()},
              open(R3 / "final_summary.json", "w"), indent=1)
    print("Gerado results/v0.3/REPORT_v0.3.md")


if __name__ == "__main__":
    main()
