"""v0.2: gera results/v0.2/REPORT_v0.2.md e graficos a partir dos JSON (nenhum numero manual)."""
from __future__ import annotations

import json

from vetor.model_adapter import ROOT

R2 = ROOT / "results" / "v0.2"
KS = (3, 5, 6, 8, 10)


def J(name):
    p = R2 / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}%"


def main():
    dec = json.loads((ROOT / "configs" / "v2_decision.json").read_text())
    fin, comp, orc, otok = J("final_eval.json"), J("compensators.json"), J("oracle_val.json"), J("oracle_token_val.json")
    q3 = J("q3_old_teacher_LR32.json")
    bench = J("benchmark_final.json")
    v01 = json.loads(sorted((ROOT / "results").glob("*_eval_test.json"))[-1].read_text())
    pol = fin["policies"]
    prim = dec["primary"]
    d, dff, L = 576, 1536, 30
    mlp_f = 6 * d * dff
    total_f = 2 * 134_515_008
    L_ = []
    a = L_.append
    a("# VETOR v0.2 — relatório comparativo (gerado automaticamente)\n")
    a(f"- **Conjunto final:** {fin['n_windows']} janelas de artigos inéditos (WikiText-103 fora do WikiText-2), "
      f"avaliado **uma vez**. Decisão congelada no commit `{fin['decision_commit'].split()[0][:10]}` "
      f"({fin['decision_commit'].split()[1]}); início da avaliação {fin['started']}.")
    a(f"- **Configuração principal (escolhida na validação):** `{prim}`")
    a(f"- Ambiente: {fin['env']['cpu']}, torch {fin['env']['torch']}, transformers {fin['env']['transformers']}, `{fin['env']['device']}`\n")

    # ---------------------------------------------------------------- criterio
    p6 = pol[f"{prim}@k6"]
    sg6 = pol["STATIC-GREEDY-LR32@k6"]
    rd6 = pol["RANDOM-LR32@k6"]
    sp = bench["decode"].get(f"{prim}@k6") if bench else None
    a("## 1. Critério de sucesso (inalterado desde a v0.1), em k = 6 (23,1% dos MLPs elegíveis)\n")
    a("| Critério | Meta | v0.2 (conjunto final) | Atingido? |\n|---|---|---|---|")
    a(f"| MLPs elegíveis omitidos | ≥ 20% | {100 * p6['skips_per_token'] / 26:.1f}% ({p6['skips_per_token']:.2f}/token) | "
      f"{'sim' if p6['skips_per_token'] / 26 >= 0.2 else 'não'} |")
    a(f"| Aumento relativo de NLL | ≤ 5% | {pct(p6['rel_dnll_pct'])} [IC95 {p6['ci95'][0]:.2f}, {p6['ci95'][1]:.2f}] | "
      f"{'sim' if p6['ci95'][1] <= 5 else ('sim (ponto), IC95 cruza 5%' if p6['rel_dnll_pct'] <= 5 else 'não')} |")
    if sp:
        a(f"| Ganho de velocidade de geração | ≥ 5% | {100 * (sp['speedup_median'] - 1):+.1f}% "
          f"[IC95 {100 * (sp['speedup_ci95'][0] - 1):+.1f}, {100 * (sp['speedup_ci95'][1] - 1):+.1f}] | "
          f"{'sim' if sp['speedup_ci95'][0] >= 1.05 else 'não'} |")
    beats = p6["ci95"][1] < min(sg6["ci95"][0], rd6["ci95"][0])
    a(f"| Superior a RANDOM e STATIC-GREEDY no mesmo orçamento (IC95 sem sobreposição) | sim | "
      f"{pct(p6['rel_dnll_pct'])} vs {pct(sg6['rel_dnll_pct'])} / {pct(rd6['rel_dnll_pct'])} | {'sim' if beats else 'não'} |\n")

    # ---------------------------------------------------------------- qualidade
    a("## 2. Qualidade no conjunto final (ΔNLL relativo, IC95)\n")
    names = [n for n in dict.fromkeys(k.rsplit("@", 1)[0] for k in pol)]
    a("| Política | " + " | ".join(f"k={k}" for k in KS) + " |\n|---|" + "---|" * len(KS))
    for n in names:
        cells = []
        for k in KS:
            v = pol.get(f"{n}@k{k}")
            cells.append("—" if v is None else f"{v['rel_dnll_pct']:.2f} [{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]")
        a(f"| {n} | " + " | ".join(cells) + " |")
    a("\nTaxa efetiva medida (omissões/token) do roteador principal: " +
      ", ".join(f"k={k}: {pol[f'{prim}@k{k}']['skips_per_token']:.2f}" for k in KS) + "\n")

    # ---------------------------------------------------------------- v0.1 x v0.2
    v01p = {(r["policy"], r["k"]): r["rel_delta_nll_pct"] for r in v01["summary"]}
    a("## 3. Comparação com a v0.1\n")
    a("A v0.1 foi avaliada no seu próprio conjunto de teste (WikiText-2); a v0.2, no conjunto final novo. "
      "Por isso, as políticas da v0.1 foram **reavaliadas no conjunto final da v0.2** (linhas `v0.1-*`).\n")
    a("| k | melhor v0.1 (teste v0.1) | v0.1-GUIDED-D1 (final v0.2) | STATIC-GREEDY-LR32 | **v0.2 principal** |\n|---|---|---|---|---|")
    for k in KS:
        best01 = min((v for (p, kk), v in v01p.items() if kk == k and p != "ORACLE-DIAG"), default=None)
        hist = pol.get(f"v0.1-GUIDED-D1-ZERO@k{k}", {}).get("rel_dnll_pct")
        a(f"| {k} | {pct(best01)} | {pct(hist)} | {pct(pol[f'STATIC-GREEDY-LR32@k{k}']['rel_dnll_pct'])} | "
          f"**{pct(pol[f'{prim}@k{k}']['rel_dnll_pct'])}** |")
    a("")

    # ---------------------------------------------------------------- perguntas
    a("## 4. Perguntas do pré-registro (evidência de validação; final quando indicado)\n")
    sgv = lambda c, k: orc["comps"][c]["policies"][f"STATIC-GREEDY@k{k}"]["rel_dnll_pct"]
    a(f"- **Q1 — compensar reduz o dano?** STATIC-GREEDY k=6 (validação): ZERO {pct(sgv('ZERO', 6))}, MEAN "
      f"{pct(sgv('MEAN', 6))}, LR8 {pct(sgv('LR8', 6))}, LR16 {pct(sgv('LR16', 6))}, LR32 {pct(sgv('LR32', 6))}. "
      f"No final: ZERO {pct(pol['STATIC-GREEDY-ZERO@k6']['rel_dnll_pct'])} → LR32 {pct(sg6['rel_dnll_pct'])}.")
    a(f"- **Q2 — existe combinação viável?** ORACLE-GREEDY-SEQ k=6 (val, LR32): "
      f"{pct(orc['comps']['LR32']['policies']['ORACLE-GREEDY-SEQ@k6']['rel_dnll_pct'])} → por sequência, não. "
      f"ORACLE-GREEDY-TOKEN k=6 (val, {otok['n_windows']} janelas): LR32 "
      f"{pct(otok['comps']['LR32']['ORACLE-GREEDY-TOKEN']['rel_dnll_pct'])}, ZERO "
      f"{pct(otok['comps']['ZERO']['ORACLE-GREEDY-TOKEN']['rel_dnll_pct'])} → por token, sim.")
    a(f"- **Q3 — o professor novo ajuda?** GUIDED-T + LR32 k=6 (val): rótulo v0.1 "
      f"{pct(min(r['rel_dnll_pct'] for r in q3['runs']))} vs rótulo acumulado (rodada 0) "
      f"{pct(J('router_LR32_r0.json')['best']['T']['curve_val']['6']['rel_dnll_pct'])}.")
    dl = [n for n in names if n.startswith("GUIDED-DL-LR32")]
    if dl:
        a(f"- **Q4 — decidir na travessia ajuda?** No final, k=6, mesmo professor/compensador/taxa: GUIDED-DL "
          f"{pct(pol[f'{dl[0]}@k6']['rel_dnll_pct'])} vs GUIDED-T {pct(p6['rel_dnll_pct'])}.")
    a(f"- **Dependência do compensador:** com ZERO, GUIDED-T k=6 = {pct(pol.get('GUIDED-T-ZERO@k6', {}).get('rel_dnll_pct'))} "
      f"vs STATIC-GREEDY-ZERO {pct(pol['STATIC-GREEDY-ZERO@k6']['rel_dnll_pct'])} (final).\n")

    # ---------------------------------------------------------------- custos
    a("## 5. Operações, custos e velocidade (separados)\n")
    comp_f = 2 * d * 32 + 2 * d
    router_f = 2 * d * 16 + 2 * L * 16 + 2 * 16 + 5 * d
    sk = p6["skips_per_token"]
    saved = sk * mlp_f - sk * comp_f - 26 * router_f
    a("| Item (por token de decode, k=6) | Valor |\n|---|---|")
    a(f"| FLOPs de MLP evitados | {sk * mlp_f / 1e6:.1f} M ({100 * sk / L:.1f}% dos MLPs) |")
    a(f"| Custo da compensação LR32 | {sk * comp_f / 1e6:.2f} M FLOPs ({comp_f:,} por omissão = {100 * comp_f / mlp_f:.2f}% de um MLP) |")
    a(f"| Custo do roteador (≈26 decisões) | {26 * router_f / 1e6:.2f} M FLOPs |")
    a(f"| **Saldo de operações evitadas** | **{saved / 1e6:.1f} M FLOPs = {100 * saved / (L * mlp_f):.1f}% dos FLOPs de MLP, "
      f"{100 * saved / total_f:.1f}% do total do modelo** |")
    if bench:
        a(f"| Tempo de um MLP isolado (1 token) | {bench['mlp_us']:.1f} µs |")
        for c, v in bench["comp_us"].items():
            a(f"| Tempo do compensador {c} (1 token) | {v:.1f} µs |")
        for n, v in bench["router"].items():
            a(f"| Roteador {n}: tempo por decisão com sincronização | {v['us_per_decision_with_sync']:.0f} µs "
              f"({v['syncs_per_token']} sincronizações/token, {v['params']:,} parâmetros) |")
        a("\n**Latência e velocidade real (execução condicional, decode de 63 tokens, mediana de repetições pareadas):**\n")
        a("| Política | ms/token | p95 | tok/s | speedup vs FULL [IC95] | omissões/token |\n|---|---|---|---|---|---|")
        for n, v in bench["decode"].items():
            a(f"| {n} | {v['ms_per_token_median']:.2f} | {v['ms_per_token_p95']:.2f} | {v['tok_s_median']:.0f} | "
              f"{v['speedup_median']:.3f} [{v['speedup_ci95'][0]:.3f}, {v['speedup_ci95'][1]:.3f}] | {v['skips_per_token']:.2f} |")
    a("")
    a("## 6. Compensadores (R² na validação, média das camadas 2–27)\n")
    a("| " + " | ".join(comp["r2_val"]) + " |\n|" + "---|" * len(comp["r2_val"]))
    a("| " + " | ".join(f"{sum(v[2:28]) / 26:.3f}" for v in comp["r2_val"].values()) + " |\n")
    (R2 / "REPORT_v0.2.md").write_text("\n".join(L_) + "\n", encoding="utf-8")
    print("Gerado results/v0.2/REPORT_v0.2.md")


if __name__ == "__main__":
    main()
