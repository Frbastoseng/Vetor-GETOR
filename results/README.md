# Índice de resultados

Todos os arquivos foram gerados pelos scripts, sem edição manual. Os nomes seguem `AAAAMMDD-HHMMSS_<etapa>`.

## Execuções canônicas (usadas no relatório)

| Arquivo | Etapa | O que contém |
|---|---|---|
| `REPORT.md`, `pareto_*.png` | 06 | relatório final gerado automaticamente |
| `00_sanity.json` | 0 | ambiente, config do modelo, FLOPs do MLP |
| `20261122-172602_baseline.json` | 1 | FULL: equivalência, CPU×MPS, tempos com e sem throttle |
| `20261122-173621_teacher.json`, `_layers.csv`, `sensitivity.png` | 2–3 | professor (3.622/240/240 prefixos) e mapa de sensibilidade por camada |
| `20261122-174506_train.json`, `guide.pt` | 5 | candidatos do guia, seleção por família na validação, guia final |
| `20261122-174643_eval_val*` | 6–7 | qualidade na validação |
| `20261122-180647_eval_test*` | 6–7 | **qualidade no teste (rodado uma vez)** |
| `20261122-184318_final_bench*` | 7 | **tempo: 29 políticas × 20 repetições pareadas**, overhead do guia |
| `20261122-173234_m2_controls*` | 4 | gate M2: STATIC/RANDOM sem guia |
| `diag_dispatch.json` | 4 | decode limitado pela CPU; teto do skip |
| `diag_interaction_val.json` | pós-validação | interação entre camadas × acumulação no KV |

## Execuções superadas (mantidas por transparência)

| Arquivo | Por que foi superada |
|---|---|
| `20261122-172116_baseline.json` | antes do throttle MPS (loop ~2× mais lento) |
| `20261122-172828_teacher*` | professor com 2 prefixos/documento; trocado por 6/4/4 após overfitting do guia |
| `20261122-173431_train.json`, `20261122-173537_train.json` | treinos sobre o professor menor |
| `20261122-174331_train.json` | seleção pela melhor semente individual, trocada pela regra por família (evita viés de seleção) |
| `ablation_smoke.json` | smoke test do protótipo inicial |

Os tensores do professor (`data/teacher_*.pt`, ~144 MB) não são versionados; `scripts/02_ablate.py` os recria de forma determinística.
