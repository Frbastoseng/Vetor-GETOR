# VETOR v0.2 — relatório comparativo (gerado automaticamente)

- **Conjunto final:** 200 janelas de artigos inéditos (WikiText-103 fora do WikiText-2), avaliado **uma vez**. Decisão congelada no commit `e18af9110e` (2026-11-22T21:52:46-03:00); início da avaliação 2026-11-22 21:52:56.
- **Configuração principal (escolhida na validação):** `GUIDED-T-LR32`
- Ambiente: Apple M4 Max, torch 2.15.0, transformers 5.24.0, `mps`

## 1. Critério de sucesso (inalterado desde a v0.1), em k = 6 (23,1% dos MLPs elegíveis)

| Critério | Meta | v0.2 (conjunto final) | Atingido? |
|---|---|---|---|
| MLPs elegíveis omitidos | ≥ 20% | 23.3% (6.06/token) | sim |
| Aumento relativo de NLL | ≤ 5% | 4.47% [IC95 4.03, 4.92] | sim |
| Ganho de velocidade de geração | ≥ 5% | -80.6% [IC95 -82.2, -79.8] | não |
| Superior a RANDOM e STATIC-GREEDY no mesmo orçamento (IC95 sem sobreposição) | sim | 4.47% vs 10.86% / 15.82% | sim |

## 2. Qualidade no conjunto final (ΔNLL relativo, IC95)

| Política | k=3 | k=5 | k=6 | k=8 | k=10 |
|---|---|---|---|---|---|
| STATIC-GREEDY-ZERO | 6.75 [6.2, 7.3] | 14.76 [13.9, 15.6] | 18.54 [17.6, 19.6] | 27.56 [26.2, 29.0] | 36.51 [34.8, 38.4] |
| STATIC-ZERO | 7.67 [7.2, 8.2] | 16.00 [15.1, 16.9] | 39.19 [37.2, 41.4] | 75.43 [71.6, 79.5] | 117.02 [111.6, 122.8] |
| RANDOM-ZERO | 12.10 [11.3, 13.0] | 21.06 [19.9, 22.3] | 26.59 [25.2, 28.2] | 39.02 [36.9, 41.1] | 54.71 [52.2, 57.5] |
| GUIDED-T-ZERO | 4.40 [3.9, 4.9] | 12.63 [11.6, 13.7] | 18.59 [17.3, 20.1] | 34.16 [32.3, 36.2] | 54.48 [51.7, 57.6] |
| GUIDED-DL-ZERO | 6.44 [5.9, 7.0] | 18.42 [17.0, 19.9] | 26.57 [24.7, 28.5] | 49.49 [46.2, 52.9] | 79.07 [74.4, 83.8] |
| STATIC-GREEDY-MEAN | 6.52 [5.9, 7.2] | 12.08 [11.2, 13.0] | 16.23 [15.3, 17.3] | 24.19 [22.9, 25.7] | 32.54 [31.0, 34.3] |
| STATIC-MEAN | 7.19 [6.7, 7.7] | 13.60 [12.8, 14.4] | 28.07 [26.5, 29.7] | 53.02 [50.2, 56.1] | 74.59 [71.1, 78.7] |
| RANDOM-MEAN | 10.94 [10.1, 11.9] | 18.32 [17.2, 19.5] | 22.62 [21.4, 23.9] | 32.22 [30.7, 33.9] | 43.69 [41.7, 46.0] |
| STATIC-GREEDY-LR8 | 5.29 [4.8, 5.8] | 9.96 [9.2, 10.8] | 13.35 [12.5, 14.3] | 19.14 [18.1, 20.3] | 26.53 [25.1, 28.0] |
| STATIC-LR8 | 5.66 [5.2, 6.1] | 11.02 [10.4, 11.7] | 19.21 [18.2, 20.3] | 33.90 [32.2, 35.7] | 44.09 [42.1, 46.2] |
| RANDOM-LR8 | 9.50 [8.7, 10.4] | 15.57 [14.6, 16.7] | 18.61 [17.5, 19.8] | 26.38 [25.1, 27.8] | 34.23 [32.6, 36.1] |
| STATIC-GREEDY-LR16 | 4.86 [4.4, 5.3] | 9.69 [9.0, 10.4] | 12.00 [11.2, 12.9] | 17.48 [16.4, 18.6] | 24.14 [22.8, 25.5] |
| STATIC-LR16 | 5.05 [4.6, 5.5] | 9.70 [9.1, 10.3] | 16.18 [15.3, 17.2] | 27.82 [26.4, 29.4] | 36.61 [34.9, 38.4] |
| RANDOM-LR16 | 9.01 [8.3, 9.8] | 14.50 [13.6, 15.6] | 17.31 [16.2, 18.5] | 24.39 [23.1, 25.7] | 31.33 [29.8, 33.0] |
| STATIC-GREEDY-LR32 | 4.39 [4.0, 4.8] | 8.00 [7.4, 8.6] | 10.86 [10.2, 11.6] | 15.64 [14.7, 16.6] | 20.68 [19.5, 22.0] |
| STATIC-LR32 | 4.28 [3.9, 4.7] | 8.00 [7.4, 8.6] | 11.97 [11.2, 12.7] | 19.73 [18.7, 20.9] | 26.71 [25.4, 28.1] |
| RANDOM-LR32 | 8.19 [7.4, 9.0] | 13.22 [12.3, 14.2] | 15.82 [14.7, 17.0] | 22.18 [21.0, 23.4] | 27.95 [26.5, 29.5] |
| GUIDED-T-LR32 | 1.52 [1.3, 1.8] | 3.08 [2.7, 3.5] | 4.47 [4.0, 4.9] | 7.34 [6.7, 8.0] | 11.88 [11.1, 12.7] |
| GUIDED-DL-LR32 | 3.28 [2.8, 3.8] | 6.43 [5.8, 7.1] | 8.11 [7.5, 8.8] | 12.53 [11.7, 13.4] | 18.55 [17.5, 19.6] |
| v0.1-GUIDED-S-ZERO | 10.54 [9.6, 11.5] | 22.47 [21.1, 24.0] | 32.01 [29.9, 34.5] | 61.44 [57.0, 66.3] | 93.55 [87.3, 100.5] |
| v0.1-GUIDED-D1-ZERO | 8.44 [7.8, 9.1] | 18.11 [17.0, 19.4] | 25.39 [24.0, 27.0] | 47.63 [45.3, 50.2] | 76.09 [72.2, 80.5] |

Taxa efetiva medida (omissões/token) do roteador principal: k=3: 3.06, k=5: 5.10, k=6: 6.06, k=8: 8.08, k=10: 10.10

## 3. Comparação com a v0.1

A v0.1 foi avaliada no seu próprio conjunto de teste (WikiText-2); a v0.2, no conjunto final novo. Por isso, as políticas da v0.1 foram **reavaliadas no conjunto final da v0.2** (linhas `v0.1-*`).

| k | melhor v0.1 (teste v0.1) | v0.1-GUIDED-D1 (final v0.2) | STATIC-GREEDY-LR32 | **v0.2 principal** |
|---|---|---|---|---|
| 3 | 6.84% | 8.44% | 4.39% | **1.52%** |
| 5 | 15.59% | 18.11% | 8.00% | **3.08%** |
| 6 | — | 25.39% | 10.86% | **4.47%** |
| 8 | 38.57% | 47.63% | 15.64% | **7.34%** |
| 10 | 54.83% | 76.09% | 20.68% | **11.88%** |

## 4. Perguntas do pré-registro (evidência de validação; final quando indicado)

- **Q1 — compensar reduz o dano?** STATIC-GREEDY k=6 (validação): ZERO 18.19%, MEAN 16.21%, LR8 13.20%, LR16 11.76%, LR32 10.46%. No final: ZERO 18.54% → LR32 10.86%.
- **Q2 — existe combinação viável?** ORACLE-GREEDY-SEQ k=6 (val, LR32): 7.80% → por sequência, não. ORACLE-GREEDY-TOKEN k=6 (val, 16 janelas): LR32 1.19%, ZERO 2.37% → por token, sim.
- **Q3 — o professor novo ajuda?** GUIDED-T + LR32 k=6 (val): rótulo v0.1 7.17% vs rótulo acumulado (rodada 0) 4.70%.
- **Q4 — decidir na travessia ajuda?** No final, k=6, mesmo professor/compensador/taxa: GUIDED-DL 8.11% vs GUIDED-T 4.47%.
- **Dependência do compensador:** com ZERO, GUIDED-T k=6 = 18.59% vs STATIC-GREEDY-ZERO 18.54% (final).

## 5. Operações, custos e velocidade (separados)

| Item (por token de decode, k=6) | Valor |
|---|---|
| FLOPs de MLP evitados | 32.2 M (20.2% dos MLPs) |
| Custo da compensação LR32 | 0.23 M FLOPs (38,016 por omissão = 0.72% de um MLP) |
| Custo do roteador (≈26 decisões) | 0.58 M FLOPs |
| **Saldo de operações evitadas** | **31.3 M FLOPs = 19.7% dos FLOPs de MLP, 11.6% do total do modelo** |
| Tempo de um MLP isolado (1 token) | 69.4 µs |
| Tempo do compensador ZERO (1 token) | 0.7 µs |
| Tempo do compensador MEAN (1 token) | 1.0 µs |
| Tempo do compensador LR8 (1 token) | 18.6 µs |
| Tempo do compensador LR16 (1 token) | 17.8 µs |
| Tempo do compensador LR32 (1 token) | 16.7 µs |
| Roteador GUIDED-T-LR32@k6: tempo por decisão com sincronização | 563 µs (26 sincronizações/token, 11,390 parâmetros) |
| Roteador GUIDED-DL-LR32@k6: tempo por decisão com sincronização | 555 µs (26 sincronizações/token, 11,390 parâmetros) |

**Latência e velocidade real (execução condicional, decode de 63 tokens, mediana de repetições pareadas):**

| Política | ms/token | p95 | tok/s | speedup vs FULL [IC95] | omissões/token |
|---|---|---|---|---|---|
| FULL | 3.74 | 3.78 | 268 | 1.000 [1.000, 1.000] | 0.00 |
| STATIC-GREEDY-ZERO@k6 | 3.65 | 3.73 | 274 | 1.020 [1.014, 1.029] | 6.00 |
| STATIC-GREEDY-MEAN@k6 | 3.63 | 3.71 | 275 | 1.026 [1.021, 1.035] | 6.00 |
| STATIC-GREEDY-LR8@k6 | 3.69 | 3.75 | 271 | 1.008 [0.995, 1.019] | 6.00 |
| STATIC-GREEDY-LR16@k6 | 3.72 | 3.77 | 269 | 1.000 [0.997, 1.009] | 6.00 |
| STATIC-GREEDY-LR32@k6 | 3.67 | 3.76 | 272 | 1.009 [0.998, 1.017] | 6.00 |
| GUIDED-T-LR32@k6 | 19.10 | 21.73 | 52 | 0.194 [0.178, 0.202] | 8.04 |
| GUIDED-DL-LR32@k6 | 20.07 | 21.04 | 50 | 0.187 [0.180, 0.200] | 7.22 |

## 6. Compensadores (R² na validação, média das camadas 2–27)

| ZERO | MEAN | LR8 | LR16 | LR32 | LINEAR_FULL_RANK_ref |
|---|---|---|---|---|---|
| -0.066 | -0.000 | 0.199 | 0.249 | 0.310 | 0.508 |

