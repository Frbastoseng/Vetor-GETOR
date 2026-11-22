# Por que o VETOR v0.1 não atingiu o critério — e o que testar a seguir

Este documento usa **somente evidências de treino e validação** (o teste foi rodado uma única vez, com decisões congeladas). Cada proposta abaixo é uma hipótese com critério definido antes de rodar. Para opinar ou assumir uma delas, abra uma issue do tipo *Proposta de experimento* citando o número (M1, M2, ...).

## 1. Diagnóstico

O guia aprendeu o que o professor ensinou: na validação, ele distingue contextos seguros dentro da mesma camada (AUC 0,72 no guia selecionado; até 0,77 em outras variantes) e, no proxy offline, escolhe camadas com menos KL isolado que o STATIC em 3 de 4 orçamentos. Na decodificação real, porém, **nenhuma política atinge ΔNLL ≤ 5%**, nem mesmo com 3 MLPs omitidos (o STATIC fica em +7%).

O descolamento tem duas causas, medidas em `results/diag_interaction_val.json`:

| conjunto omitido (validação) | Σ KL isolados | KL conjunto (mesmo token) | KL no decode (acumulado) |
|---|---|---|---|
| STATIC k=3 `[12,13,14]` | 0,112 | 0,114 (×1,0) | 0,259 (×2,3) |
| STATIC k=5 `[12,13,14,16,17]` | 0,187 | 0,279 (×1,5) | 0,529 (×1,9) |
| STATIC k=8 `[11–18]` | 0,319 | 0,929 (×2,9) | 2,400 (×2,6) |
| RANDOM k=8 (espalhado) | 0,760 | 1,053 (×1,4) | 1,368 (×1,3) |
| STATIC espalhado k=8 | 0,417 | 0,709 (×1,7) | não medido |

1. **Interação entre camadas.** Omissões *não* são aditivas, como a especificação já alertava. Blocos contíguos de camadas (que o STATIC escolhe, porque as camadas 11–18 são individualmente as mais baratas) custam até ~3× a soma das partes. É por isso que, com k≥8, o STATIC fica **pior que o RANDOM**.
2. **Acumulação no KV cache.** No decode, o MLP pulado altera o estado residual do token, que vira K/V das camadas seguintes e é lido por todos os tokens futuros. O professor mede só o efeito imediato, com o prefixo intacto, e isso subestima o dano em ~2×.
3. **Velocidade.** O decode do SmolLM2-135M em PyTorch/MPS é limitado pela CPU (despacho Python): 3,54 de 3,60 ms/token são CPU. Cada MLP pulado economiza ~18 µs, e o teto pulando os 26 MLPs elegíveis é ~13%. Os FLOPs evitados quase não importam neste regime.

## 2. Propostas (ordem sugerida: barato e informativo primeiro)

| # | Ideia | Ataca | Custo | Critério de sucesso (validação) |
|---|---|---|---|---|
| M1 | **Substituir em vez de zerar**: devolver a média do MLP por camada (ou um mapa linear de baixo posto `z→MLP(z)`, r≈16–32) no lugar de zero | dano por omissão | horas | ΔNLL do STATIC@k5 cai ≥50% sem perder o speedup medido |
| M2 | **Professor de conjuntos**: rotular conjuntos de camadas (KL conjunto) e treinar risco de conjunto, ou penalizar camadas adjacentes no top-k | interação | horas | GUIDED vence STATIC e RANDOM em ΔNLL nos 4 orçamentos |
| M3 | **STATIC-JOINT** (controle mais forte): seleção gulosa de camadas avaliando o efeito *conjunto* e *acumulado* no treino | interação + acumulação | horas | novo baseline; o guia precisa superá-lo |
| M4 | **Professor com rollout**: rótulo = KL acumulado após N tokens decodificados com a camada omitida (N=8–16) | acumulação | ~1 dia de coleta | correlação do rótulo com o KL do decode > a do rótulo de 1 token |
| M5 | **Coleta iterativa (estilo DAgger)**: coletar `z` sob a política guiada e retreinar (seção 9 da especificação) | mudança de distribuição | horas | GUIDED-D1 melhora vs v0.1 no mesmo orçamento |
| M6 | **Orçamento adaptativo**: limiar `tau` em vez de k fixo (pular só quando confiante), comparando com controles na mesma taxa *efetiva* | caudas de risco | horas | curva Pareto acima de STATIC/RANDOM |
| M7 | **Backend sem gargalo de despacho**: `torch.compile`/MLX/llama.cpp com grafo pré-compilado por máscara, ou modelo maior (≥1B) onde o MLP pesa em banda de memória | velocidade | dias | skip de 26 MLPs → ganho ≥ 25% |
| M8 | **Granularidade diferente**: pular blocos inteiros (atenção+MLP) só em tokens "fáceis" (early-exit / mixture-of-depths), com KV aproximado | velocidade + qualidade | dias | comparar na mesma economia de tempo real |

## 3. O que *não* mudar

- Splits por documento, seleção só na validação, controles com a mesma taxa efetiva, sincronização só nas bordas: são o que torna esses números confiáveis.
- O critério de sucesso da especificação (ΔNLL ≤ 5%, ≥20% de MLPs elegíveis, ≥5% de tok/s) não deve ser afrouxado retroativamente.
