# VETOR v0.4 — Especificação (pré-registro)

**Data:** 24/11/2026 · **Branch:** `v0.4`, criada a partir da `v0.3` (que contém a `main` com a v0.2). A v0.3 é a base porque a bateria de tarefas (`src/vetor/v3/bench.py`) é reaproveitada. Os resultados históricos da v0.1, v0.2 e v0.3 não são alterados.

**Escrito antes de qualquer treino da v0.4.** Alterações posteriores vão para a seção 9 de `PLANO_EXPERIMENTAL_V0.4.md`, com data e justificativa.

## 1. Pergunta

Um vetor-guia treinável identifica operações redundantes durante a inferência de um Transformer pré-treinado, permitindo executá-lo **mais rápido**, **sem degradação relevante** de qualidade e **com vantagem** sobre estratégias convencionais?

A v0.4 não introduz arquitetura nova. Ela testa o mecanismo da v0.2 em três passos:

1. reprodução no 135M;
2. generalização para o SmolLM2-1.7B-Instruct;
3. conversão em ganho real de velocidade, só se os passos anteriores justificarem.

## 2. Mecanismo (idêntico ao da v0.2)

Para a camada `l` e a posição `t`:

```
u[l,t]   = h[l,t] + Attn_l(RMSNorm(h[l,t]), KV)          (sempre executada)
z[l,t]   = RMSNorm_post(u[l,t])                           (entrada do MLP)
h[l+1,t] = u[l,t] + m[l,t]·MLP_l(z[l,t]) + (1 − m[l,t])·C_l(z[l,t])
```

- **Compensador:** `C_l(z) = ((z − z̄_l) V_l) U_l + ȳ_l`, de posto `r`. É ajustado em forma fechada por regressão de posto reduzido com ridge sobre pares `(z, MLP_l(z))` do modelo FULL, em textos de treino. As variantes são ZERO, MEAN e LR-r.
- **Roteador GUIDED-T** (decide antes do MLP da camada `l`, no token atual):

  ```
  ŷ = wᵀ tanh(P·LN(z[l,t]) + e[l] + S·s(t,<l) + a·n(t,<l)/|E|) + b[l]
  omitir ⇔ l ∈ E  ∧  ŷ ≤ τ  ∧  n(t,<l) < k_max
  ```

  `s(t,<l)` são as decisões já tomadas no token para as camadas anteriores, e `n(t,<l)` é quantas omissões já houve.
- **GUIDED-DL:** o mesmo roteador, mas recebendo `z[l,t−1]`. Isola o efeito de decidir com informação atrasada.
- **Professor:** efeito acumulado `Δ_N(t,l | M_b) = Σ_{p=t}^{t+N−1} KL(q_{M_b}(p) ‖ q_{M_b⊕(t,l)}(p))`, sob a máscara-base `M_b`. A rodada 0 usa FULL; as rodadas seguintes usam as decisões do roteador anterior, no esquema DAgger, com R ≤ 2.
- **Camadas protegidas:** as 2 primeiras e as 2 últimas. No 1.7B (L = 24) as elegíveis são `E = {2..21}`, |E| = 20. O prefill nunca é omitido.

**Correção de implementação (auditoria):** o treino do roteador fixava `n_el = 26`, o |E| do 135M. Agora o valor é um parâmetro, e no 1.7B se usa `n_el = 20`. O comportamento da v0.2 está preservado.

## 3. Custos por token (contados pelo PyTorch, `torch.utils.flop_counter`)

| Item | 135M (d = 576, d_ff = 1536, L = 30) | 1.7B (d = 2048, d_ff = 8192, L = 24) |
|---|---|---|
| 1 MLP | 5,31 M | 100,7 M |
| Compensador LR-32 (multiplicações) | 73,7 k (1,39% do MLP) | 262 k (0,26% do MLP) |
| Roteador, 1 decisão (multiplicações) | 19,4 k | 66,3 k |
| Token de decode completo (contexto 48) | 272 M (MLPs = 58%) | 3,43 G (MLPs = 70%) |

A v0.2 relatou 38.016 FLOPs para o LR-32. O valor correto é ~74 mil, porque só uma das duas multiplicações foi contada. Os detalhes estão em `AUDITORIA_V0.4.md`.

**Teto teórico no 1.7B** com k = 5 omissões por token (25% dos elegíveis): 5 × 100,7 M = 503 M FLOPs, que são **14,7% do token**, antes de descontar compensador e roteador. Medição preliminar de planejamento (MPS, fp32): os 24 MLPs somam ~60–66% do tempo de decode, contra os 2% de ganho real no 135M. É um indicativo, não um resultado. A medição formal é feita na Fase 3.

## 4. Modos de execução e o que cada um pode provar

| Modo | Uso | Pode provar economia? |
|---|---|---|
| Mascarado em lote (calcula MLP e compensador e escolhe) | **só** medir qualidade, com equivalência testada contra o decode real | **Não** (regra 9) |
| Decode real com omissão estática (PyTorch) | teto de velocidade sem roteador | Sim, pelo contador no MLP original |
| Decode real GUIDED-T na v0.2 (1 sincronização por camada) | referência de equivalência | Não serve como implementação final (regra 10) |
| Implementação otimizada com decisão no dispositivo (Fase 4) | velocidade final | Sim, se o MLP não for executado (instrumentação) e não houver leitura síncrona por camada |

**Possibilidade a ser verificada na Fase 3, sem pressupor que funcione:** em decode de 1 token o MLP é limitado pela leitura de pesos. Um kernel Metal que lê a decisão da memória do dispositivo e encerra antes de ler os pesos eliminaria o custo dominante sem sincronizar com a CPU. Ainda assim, o lançamento do kernel continuaria acontecendo. Isso só será construído se o teto medido na Fase 3 for ≥ 5%.

## 5. O que não muda em relação à v0.2

- Janelas: prompt de 64 tokens e continuação de 64.
- Métricas:
  - ΔNLL relativo pela razão de somas;
  - KL contra o FULL;
  - concordância top-1.
- Esquema do roteador (dimensão 16) e do professor.
- Grade de alvos e perdas: N ∈ {8, 16, 32}, MSE ou pinball (q = 0,3).
- Calibração de τ por bisseção.
- `k_max = ⌈1,5·k⌉`.
- Controles:
  - STATIC-GREEDY, isto é, a ordem gulosa ajustada no treino;
  - RANDOM, sorteado por janela;
  - versões sem compensação (ZERO).

## 6. Novidades da v0.4, todas de protocolo e nenhuma de arquitetura

1. Novo modelo (1.7B), com compensadores e roteadores treinados do zero. Nada é reaproveitado do 135M.
2. Novos conjuntos de treino, validação e teste final, sem nada examinado antes (seção 6 do plano).
3. Avaliação de qualidade também em tarefas verificáveis e em geração livre, não só NLL (regra 7).
4. Política de orçamento entre teacher forcing e geração livre, definida na Fase 1 antes de qualquer treino no 1.7B (plano, seção 3).
5. Três sementes de treino do roteador.
6. Comparações por bootstrap **pareado da diferença**, em vez de "IC sem sobreposição".
7. Medição de velocidade só com execução condicional real, contra um FULL no **mesmo** backend e precisão.
