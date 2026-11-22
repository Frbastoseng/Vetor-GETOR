# VETOR v0.2 — Especificação matemática e protocolo (pré-registro)

**Versão:** 0.2 · **Data:** 22/11/2026 · **Baseline imutável:** tag `v0.1` (resultados em `results/`, não alterados)
**Escrito antes dos experimentos da v0.2.** Mudanças posteriores a este documento serão registradas na seção 11 com data e justificativa.

## 1. Objetivo

Verificar se um vetor-guia adaptativo consegue **identificar e substituir** computações redundantes durante a passagem de um token por um Transformer denso pré-treinado congelado (SmolLM2-135M-Instruct, revisão `12fd25f7`).

A v0.1 não é tratada como refutação do conceito. Ela mostrou três falhas de *desenho*:

- o rótulo do professor media o efeito isolado de uma camada em um token;
- a omissão devolvia zero;
- a decisão era atrasada: usava `z` do token anterior.

A v0.2 ataca cada uma dessas falhas separadamente, para que o efeito de cada mudança seja mensurável.

## 2. Notação

Para a camada `l ∈ {0..L-1}` (L = 30) e a posição `t`:

```
u[l,t] = h[l,t] + Attn_l(RMSNorm(h[l,t]), KV)            (sempre executado)
z[l,t] = RMSNorm_post(u[l,t])                             (entrada do MLP, d = 576)
h[l+1,t] = u[l,t] + m[l,t]·MLP_l(z[l,t]) + (1 − m[l,t])·C_l(z[l,t])
```

`m[l,t] ∈ {0,1}` (1 = executar) e `C_l` é o **compensador** da camada `l`. Camadas protegidas: `l ∈ {0,1,28,29}` ⇒ `m = 1`. As camadas elegíveis são `E = {2..27}`, com |E| = 26. O prefill nunca é omitido; as decisões valem só para posições de decodificação.

## 3. Compensação residual

| Variante | `C_l(z)` | FLOPs/token | Parâmetros/camada |
|---|---|---|---|
| ZERO (v0.1) | `0` | 0 | 0 |
| MEAN | `μ_l = E[MLP_l(z)]` | `d` (soma) | `d` |
| LR-r, r ∈ {8,16,32} | `U_l (V_lᵀ z) + c_l` | `2·d·r + d` | `2·d·r + d` |

Para comparação, um MLP custa `6·d·d_ff = 5.308.416` FLOPs/token. O LR-32 custa 37.440 FLOPs (0,7% do MLP).

**Ajuste (só o compensador é treinado; o LLM fica congelado):** regressão de posto reduzido com ridge, em forma fechada, sobre pares `(z, MLP_l(z))` coletados do modelo FULL em textos de **treino**:

```
W* = argmin_{posto(W) ≤ r} E‖MLP_l(z) − W(z − z̄) − ȳ‖² + λ‖W‖²
W_ols = (Σ_zz + λI)⁻¹ Σ_zy,    Σ_ŷŷ = W_olsᵀ Σ_zz W_ols
V_r = top-r autovetores de Σ_ŷŷ,   W_r = W_ols V_r V_rᵀ
⇒ C_l(z) = (z − z̄) W_r + ȳ   (fatorado: U = W_ols V_r [d×r], Vᵀ = V_rᵀ)
```

As estatísticas suficientes (`Σ_zz`, `Σ_zy`, médias) são acumuladas em float64, sem guardar os pares. O λ é escolhido na validação. A mudança de distribuição (`z` sob omissões anteriores) é medida, e o reajuste sob a política fica registrado como variante (`LR-r/iter`).

## 4. Efeito conjunto e acumulado (novo professor)

Seja uma janela com prompt de `P` tokens e continuação verdadeira `y_1..y_n`. As posições de decodificação são `D = {P, ..., P+n−2}`, e a máscara é `M ∈ {0,1}^{D×L}`. Para cada posição `p`, `q_M(p)` é a distribuição do próximo token sob `M` (teacher forcing com KV).

```
Qualidade da janela:   KL_tot(M) = Σ_{p∈D} KL(q_FULL(p) ‖ q_M(p))
                       NLL(M)    = −Σ_{p∈D} log q_M(p)[y_{p−P+1}]
Efeito marginal acumulado da decisão (t,l) sob a máscara-base M_b:
    Δ_N(t,l | M_b) = Σ_{p=t}^{min(t+N−1, max D)} KL(q_{M_b}(p) ‖ q_{M_b ⊕ (t,l)}(p)),   N ∈ {8, 16, 32}
```

`M_b ⊕ (t,l)` liga a omissão em `(t,l)` e mantém fixas todas as outras decisões (contrafactual com as demais congeladas). O rótulo inclui:

- a **acumulação**, pelo efeito em `p > t` via KV;
- a **interação**, porque é medido sob as omissões reais de `M_b`, e não sob FULL.

O ΔNLL acumulado correspondente também é guardado.

**Política-base iterativa (estilo DAgger):** a rodada 0 usa `M_b = FULL` (efeito isolado, porém acumulado). A rodada `r+1` usa `M_b` = as decisões do roteador da rodada `r`, e os rótulos de todas as rodadas são unidos. São R ≤ 2 rodadas, escolhidas na validação.

## 5. Oráculos diagnósticos (não implementáveis em produção)

- **ORACLE-GREEDY-SEQ:** para cada janela, uma busca gulosa adiciona, uma a uma, a camada `l ∈ E` que minimiza `KL_tot` da própria janela, até k camadas; a máscara é constante ao longo da sequência. Ela usa a continuação da própria janela, portanto é um limite superior para máscaras por sequência.
- **ORACLE-GREEDY-TOKEN (amostrado):** igual, mas por posição. A busca gulosa por camada usa `Δ_N` sob a máscara corrente, em um subconjunto de posições. É só diagnóstico de viabilidade.

**STATIC-GREEDY (controle implementável, novo):** a mesma busca gulosa, mas minimizando o `KL_tot` médio sobre as janelas de **treino**. Ele substitui o STATIC da v0.1 como controle mais forte, e o STATIC da v0.1 continua relatado.

Pergunta de viabilidade: se nem o ORACLE-GREEDY-SEQ, com o melhor compensador, atinge ΔNLL ≤ 5% com 6 omissões, nenhum roteador por sequência atingirá. Essa conclusão será relatada.

## 6. Roteador durante a travessia (GUIDED-T)

Decisão tomada **antes do MLP da camada `l` no token atual `t`**, usando apenas informação causal:

```
φ(t,l) = [ LN(z[l,t]) ; e[l] ; s(t,<l) ; n(t,<l)/|E| ]
    s(t,<l) ∈ {0,1}^L : decisões já tomadas no token t para camadas < l (zeros para ≥ l)
    n(t,<l)           : número de omissões já feitas no token t
g = tanh(P·LN(z) + e[l] + S·s + a·n),   P: d→16, S: L→16
ŷ(t,l) = wᵀg + b[l]   ≈ log(Δ_N(t,l | M_b) + ε)   (regressão ou quantil, escolhido na validação)
m[l,t] = 0  ⇔  l ∈ E  ∧  ŷ(t,l) ≤ τ  ∧  n(t,<l) < k_max
```

`τ` é calibrado na **validação** para atingir a taxa média-alvo ρ, e `k_max = ⌈1,5·ρ·|E|⌉`. São cerca de 9,8 mil parâmetros, mais o compensador.

**Comparação com o roteamento atrasado:** o GUIDED-D1 da v0.1 é retreinado com os rótulos novos e usa o mesmo compensador e a mesma taxa efetiva. A diferença isola o efeito de decidir antes do MLP atual contra decidir com `z` do token anterior.

**Avaliação de qualidade em um forward.** A decodificação com KV e máscara em `D` é matematicamente idêntica a um forward da sequência `P+n−1` com a substituição aplicada só nas posições `D`, porque a atenção é causal. O GUIDED-T também é exato nesse forward: na camada `l`, `z[l,·]` já incorpora as decisões das camadas `< l` de cada posição. Esse modo calcula o MLP e o substitui, então **serve só para medir qualidade**. A equivalência com o decode de skip real é verificada por teste automático (|Δ log q| ≤ 1e-3).

## 7. Orçamentos e controles

| k (omissões/token) | 3 | 5 | **6** | 8 | 10 |
|---|---|---|---|---|---|
| % dos elegíveis | 11,5 | 19,2 | **23,1** | 30,8 | 38,5 |

**k = 6 é o primeiro orçamento que cumpre o critério de "≥ 20% dos MLPs elegíveis"**; na v0.1, o k = 5 correspondia a 19,2%. As políticas com taxa variável (GUIDED-T) são comparadas na **mesma taxa média efetiva** medida.

Controles, todos com o mesmo compensador:

- RANDOM, sorteado por sequência;
- STATIC (v0.1);
- STATIC-GREEDY;
- GUIDED-S e GUIDED-D1 da v0.1, só com ZERO, como baseline histórico.

## 8. Métricas, relatadas separadamente

1. **Qualidade:** ΔNLL relativo (IC95 bootstrap por janela), KL médio/p90, concordância top-1 e greedy.
2. **Operações evitadas:** FLOPs de MLP evitados − FLOPs do compensador − FLOPs do roteador, por token.
3. **Custo de compensação:** FLOPs e µs medidos por chamada.
4. **Overhead do guia:** FLOPs, µs por decisão e sincronizações por token.
5. **Latência real:** ms/token de decode, mediana e p95, com execução condicional **real** (sem calcular e zerar).
6. **Velocidade de geração:** tok/s, speedup pareado contra FULL com IC95.

No MPS/eager, o GUIDED-T com skip real exige uma sincronização GPU→CPU por camada. Sua latência será medida e relatada, mas **a v0.2 não exige ganho de velocidade do GUIDED-T**. A pergunta desta fase é se existe vantagem *matemática* sobre o roteamento atrasado. A versão acelerada exige decisão no dispositivo e fica para trabalho futuro.

## 9. Critério de sucesso (inalterado em relação à v0.1)

Com **k ≥ 6** (≥ 20% dos MLPs elegíveis):

- aumento relativo de NLL **≤ 5%**;
- ganho de tok/s de geração **≥ 5%**;
- desempenho **superior** a RANDOM e STATIC-GREEDY na mesma taxa, com IC95 sem sobreposição.

Perguntas intermediárias da v0.2, relatadas mesmo sem sucesso final:

| Pergunta | Evidência |
|---|---|
| Q1: compensar reduz o dano? | ΔNLL de STATIC-GREEDY: LR-r contra ZERO, mesmo k |
| Q2: existe combinação viável? | ORACLE-GREEDY-SEQ/TOKEN com ΔNLL ≤ 5% em k = 6 |
| Q3: o professor novo ajuda? | GUIDED-D1 com rótulo novo contra rótulo v0.1 |
| Q4: decidir na travessia ajuda? | GUIDED-T contra GUIDED-D1, mesmo rótulo, compensador e taxa |

## 10. Dados e validade

- Splits pré-registrados em `configs/v2_splits_manifest.json`, com hash por artigo:
  - **train:** WT2-train + 1.000 artigos WT103;
  - **val:** WT2-val + 140 artigos WT103;
  - **final:** 200 artigos WT103 fora do WT2, lidos só com `allow_final=True`;
  - **v01test:** o teste da v0.1, só para comparação histórica.
- Toda escolha (compensador, λ, r, N, alvo, τ, rodadas, arquitetura) é feita em train/val. O conjunto final é avaliado **uma vez**, com a configuração congelada num arquivo de decisão versionado *antes* da avaliação.
- Os resultados da v0.2 vão para `results/v0.2/`. Os arquivos da v0.1 não são alterados.
- O modelo de 135M permanece até que a v0.2 mostre se o novo guia melhora a relação qualidade × operações.

## 11. Registro de alterações deste protocolo

*(vazio no momento do pré-registro)*
