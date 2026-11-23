# VETOR v0.3 — Especificação matemática (Quality-First Adaptive Inference)

**Versão:** 0.3 · **Data:** 23/11/2026 · **Baselines imutáveis:** tags `v0.1` e `v0.2`
**Escrito antes de qualquer experimento de treino da v0.3.** Alterações posteriores vão para a seção 12, com data e justificativa.

## 1. Pergunta

> Podemos fazer um modelo pequeno responder melhor, não necessariamente realizando mais cálculos, mas escolhendo e corrigindo melhor os cálculos que realiza?

A partir da v0.3, o objetivo principal é **qualidade em tarefas verificáveis**, com velocidade próxima à original (≥ 98% do throughput de decode do FULL). Reduzir operações deixa de ser fim e passa a ser meio: a economia pode ser realocada.

## 2. O que já sabemos (v0.1–v0.2) e o que isso implica

1. **Há redundância por token.** Um guia de ~11 mil parâmetros substitui 23% dos MLPs elegíveis com +4,5% de NLL (v0.2).
2. **Decidir no host custa caro.** Uma decisão por camada exige uma sincronização GPU→CPU (~0,56 ms), e o GUIDED-T real fica 5× mais lento. Na v0.3, **nenhuma decisão discreta poderá exigir sincronização por camada**.
3. **O decode do 135M em PyTorch/MPS é limitado pelo despacho na CPU:** cada operação extra custa ~5–10 µs, independentemente dos FLOPs.
4. **Orçamento de latência medido (23/11/2026, M4 Max, float32):**

| Acréscimo por token (sem sincronização) | Throughput relativo |
|---|---|
| correção de posto 8 em ≤ 8 camadas | 98,8–100% (ruído ~1%) |
| correção de posto 8 com portão suave em 4 camadas | 98,4% |
| correção de posto 8 em 30 camadas | 93,7% |

**Assimetria estrutural:** um adapter LoRA nos pesos lineares pode ser **fundido** aos pesos depois do treino, com custo de latência zero. Uma correção **dependente do token** (adaptativa) não pode ser fundida. Portanto, o vetor-guia só agrega valor se a adaptação por token gerar mais qualidade do que um LoRA fundido de mesmo número de parâmetros, treinado com os mesmos dados e a mesma perda. Essa é a hipótese central a refutar ou confirmar.

## 3. Modelo geral

Para a camada `ℓ`, o token `t` e `z = RMSNorm_post(u_{ℓ,t})`:

```
h_{ℓ+1,t} = u_{ℓ,t} + m_{ℓ,t}·F_ℓ(z) + (1 − m_{ℓ,t})·C_ℓ(z) + a_{ℓ,t}·G_ℓ(z)

F_ℓ : MLP original (congelado)
C_ℓ : compensador barato (v0.2: MEAN ou LR-r), congelado ou treinável
G_ℓ : correção residual treinável de posto baixo   G_ℓ(z) = B_ℓ σ(A_ℓᵀ z),  A_ℓ ∈ R^{d×r}, B_ℓ ∈ R^{r×d}
m_{ℓ,t} ∈ {0,1} : executar ou substituir o MLP
a_{ℓ,t} ∈ [0,1] : intensidade da correção (portão)
g_{ℓ,t} = φ_ℓ(z) ∈ R^{16} : estado do vetor-guia (projeção da v0.2)
```

**Regra de execução sem sincronização (restrição dura):**

- `a_{ℓ,t}` é **contínuo**, calculado no dispositivo e multiplicado diretamente, sem leitura no host: `a = σ(wᵀ g + b_ℓ)`.
- `m_{ℓ,t}` discreto só pode ser decidido **sem leitura por camada**, de uma de duas formas:
  - (i) estático por camada;
  - (ii) por sequência, decidido uma vez após o prefill (1 sincronização por resposta, contabilizada no TTFT).
- Decisões com atraso ≥ 2 tokens podem ser lidas sem esperar a GPU, porque o throttle de eventos da v0.1 já garante que esses valores estão prontos. Essa variante é exploratória (v0.2 mostrou que o atraso custa qualidade).

`G_ℓ` só existe em um conjunto `S` de camadas, com `|S| ≤ 8`, por causa do orçamento da seção 2.

## 4. Arquiteturas candidatas

| | Componentes treináveis | Adaptativo? | Pode ser fundido? |
|---|---|---|---|
| **A — Adaptive Residual Correction** | `G_ℓ` em `S` e portão `a_{ℓ,t}` do guia | sim (portão contínuo por token) | não |
| **B — Compute Reallocation** | `m_ℓ` estático ou por sequência com `C = MEAN`; o tempo economizado paga `G_ℓ` em outras camadas | parcial | não |
| **C — Quality-Aware Routing** | o guia prevê o **ganho** esperado de cada alternativa {F, C, F+G}, e não só o risco | sim | não |
| **D — Guided Knowledge Distillation** | qualquer uma acima, treinada com perda de destilação de um professor maior; professor só no treino | — | — |

- **A:** a correção estática A₀ (`a ≡ 1`) é o controle direto.
- **B:** cada MLP substituído por MEAN economiza ~18 µs medidos (v0.1–v0.2), que pagam ~2 correções. Pergunta: o ganho das correções supera a perda das substituições?
- **C:** é a generalização de A e B, com a alternativa escolhida por um portão contínuo (A) ou por decisão por sequência (B).
- **D — professor:** `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Apache-2.0), da **mesma família e com tokenizador idêntico ao do aluno** (verificado: vocabulário de 49.152 itens, tokenização e template de chat iguais). Isso permite destilação logit a logit.

## 5. Função objetivo

```
L = L_task + α·L_distill + β·L_latency + γ·L_stability

L_task     = −(1/|Y|) Σ_{t∈Y} log p_θ(y_t | y_<t)                  (tokens de resposta do conjunto de treino)
L_distill  = (1/|Y|) Σ_{t∈Y} τ² · KL( softmax(z_T/τ) ‖ softmax(z_θ/τ) )   (professor T, temperatura τ)
L_latency  = Σ_{ℓ∈S} c_ℓ · E_t[a_{ℓ,t}]   +   Σ_ℓ c_F · E[m_ℓ]       (c_ℓ, c_F em µs MEDIDOS, não FLOPs)
L_stability = E_{x∈R} KL( p_FULL(·|x) ‖ p_θ(·|x) )  +  λ_G Σ_ℓ ‖G_ℓ‖²   (R = texto genérico de referência)
```

- `L_stability` mede a divergência em relação ao modelo original **em texto genérico fora das tarefas**. Ela não mede qualidade: serve para evitar deriva e regressões fora do domínio de treino.
- **Pareto:** `β` é varrido em uma grade, e cada ponto (qualidade de validação, throughput medido) é registrado. A configuração final é o ponto de maior qualidade com throughput ≥ 98%, escolhido **na validação**.
- **FLOPs e latência** são medidos e relatados separadamente. `L_latency` usa os custos em µs medidos no MPS.

## 6. Controles obrigatórios e o que cada um isola

| # | Controle | Treino | Isola |
|---|---|---|---|
| 1 | FULL | nenhum | referência |
| 2 | VETOR v0.2 (GUIDED-T LR32, k = 6) | — | histórico |
| 3 | Correção estática A₀ (`a ≡ 1`, mesmos `S`, `r`) | mesma perda e dados | valor do **portão adaptativo** |
| 4 | Correção adaptativa A (portão pelo guia) | mesma perda e dados | arquitetura VETOR |
| 5 | LoRA (q, v, MLP; posto ajustado ao **mesmo nº de parâmetros** de 4), fundido | mesma perda e dados | adapter convencional |
| 6 | Destilação convencional = 5 treinado só com `L_distill` | mesmos dados | ganho de destilação |
| 7 | VETOR com realocação (B) | mesma perda e dados | valor da realocação |

- O **ganho atribuível ao vetor-guia** é definido como (4 − 3) e (4 − 5), ambos pareados por exemplo.
- O **ganho de dados/treino** é (5 − 1).
- O **ganho de destilação** é (6 − 1), contra o mesmo modelo treinado só com `L_task`.

## 7. Custos esperados (antes do treino)

| Arquitetura | Parâmetros treináveis (r = 8, \|S\| = 8) | FLOPs extras/token | Latência esperada |
|---|---|---|---|
| A₀ / A | 8·(2·576·8) ≈ 73,7 mil (+ guia ≈ 9,2 mil) | ≈ 0,15 M (0,06% do modelo) | 98,4–99% |
| LoRA (mesmo nº de parâmetros) | ≈ 74 mil | 0 após fusão | 100% |
| B (4 MLPs → MEAN, 8 correções) | ≈ 74 mil | −21 M + 0,15 M | ≈ 100% |
| Professor (só no treino) | — | — | — |

## 8. Dados

- **Avaliação:** 22 tarefas do lm-evaluation-harness, em EN e PT-BR, nas categorias raciocínio, conhecimento, interpretação, veracidade, matemática, instruções, programação e português. Cada tarefa é dividida em DEV e FINAL por hash (`configs/v3_eval_manifest.json`, commit `4acb3dd`). A metade FINAL só pode ser lida com `allow_final=True`.
- **Treino:** dados genéricos de instrução (prompts e respostas), **disjuntos das tarefas de avaliação**. A fonte e os hashes serão registrados no plano antes do treino.
  - Os splits de treino das próprias tarefas **não** serão usados no regime principal, para que o ganho reflita qualidade geral, e não ajuste a benchmark.
  - Um regime "in-domain" com esses splits pode ser relatado à parte, como exploratório.
- **Conhecimento novo:** a destilação transfere comportamento do professor. Não se assume que o aluno aprenda fatos ausentes do seu pré-treino. Isso entra na interpretação dos resultados de conhecimento.

## 9. Riscos e prior honesto (registrado antes dos experimentos)

- **Pisos de desempenho:** no 135M, várias tarefas ficam no nível do acaso (por exemplo, Belebele PT = 25,5% na dev, com 4 alternativas). Ganhar 5 pontos em média exige tarefas com margem. A regra para tarefas aplicáveis está no plano.
- **Prior:** com ~74 mil parâmetros extras e o modelo congelado, é **improvável** um ganho médio de +5 p.p. em tarefas gerais. Um ganho de 0,5–2 p.p., concentrado em seguir instruções e em veracidade, é o cenário mais provável. Também é provável que o LoRA de mesmo tamanho empate com a correção adaptativa. Esses priors servem para calibrar a interpretação, não para alterar critérios.
- **Destilação vs. arquitetura:** se só os controles com destilação melhorarem, o resultado é "destilação ajuda", e não "o VETOR ajuda".

## 10. Critérios

Os critérios congelados estão em `PLANO_EXPERIMENTAL_V0.3.md` (seção 3). São os critérios preliminares da diretriz, operacionalizados sem alteração de metas.

## 11. Entregáveis

Especificação (este documento), plano experimental, módulos em `src/vetor/v3/`, scripts `scripts/v3_*.py`, testes `tests/test_v3_*.py`, gráficos de qualidade × throughput, análise de custo por arquitetura, relatório comparativo FULL / v0.2 / v0.3 e pull request da v0.3.

## 12. Registro de alterações

*(vazio no momento do pré-registro)*
