# VETOR v0.4 — Reprodução controlada da v0.2 (Fase 1)

**Data:** 24/11/2026 · **Modelo:** SmolLM2-135M-Instruct (`12fd25f7`), PyTorch MPS, float32 · **Script:** `scripts/v4_02_reproduce_v02.py` · **Resultados brutos:** `results/v0.4/repro/`

**Conclusão: a v0.2 se reproduz, e o portão da Fase 1 foi cumprido.**

- **(a)** A avaliação congelada reproduz o conjunto final da v0.2 com diferença < 0,00001 p.p.
- **(b)** Os roteadores retreinados com 3 sementes superam o STATIC-GREEDY com folga, em teste pareado.
- **(c)** A divergência entre teacher forcing e geração livre é real, foi explicada e passa a ser controlada por uma política de orçamento.

## 1. Avaliação congelada (portão a)

Os artefatos publicados da v0.2 foram reavaliados no conjunto final da v0.2, com o mesmo protocolo e a cópia compacta dos compensadores (`results/v0.2/compensators_f32.pt`). Isso prova que a reprodução pública funciona sem o arquivo original de 248 MB.

| Política (k = 6) | Reproduzido | Relatado na v0.2 | Diferença |
|---|---|---|---|
| GUIDED-T-LR32 | 4,4672% | 4,4672% | < 1e-5 p.p. |
| GUIDED-DL-LR32 | 8,1141% | 8,1141% | < 1e-5 |
| STATIC-GREEDY-LR32 | 10,8628% | 10,8628% | < 1e-5 |
| RANDOM-LR32 | 15,8236% | 15,8236% | < 1e-5 |
| GUIDED-T-ZERO / GUIDED-DL-ZERO / STATIC-GREEDY-ZERO / RANDOM-ZERO | — | — | < 1e-5 |

Na primeira tentativa, o RANDOM diferiu em ~0,07–0,5 p.p. A causa era a ordem dos sorteios: a v0.2 sorteava os conjuntos de todos os k em sequência (3, 5, 6, 8, 10). Reproduzida essa ordem, a diferença zerou. Não havia erro na v0.2.

**Professor:** os rótulos Δ_N da rodada 0, regerados em 16 janelas (4.992 células), coincidem exatamente com os salvos (erro máximo 0,0).

## 2. Variabilidade entre sementes (portão b)

O GUIDED-T e o GUIDED-DL foram retreinados a partir dos rótulos salvos das rodadas 0–2, com a configuração escolhida na v0.2:

- GUIDED-T: N = 16, MSE;
- GUIDED-DL: N = 8, pinball q = 0,3.

Em cada semente, τ foi calibrado em k = 6. Diferenças com IC95 por bootstrap pareado (10.000 reamostragens).

| Roteador | ΔNLL val | ΔNLL final v0.2 | vs STATIC-GREEDY (final) |
|---|---|---|---|
| GUIDED-T s0 | 4,00% | **4,467%** (idêntico ao publicado) | −6,40 p.p. [−7,07; −5,74] |
| GUIDED-T s1 | 4,19% | 4,31% | −6,56 [−7,25; −5,89] |
| GUIDED-T s2 | 3,86% | 4,55% | −6,31 [−6,99; −5,65] |
| GUIDED-DL s0 | 7,84% | 7,98% | −2,88 [−3,64; −2,12] |
| GUIDED-DL s1 | 8,65% | 8,52% | −2,34 [−3,13; −1,57] |
| GUIDED-DL s2 | 8,23% | 8,67% | −2,19 [−2,97; −1,43] |
| *Controles:* STATIC-GREEDY-LR32 | 10,46% | 10,86% | — |
| RANDOM-LR32 (3 sementes) | 15,85–16,08% | 15,82–16,02% | — |

O resultado da v0.2 não depende da semente: a dispersão entre sementes é de ±0,15 p.p., contra uma vantagem de ~6,4 p.p. sobre o melhor controle. A vantagem de decidir durante a travessia (T) sobre decidir com atraso (DL) também se repete nas 3 sementes.

## 3. Divergência entre teacher forcing e geração livre (portão c)

Na v0.2, o τ calibrado em texto natural deu 6,06 omissões por token na avaliação de qualidade e 8,04 no benchmark de geração. Medições com o roteador publicado (GUIDED-T-LR32 r2, τ de k = 6) em 128 janelas de validação que não entraram na calibração:

| Medição | Omissões/token |
|---|---|
| Texto natural, modo em lote | 6,03 |
| Texto natural, **decode real** com tokens forçados | 6,12 |
| Geração livre, **decode real** (texto gerado pela própria política) | 7,72 |
| Mesmo texto gerado, reavaliado no modo em lote | 7,72 (idêntico) |
| Texto gerado pelo FULL, modo em lote | 7,71 |

**Causa: mudança de distribuição do texto, não erro de implementação.**

- Decode real e modo em lote concordam no mesmo texto: 6,12 contra 6,03 em texto natural, 7,72 contra 7,72 em texto gerado.
- O texto que o próprio modelo gera em greedy é mais previsível. A entropia média do FULL é 2,19 nats no texto gerado, contra 3,24 nats no texto natural.
- O roteador prevê menos risco nesse texto e omite mais, sobretudo nas camadas iniciais. A camada 3, por exemplo, sobe de 64% para 78% das posições.
- A taxa também cresce ao longo da geração: 7,08 omissões no 1º quartil da continuação e 8,34 no 4º.

**Outro achado:** em 46% dos tokens de texto natural, o roteador atinge o teto por token (`k_max = 9`), e em 10% não omite nada. As omissões se concentram em tokens "fáceis". É isso que o diferencia de uma política estática.

### Política de orçamento adotada (vale para a v0.4 inteira)

O τ é calibrado **por regime**, com a mesma média alvo k:

- **τ_tf:** texto natural, para NLL e tarefas de verossimilhança;
- **τ_gen:** continuações greedy do FULL a partir dos prompts de calibração, para geração livre, tarefas de geração e velocidade.

A taxa efetiva é sempre medida e relatada.

**Verificação no 135M:**

- O τ_gen calibrado em texto gerado pelo FULL produziu **6,03** omissões por token na geração livre real da própria política (alvo 6).
- O mesmo τ_gen aplicado a texto natural dá 3,93 omissões por token e ΔNLL de 1,77%.

### Qualidade na geração livre real, mesmo orçamento (k = 6)

128 prompts de validação, 64 tokens greedy, decode real:

| Política | Omissões/token | KL(FULL‖política) na própria trajetória | Concordância com o FULL | Prefixo idêntico (tokens) | Respostas degeneradas |
|---|---|---|---|---|---|
| FULL | 0 | 0 | 1 | 64 | 33,6% |
| **GUIDED-T (τ_gen)** | 6,03 | **0,101** | **21,2%** | **10,0** | 35,2% |
| GUIDED-T (τ_tf, como na v0.2) | 7,72 | 0,173 | 17,1% | 6,8 | 37,5% |
| STATIC-GREEDY-LR32 | 6,00 | 0,305 | 11,3% | 4,1 | 33,6% |
| RANDOM-LR32 | 6,00 | 0,263 | 10,6% | 4,1 | 63,3% |

- Na geração autoregressiva real, com o mesmo orçamento, o roteador mantém a vantagem: KL 3× menor que o STATIC-GREEDY e um prefixo idêntico ao do FULL 2,4× mais longo.
- Mesmo assim, a geração do 135M diverge do FULL depressa: 21% de concordância.
- O greedy do próprio FULL já degenera muito nesse modelo pequeno (33,6% de respostas com ≥ 50% de 4-gramas repetidos). Por isso, essa métrica só serve como comparação relativa.

## 4. Decisão

O portão da Fase 1 está cumprido e a implementação de referência está validada. Segue a Fase 2 (SmolLM2-1.7B-Instruct), com a política de orçamento por regime.
