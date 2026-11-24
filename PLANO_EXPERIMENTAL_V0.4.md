# VETOR v0.4 — Plano experimental e critérios (pré-registro)

**Data:** 24/11/2026 · Complementa `ESPECIFICACAO_V0.4.md`. Escrito antes de qualquer treino da v0.4. Alterações: seção 9.

## 1. Fases, portões e paradas

| Fase | Conteúdo | Portão para avançar | Se falhar |
|---|---|---|---|
| 0 | Auditoria (`AUDITORIA_V0.4.md`) | todas as verificações automáticas passam | corrigir antes de seguir |
| 1 | Reprodução da v0.2 no 135M (`REPRODUCAO_V0.2.md`) | (a) a avaliação congelada reproduz o ΔNLL final ±0,05 p.p.; (b) roteadores retreinados com 3 sementes superam o STATIC-GREEDY na validação; (c) a divergência TF × geração está explicada e há uma política de orçamento | corrigir a implementação; se (b) falhar, a v0.2 não se reproduz → **Resultado C** |
| 2 | 1.7B: compensadores, professor, roteadores e controles; avaliação final congelada (`VALIDACAO_1B_V0.4.md`) | critérios M1–M4 (seção 2) na avaliação final | **Resultado C** e fim da linha principal |
| 3 | Viabilidade de desempenho (`VIABILIDADE_DESEMPENHO_V0.4.md`) | teto medido de ganho real ≥ 5% (V1) | **Resultado B** sem construir kernel |
| 4 | Implementação com decisão no dispositivo | equivalência com a referência; MLP comprovadamente não executado; zero leituras síncronas por camada | **Resultado B** |
| 5 | Benchmark definitivo | D1–D3 | **Resultado B** |

Ao atingir uma parada, os experimentos da linha principal param e o relatório correspondente é escrito. Nenhuma pesquisa nova começa automaticamente.

## 2. Critérios (congelados)

Todos são medidos no **conjunto final da v0.4**, avaliado uma única vez após o congelamento (seção 7).

**Configuração principal:** GUIDED-T com o compensador escolhido na validação, orçamento k = 5 omissões por token (25% das 20 camadas elegíveis), média das 3 sementes.

**Matemáticos (Fase 2)**

- **M1 — Taxa:** omissões efetivamente executadas ≥ 20% dos MLPs elegíveis (≥ 4,0 por token), medidas no texto avaliado e na geração livre.
- **M2 — NLL:** aumento relativo de NLL ≤ 5%, com o limite **superior** do IC95 também ≤ 5%, no agregado dos 4 domínios do conjunto final.
- **M3 — Vantagem:** ΔNLL(GUIDED-T) − ΔNLL(melhor controle de mesmo orçamento) < 0, com o limite superior do IC95 pareado < 0.
  - O melhor controle é o de menor ΔNLL entre STATIC-GREEDY e RANDOM, com o mesmo compensador e k = 5, escolhido **no próprio conjunto final**, o que favorece o controle.
  - Cada uma das 3 sementes também precisa ter diferença pontual < 0.
- **M4 — Sem regressões graves** (estimativas pontuais, VETOR − FULL):
  - média macro das tarefas verificáveis aplicáveis ≥ −2,0 p.p.;
  - nenhuma categoria aplicável com queda > 5,0 p.p.;
  - na geração livre, a fração de respostas degeneradas (≥ 50% de 4-gramas repetidos) não pode passar a do FULL em mais de 3 p.p.

**Desempenho (Fases 3 a 5)**

- **V1 — Viabilidade:** o speedup medido com omissão estática real de 5 MLPs por token (com compensador, sem roteador), no backend escolhido, precisa ser ≥ 1,05. Se não for, nenhum kernel é construído.
- **D1 — Velocidade:** tok/s de decode do VETOR ÷ tok/s do FULL no **mesmo backend otimizado e mesma precisão** ≥ 1,05, com o limite inferior do IC95 > 1,00.
  - Mediana de razões pareadas, ≥ 15 repetições, ordem alternada.
  - Prompts reais e sintéticos.
- **D2 — Qualidade preservada na implementação otimizada:** equivalência com a referência (diferença máxima de log-probabilidade dentro da tolerância da precisão) e M1, M2 e M4 inalterados.
- **D3 — Honestidade de execução:** o MLP não é executado nos caminhos omitidos (contador ou instrumentação) e não há leitura síncrona de decisão por camada.

**Conclusão**

| Resultado | Condição |
|---|---|
| **A** — sucesso tecnológico | M1–M4 e D1–D3 |
| **B** — sucesso matemático sem ganho operacional | M1–M4, mas V1 ou D1–D3 falham |
| **C** — hipótese não confirmada | M2, M3 ou M4 falham, ou a Fase 1 não reproduz |

Meta exploratória, que não entra em nenhum critério: ganho ≥ 10%.

## 3. Política de orçamento (TF × geração)

Na v0.2, o τ calibrado em texto natural produziu 6,06 omissões por token em teacher forcing e 8,04 na geração livre.

A Fase 1 mede a causa:

- mesmo roteador, decode real com tokens forçados contra modo em lote;
- texto natural contra texto gerado pelo próprio modelo;
- distribuição por camada e por posição.

A partir disso, a política é fixada **antes de qualquer treino no 1.7B** e registrada na seção 9. A opção prevista, se a causa for a mudança de distribuição do texto:

- τ é calibrado **por regime**, com texto natural para métricas em teacher forcing e texto gerado pelo modelo para geração livre e velocidade;
- ambos visam a mesma média k;
- a taxa efetiva é sempre relatada.

Os controles estáticos têm k exato nos dois regimes.

## 4. Experimentos da Fase 2 (1.7B, PyTorch MPS, float32)

1. FULL.
2. STATIC-GREEDY: ordem gulosa por KL médio em janelas de treino, para cada compensador.
3. RANDOM: conjunto sorteado por janela, 3 sementes.
4. GUIDED-T: 3 sementes de treino do roteador.
5. GUIDED-DL: mesmo treino e seleção do GUIDED-T, com 3 sementes.
6. Os itens 2–5 rodam com e sem compensação, ou seja, com ZERO e com o compensador escolhido.
7. ORACLE-GREEDY-SEQ e ORACLE-GREEDY-TOKEN em 16 janelas de validação, só como diagnóstico.

Escolhas, todas na validação:

- **Compensador:** entre ZERO, MEAN, LR16, LR32 e LR64, o de menor ΔNLL do STATIC-GREEDY com k = 5, como na v0.2.
- **Roteador:**
  - alvo N e perda, na grade da v0.2;
  - rodadas DAgger ≤ 2;
  - τ calibrado em 64 janelas de validação.

**Sementes:**

- O professor (rótulos Δ_N) é gerado uma vez por rodada, numa cadeia DAgger da semente 0, e compartilhado.
- As 3 sementes variam inicialização e ordem do treino do roteador e são calibradas separadamente.
- Limitação registrada: a variabilidade dos rótulos não é coberta.

**Fora do escopo, por decisão prévia:** reimplementar o DiffSkip (roteador sobre ΔAttn e adaptador FFN pequeno), que é a arte anterior mais próxima (`docs/LITERATURA_V0.4.md`). Trocar o compensador ou a entrada do roteador seria testar outra arquitetura (regras da v0.4). A comparação com a literatura é qualitativa e fica no relatório.

**Registro da distribuição das omissões:** por camada, por posição na continuação, por janela (histograma de omissões por token) e por domínio.

## 5. Estatística

- **ΔNLL relativo:** `Σ NLL / Σ NLL_FULL − 1` sobre as posições de decode.
- **IC95:** bootstrap com 10.000 reamostragens de janelas. As diferenças entre políticas usam bootstrap **pareado** (mesmas janelas reamostradas para as duas).
- **Média das sementes:** NLL por janela é a média das 3 sementes, e o bootstrap é feito sobre essa média.
- **Tarefas:**
  - acerto por exemplo;
  - bootstrap pareado por tarefa;
  - média macro sobre as tarefas aplicáveis;
  - categorias são a média das suas tarefas.
- **Velocidade:**
  - razões pareadas por repetição (mesmo prompt, ordem alternada);
  - mediana;
  - IC95 por bootstrap das razões;
  - p50 e p95 da latência por token.

## 6. Dados (novos; nada examinado em versões anteriores entra no conjunto final)

**Texto para professor, compensadores e NLL:** 4 domínios, documentos disjuntos entre train, val e final. O manifesto `configs/v4_text_manifest.json` (ids e hashes) é versionado antes do 1º treino no 1.7B.

| Domínio | Fonte | Licença | Exclusões |
|---|---|---|---|
| en_wiki | WikiText-103 (train) | CC BY-SA 3.0 | todos os artigos do WikiText-2 e do manifesto da v0.2 |
| pt_wiki | Wikipédia PT (`wikimedia/wikipedia`, 20231101.pt) | CC BY-SA 4.0 | — |
| chat_en | `HuggingFaceTB/smol-smoltalk` | Apache-2.0 | ids usados na v0.3 |
| chat_pt | `recogna-nlp/UltrachatBR` | MIT | ids usados na v0.3 |

**Tarefas verificáveis:** usam **partições nunca avaliadas** em versões anteriores. Na v0.3 foram usadas as partições oficiais de teste e validação. Aqui, cada tarefa é dividida em DEV e FINAL por SHA-256(`tarefa:índice:vetor-v0.4`), com limite de exemplos por metade. O manifesto `configs/v4_task_manifest.json` é versionado antes do 1º treino.

| Categoria | Tarefas (partição) |
|---|---|
| raciocínio (EN) | hellaswag (train), piqa (train), winogrande (train), arc_challenge (train) |
| conhecimento (EN) | arc_easy (train), sciq (train) |
| compreensão (EN) | boolq (train) |
| matemática | gsm8k (train, 5 exemplos de demonstração tirados do *test*) |
| instruções | **IF-v0.4**: prompts novos de `smol-smoltalk`, não usados antes, com restrições verificáveis geradas e checadas pelo código oficial do IFEval (`lm_eval.tasks.ifeval`) |
| português | arc_pt (train), assin2_rte (train), global_mmlu_full_pt (test, nunca usado no projeto) |

- **Aplicabilidade** (mesma regra da v0.3): o FULL na DEV precisa ficar ≥ acaso + 3 p.p. em múltipla escolha, ou ≥ 2% em geração.
- **Ressalva:** partições de treino públicas podem ter sido vistas no pré-treino do SmolLM2. Isso afeta igualmente o FULL e as políticas, e a comparação é pareada.

**Geração livre:** prompts reais do conjunto final dos domínios de chat (EN e PT). Gera 128 tokens greedy por prompt e mede:

- repetição (fração de 4-gramas repetidos e respostas degeneradas);
- concordância com o texto do FULL;
- NLL do texto gerado sob o FULL.

## 7. Congelamento antes da avaliação final

O arquivo `configs/v4_decision.json` é commitado e publicado na branch, e o script da avaliação final se recusa a rodar se ele não estiver commitado e sem alterações locais. Ele contém:

- arquitetura;
- hiperparâmetros;
- compensador;
- k e τ por regime;
- caminhos e hashes dos checkpoints;
- modelo e revisão;
- métricas;
- sementes;
- método de agregação;
- commit.

## 8. Reprodutibilidade

- Revisões fixas dos modelos:
  - 135M: `12fd25f7`;
  - 1.7B: `31b70e2e`.
- `requirements-lock.txt` completo.
- Hashes dos dados.
- Checkpoints pequenos (compensadores e roteadores) versionados; os grandes ficam disponíveis como anexos de release.
- Cada execução grava ambiente e commit.

## 9. Registro de alterações

*(vazio no pré-registro)*
