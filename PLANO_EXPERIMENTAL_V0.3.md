# VETOR v0.3 — Plano experimental e critérios (pré-registro)

**Data:** 23/11/2026 · Complementa `ESPECIFICACAO_V0.3.md`. Escrito antes de qualquer treino da v0.3. Alterações: seção 9.

## 1. Fases e portões de decisão

| Fase | Conteúdo | Portão para avançar |
|---|---|---|
| 0 | Auditoria; branch `v0.3` a partir de `main` (v0.2 preservada na tag) | ✓ concluída em 23/11/2026 |
| 1 | Especificação matemática e custos | ✓ este documento e a especificação |
| 2 | Manifesto DEV/FINAL; baseline do FULL na DEV; lista de tarefas aplicáveis; dados de treino descontaminados | lista de tarefas e dados congelados no Git **antes** do 1º treino |
| 3 | Protótipos sem guia: correção estática A₀ (`L_task`, depois `+ L_distill`) | A₀ com ganho macro na DEV com IC95 > 0; caso contrário, registrar e testar no máximo 2 variações de capacidade (r, \|S\|) dentro do orçamento |
| 4 | Guia: A (portão), B (realocação), C (ganho esperado), D (destilação), com os controles 3, 5 e 6 nas mesmas condições | escolha da configuração **na DEV** |
| 5 | Otimização de execução (sem sincronização por camada) e medição de throughput, TTFT e memória | throughput ≥ 98% medido |
| 6 | Decisão congelada em `configs/v3_decision.json` (commit) → avaliação única na metade FINAL → relatório | — |

Se o portão da Fase 3 falhar com todas as variações, a v0.3 termina com resultado negativo documentado: poucos parâmetros auxiliares não melhoram o modelo congelado nesta escala.

## 2. Tarefas aplicáveis (regra fixada antes de ver qualquer resultado de treino)

Uma tarefa entra na **média principal** se, no FULL, na metade DEV:

- **múltipla escolha:** acerto ≥ acaso + 3 p.p. (acaso = 1/nº de alternativas);
- **geração verificável:** acerto ≥ 2%.

As demais são relatadas como "no piso", fora da média principal. A lista resultante é registrada na seção 8 deste documento, antes do primeiro treino.

## 3. Critérios de sucesso (congelados)

Todos são medidos na metade **FINAL**, avaliada uma única vez.

**Qualidade**

- **Q1:** média macro (por tarefa aplicável, em p.p.) de VETOR − FULL ≥ **+5,0 p.p.**, com limite inferior do IC95 > 0.
- **Q2:** nenhuma categoria aplicável com queda ≥ **2,0 p.p.** em relação ao FULL (estimativa pontual).
- **Q3:** VETOR − melhor controle sem guia (A₀, LoRA, LoRA destilado) > 0, com limite inferior do IC95 > 0, na média macro.

**Desempenho** (mesmo dispositivo, precisão float32 e backend PyTorch/MPS)

- **P1:** throughput de decode ≥ **98%** do FULL (mediana das razões pareadas, ≥ 15 repetições, ordem rotacionada; o IC95 é relatado).
- **P2:** tempo até o primeiro token (prefill + decisões) ≤ +5% em relação ao FULL.
- **P3:** memória adicional (alocação do driver MPS) ≤ +5%.
- **P4:** zero sincronizações GPU→CPU por camada, verificado por teste automático.

**Científico**

- **S1:** a vantagem do roteamento adaptativo não se explica por treino ou parâmetros. Isso exige (A − A₀) > 0 **e** (A − LoRA de mesmo nº de parâmetros) > 0, ambos com IC95 > 0, nas mesmas condições de dados e perda.

Os critérios não serão alterados depois de vistos os resultados. Se Q1 for inviável, ele é registrado como não atingido, e um novo experimento é definido com novo pré-registro.

## 4. Estatística

- **Por tarefa:** acerto por exemplo (0/1, ou o escore da métrica oficial do lm-eval) do modelo avaliado e do comparado, **nos mesmos exemplos**.
- **Média macro:** média simples entre tarefas aplicáveis; cada categoria é a média das suas tarefas.
- **IC95:** bootstrap pareado com 10.000 reamostragens, reamostrando exemplos dentro de cada tarefa de forma independente e recalculando a média macro da diferença.
- Os resultados por tarefa e por categoria são sempre relatados, mesmo quando a média melhora.

## 5. Dados de treino

| Idioma | Fonte | Licença | Uso |
|---|---|---|---|
| EN | `HuggingFaceTB/smol-smoltalk` | Apache-2.0 | prompts e respostas de instrução |
| PT-BR | `recogna-nlp/UltrachatBR` | MIT | prompts e respostas de instrução |
| referência (`L_stability`) | WikiText-103 (artigos da partição `train` da v0.2) | CC BY-SA 3.0 | texto genérico |

- **Descontaminação:** remover todo exemplo de treino com qualquer 13-grama (palavras, minúsculas) presente em qualquer exemplo de avaliação, DEV ou FINAL, de qualquer tarefa.
- O manifesto de treino (ids e hashes) é versionado antes do 1º treino.
- **Orçamento por execução:** ~4 M tokens de resposta; 2 sementes por configuração final.
- Validação de treino (early stopping, escolha de α, β, γ) usa um conjunto separado do treino e **a metade DEV** da bateria. A metade FINAL nunca é usada.

## 6. Medição de desempenho

`scripts/v3_perf.py` (Fase 5) mede:

- decode tok/s, prefill e TTFT com o throttle de eventos (igual para todas as políticas);
- 64 tokens gerados, 15–20 repetições, ordem rotacionada, razões pareadas contra o FULL;
- memória: `torch.mps.driver_allocated_memory()` após aquecimento;
- custos por componente em µs (correção, portão, compensador) e FLOPs calculados.

## 7. Reprodutibilidade

Sementes fixas; revisão fixa do modelo e do professor; hashes dos dados; checkpoints dos módulos treináveis em `results/v0.3/`; cada execução grava configuração, ambiente e commit.

## 8. Tarefas aplicáveis (preenchido após o baseline da Fase 2, antes do 1º treino)

*(pendente)*

## 9. Registro de alterações

*(vazio no momento do pré-registro)*
