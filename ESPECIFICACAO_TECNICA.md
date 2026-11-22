# PROJETO VETOR — Vector-guided Efficient Transformer Operation Routing

**Versão:** 0.1 — protocolo inicial de pesquisa | **Data:** 22/11/2026  
**Objetivo:** investigar se um vetor-guia probabilístico, associado às representações internas de um token, consegue evitar a execução de operações de uma rede Transformer densa pré-treinada, preservando uma qualidade aceitável e reduzindo o tempo real de decodificação.

> **Estado:** hipótese de pesquisa, não uma otimização comprovada. O projeto mede tanto resultados favoráveis quanto resultados negativos. Não demonstra P = NP, nem assegura acelerar inferência.

## 1. O que será demonstrado

A investigação precisa responder separadamente a cinco perguntas:

1. **Viabilidade funcional:** é possível desativar seletivamente um bloco computacional sem quebrar o `forward` do modelo?
2. **Sensibilidade:** quais blocos MLP podem ser omitidos, em contextos específicos, com baixo impacto sobre a previsão do próximo token?
3. **Previsibilidade:** um roteador pequeno pode aprender quais blocos omitir usando somente sinais disponíveis antes da decisão?
4. **Qualidade acumulada:** decisões boas para um token continuam boas ao longo de 32–128 tokens gerados?
5. **Ganho físico:** o tempo real de decodificação cai depois de contabilizar sincronizações, alocação de memória e custo do roteador?

A resposta positiva à pergunta 1 **não** implica respostas positivas às perguntas 3, 4 e 5.

## 2. Hardware e modelo

- Execução de referência: MacBook Pro com Apple Silicon M4 Max e 128 GB de memória **unificada** (não são 128 GB de VRAM dedicada).
- Primeiro modelo: `HuggingFaceTB/SmolLM2-135M-Instruct`, licença Apache-2.0.
- Arquitetura: `LlamaForCausalLM`; 30 camadas, dimensão oculta 576, dimensão intermediária 1536, 9 cabeças de atenção e 3 cabeças KV. Conferir sempre `model.config` depois do download.
- Precisão inicial: `float32` para calibração e testes de corretude; repetir os benchmarks em `float16` somente depois de assegurar compatibilidade/estabilidade com MPS.
- Ambiente inicial: Python, PyTorch e Transformers em MPS (Metal) ou CPU para testes unitários. Não começar por GGUF, Ollama nem MLX: eles dificultam a instrumentação das operações internas nesta fase.
- Não afirmar que é o menor modelo aberto existente. Foi escolhido por ser pequeno, aberto, instruído e ter 30 oportunidades de roteamento.

Referências técnicas:
- Modelo: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- Configuração: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/main/config.json
- Implementação do decoder: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
- PyTorch MPS: https://docs.pytorch.org/docs/stable/notes/mps.html
- Pesquisa relacionada: https://arxiv.org/abs/2410.20022

## 3. Definições e matemática

### 3.1 O ponto de partida

O modelo original é uma função pré-treinada `f_theta(x)` na qual `theta` representa os pesos fixos. Cada token tem um vetor de estado, alterado pela rede a cada camada. O modelo **não** funciona como uma busca literal em que um token percorre cada peso individualmente: realiza operações matriciais nas quais muitos pesos atuam em paralelo.

Para a camada `l`, posição de token `t`, escrevemos a versão residual simplificada do decoder:

```
u[l,t] = h[l,t] + Attention_l(RMSNorm(h[l,t]), contexto, KV)
z[l,t] = RMSNorm(u[l,t])
h[l+1,t] = u[l,t] + MLP_l(z[l,t])
```

O mecanismo VETOR introduz uma decisão binária `m[l,t]`:

```
h_VETOR[l+1,t] = u[l,t] + m[l,t] * MLP_l(z[l,t])
```

`m=1`: executar o MLP; `m=0`: ignorar *completamente o forward do MLP* e retornar zero do mesmo formato. Importante: a expressão é matemática; uma implementação que calcula `MLP(z)` e só depois multiplica por zero **não economiza** operações. A decisão de omitir ocorre **antes** do cálculo do MLP.

No início, mantemos Attention, RMSNorm e KV cache sempre ativos. Só roteamos os blocos MLP/FFN. A simplificação protege o alinhamento da sequência e diminui a possibilidade de quebrar o cache.

### 3.2 O vetor-guia

O guia é um estado auxiliar (*sidecar*), não um novo elemento concatenado diretamente ao token de entrada: concatenar dimensões ao estado oculto mudaria as dimensões das matrizes já treinadas.

Definir dimensão pequena `r=16`. Para uma entrada de MLP `z[l,t]` com dimensão `d=576`, usar um projetor treinável compartilhado:

```
g[l,t] = tanh(P * LayerNorm(z[l,t]) + e[l])
q[l,t] = sigmoid(w^T * g[l,t] + b[l])
```

- `P`: matriz `r x d`, inicializada com semente fixa e treinável.
- `e[l]`: embedding treinável específico da camada.
- `w`: vetor `r`, compartilhado entre camadas.
- `b[l]`: viés da camada.
- `q`: probabilidade estimada de que a omissão do bloco MLP seja **segura**, segundo os critérios do experimento. Não é a probabilidade de 'encontrar o peso correto'.
- `m=0` se `q >= tau_l` e o orçamento de omissões permitir; senão `m=1`.

O pequeno classificador é treinado, mas os pesos `theta` do LLM permanecem congelados. No caminho de produção não devem existir gradientes no LLM.

### 3.3 Custo teórico

Um MLP SwiGLU típico tem três projeções lineares principais de tamanhos `d x d_ff`, `d x d_ff` e `d_ff x d`. Sua contagem aproximada é:

```
F_MLP_por_token_por_camada ~= 6 * d * d_ff
                              = 6 * 576 * 1536
                              = 5.308.416 FLOPs
F_MLP_30_camadas ~= 159.252.480 FLOPs/token
```

Ignorar `k` blocos evitaria aproximadamente `k * 5,31 milhões` de FLOPs por token no componente MLP. A economia teórica **não** equivale à redução de tempo: acessos à memória, atenção, kernel launch, GPU/CPU synchronization, throughput e custo do guia influenciam a velocidade final. Os pesos omitidos seguem carregados na memória na abordagem v0.1: **não prometer redução de uso de RAM/VRAM**.

## 4. Protocolo em etapas

### Fase 0 — Reprodutibilidade e linha de base

- Criar ambiente isolado, gravar versões de Python, macOS, PyTorch, Transformers e hashes/revisões do modelo e dados.
- Conferir `torch.backends.mps.is_available()`; fallback para CPU com indicação explícita.
- Carregar tokenizer/modelo, `model.eval()`, `torch.inference_mode()`, `use_cache=True` nas medições autoregressivas.
- Definir entradas com sementes, tamanhos e distribuição registrados.
- Medir tempo de prefill, tempo de decodificação, tokens/s, memória e resultados de próximos tokens.
- Salvar prompts, tokens e logits base (ou estatísticas resumidas suficientes) em arquivos reprodutíveis.

### Fase 1 — Intervenção mecânica e ablação

- Instrumentar **somente** os módulos `model.model.layers[l].mlp` com um wrapper que conserva uma referência ao módulo original.
- Em modo normal, `wrapper(x) == inner_mlp(x)` com tolerância de ponto flutuante; idealmente, mesmas saídas bit a bit sob mesmo dispositivo.
- Em modo `skip_decode`, **somente se o formato for `[batch,1,d]`**, retornar `zeros_like(x)` sem executar `inner_mlp`.
- Em modo `ablate_last` para sequências `[batch,T,d]`, calcular normalmente as saídas para todos os tokens anteriores e zerar apenas a saída do **último** token. Esse modo é diagnóstico; pode calcular o MLP inteiro, pois NÃO serve para medição de velocidade.
- Em modo prefill, *nunca* pular MLP na versão 0.1. Pode capturar a entrada `z[l,last]` por camada.
- Fazer testes de restauração completa dos wrappers, não alterar checkpoints originais, não trocar atenção nem KV.

### Fase 2 — Rotulagem de sensibilidade (*professor*)

Construir amostras `(prefixo x_1...x_t, próximo token y_{t+1})` de um conjunto de textos. Dividir por texto/documento em **train, validation e test**, antes de extrair prefixos, para evitar contaminação de conteúdos semelhantes.

Para cada prefixo `x`:

1. Rodar o modelo completo, sem cache, guardando `z[l,t]` (entradas do MLP no último token) e logits `v0` do próximo token.
2. Para cada camada candidata `l`, repetir a execução com **somente o MLP da camada l no último token** desativado. Guardar logits `v_l`.
3. Calcular divergência entre probabilidades `p0=softmax(v0)` e `p_l=softmax(v_l)`:

```
D_l = KL(p0 || p_l) = sum_j p0[j] * log(p0[j] / p_l[j])
```

4. Opcionalmente calcular aumento de perda do token verdadeiro `y`:

```
DeltaNLL_l = -log p_l[y] + log p0[y]
```

5. Criar alvo supervisionado binário com limites iniciais, ajustáveis **exclusivamente na validação**:

```
safe_l = 1[ D_l <= 0.02 AND DeltaNLL_l <= 0.05 ]
```

Os números `0.02` e `0.05` são hipóteses de trabalho em nats, **não limiares científicamente comprovados**. Guardar `D_l` e `DeltaNLL_l` contínuos para refazer a política sem executar ablações novamente.

6. Salvar `{sample_id, split, layer_id, features, kl, delta_nll, safe}`. O professor tem acesso aos resultados das ablações, mas o roteador de produção **só pode usar as features e metadados disponíveis antes do MLP**, nunca o KL futuro ou a resposta verdadeira.

**Nota importante:** intervenções individuais não são aditivas. Duas omissões seguras quando testadas separadamente podem prejudicar a saída quando combinadas. Esse é um ponto obrigatório de avaliação conjunta.

### Fase 3 — Controles sem aprendizado

Executar quatro controles na mesma carga de trabalho:

A. `FULL`: nenhum MLP ignorado.
B. `RANDOM`: escolher aleatoriamente camadas elegíveis, mantendo igual taxa de omissão.
C. `STATIC`: escolher as camadas que apresentaram menor impacto **no treino/validação**, nunca a partir do teste.
D. `ORACLE-DIAG`: escolher as camadas com informação de ablação do próprio exemplo para estimar um limite superior **não implementável em produção**; identificar claramente como oracle e nunca comparar suas decisões como se fossem aprendizado sem vazamento.

Testar orçamento de 0%, 10%, 20%, 30% e 40% dos blocos MLP elegíveis, protegendo inicialmente as duas primeiras e duas últimas camadas. Se houver falhas, reduzir o orçamento.

### Fase 4 — Treinar guia probabilístico

- Treinar **apenas** `P`, `e`, `w`, `b` com `BCEWithLogitsLoss`, possível `pos_weight` ou amostragem balanceada e penalização L2.
- Entradas: `z[l,t]` e índice da camada; alvo: `safe_l` ou risco contínuo. Com orçamento limitado, a alternativa é regressão de `D_l` e decisão pelo menor risco previsto.
- Calibrar `tau_l` na **validação**, visando menor perda com orçamento de blocos poupados. Registrar curvas risco × economia; não otimizar a pontuação sobre o teste.
- Política de segurança: proteger primeiras duas e últimas duas camadas; garantir quantidade mínima de blocos MLP executados por token; rejeitar máscaras vazias/fora do limite; qualquer erro executa `FULL`.
- Comparar contra `RANDOM` e `STATIC` sob **exatamente a mesma taxa efetiva de omissão**, para comprovar que o guia aprendeu algo além da seleção trivial de camadas.

### Fase 5 — Política de execução (dois regimes)

**Regime S (route-once / por sequência):**

1. Fazer prefill completo no prompt e capturar `z[l,last]` de cada camada.
2. Aplicar o roteador às 30 features **em um único lote** e converter para uma máscara booleana por camada, uma única vez.
3. Na decodificação token a token, aplicar a mesma máscara para todas as posições geradas: o MLP retorna zero quando a camada foi omitida; atenção e KV seguem completos.
4. Medir economia real. O regime S tem baixa sobrecarga, mas não acompanha a mudança de conteúdo ao longo da resposta.

**Regime D (guia dinâmico, objetivo central):**

1. Partir da máscara do prefill.
2. Durante a decodificação, guardar `z[l,t]` antes de cada MLP, mesmo que o MLP seja ignorado.
3. Reavaliar o roteador a cada `K` tokens (K em {1,4,8,16}), em um único lote sobre todas as camadas.
4. Aplicar as decisões ao(s) **próximo(s)** token(s). Isso cria atraso de um token ou de até K tokens, explicitamente documentado; nunca usar informação do futuro para decidir o passado.
5. Evitar chamar `.item()` por camada: em MPS isso pode sincronizar GPU e CPU dezenas de vezes por token, inviabilizando o ganho. Preferir uma sincronização por atualização de máscara.
6. Por segurança, se ocorrer instabilidade, atualizar máscara menos frequentemente ou executar FULL.

**Versão futura D2:** decisão realmente intra-token antes de cada MLP, com kernel/roteamento especializado que evite sincronizações repetidas GPU→CPU. Não é escopo do MVP; sua eficácia dependerá também de implementação de baixo nível (Metal/MLX ou kernels especializados), não só de matemática.

### Fase 6 — Avaliação científica e de desempenho

Qualidade:
- NLL e perplexidade de continuação **autoregressiva com cache** e texto verdadeiro (teacher forcing passo a passo) com guia ativado durante decode. Não avaliar qualidade com `forward` de toda a sequência se a política desliga apenas MLP no decode, porque isso mediria um modelo diferente.
- Divergência KL média e percentis em relação às probabilidades do FULL em **mesmo prefixo de referência** (ablação controlada); registrar separadamente o comportamento durante geração livre, quando os caminhos textuais divergem.
- Concordância do próximo token (top-1), variação do top-k, taxa de tokens idênticos ao FULL.
- Saídas geradas com greedy (`do_sample=False`), separando métricas linguísticas de velocidade; 135M é limitado e não mede qualidade avançada de raciocínio.

Tempo:
- Separar prefill e decode; para speedup, focar decode tokens/s.
- Warm-up antes de medir e `torch.mps.synchronize()` (quando disponível) **antes/depois do trecho cronometrado**, não a cada camada.
- Repetir medições alternando a ordem FULL/STATIC/GUIDED para minimizar vieses térmicos/cache; registrar mediana e p95, semente, temperatura/regime, modelos, comprimento do prompt, número **fixo** de tokens decodificados.
- Comparar condições de mesma precisão, comprimento, batch, device, rota e mecanismo de geração.
- Distinguir taxa de MLPs evitados, FLOPs teóricos evitados e ganho temporal efetivo.
- Nunca chamar um `forward` que calcula MLP e zera depois de 'otimizado'.

Exemplo de critério de êxito **proposto, não garantido**:
- economia efetiva de >=20% dos MLPs elegíveis;
- aumento relativo de NLL <=5% no conjunto de teste, e KL reportado;
- aumento de mediana de decode tok/s >=5% frente ao FULL, com confiabilidade estatística;
- desempenho acima dos controles RANDOM e STATIC para o mesmo orçamento de MLPs evitados.

Se o guia poupar FLOPs mas ficar mais lento, registrar como resultado negativo de engenharia, não mascarar. O modelo de 135M pode ser dominado por sobrecarga de despacho/kernel, e resultados podem variar em modelos maiores.

## 5. Datasets e política contra vazamento

- Smoke-test inicial: 8–16 exemplos pequenos criados localmente.
- Piloto: ~100–200 prefixos de textos distintos ou com split por documento e comprimento 32–128 tokens.
- Fase séria: >=500 prefixos de treino, >=100 validação e >=100 teste, ajustando pela capacidade da máquina e pela variação dos dados.
- Priorizar textos licenciados e cujos termos estejam documentados; por exemplo, trechos de conteúdos próprios/autorizados, ou corpus público com licença verificada.
- Registrar identificadores dos textos; evitar prefixos sobrepostos da mesma fonte em splits diferentes.
- Dados em inglês são adequados ao primeiro teste com SmolLM2, mas qualidade final em português requer dados de avaliação em português.

## 6. Formatos e entregáveis

Arquivos obrigatórios a serem produzidos pela implementação do Claude:

```
vetor/
  README.md
  pyproject.toml
  requirements-lock.txt (ou uv.lock)
  src/vetor/
    model_adapter.py            # acesso aos LlamaDecoderLayer.mlp
    guide.py                    # projetor e classificador
    wrappers.py                 # execução/skip/ablate/captura
    collect.py                  # professor e dados de calibração
    train_router.py             # treino exclusivo do guia
    policies.py                 # full, random, static, sequence, dynamic
    decode.py                   # loop autoregressivo com KV
    evaluate.py                 # NLL, perplexidade, KL e comparações
    benchmark.py                # tempos de prefill/decode e tokens/s
    report.py                   # tabela e gráficos de Pareto
  scripts/
    00_sanity.py
    01_baseline.py
    02_ablate.py
    03_train.py
    04_evaluate.py
    05_benchmark.py
  tests/
    test_equivalence_full.py
    test_skip_really_bypasses_mlp.py
    test_ablate_last_only.py
    test_cache_consistency.py
    test_no_target_leakage.py
    test_router_budget.py
    test_reproducibility.py
  data/                       # pequenos exemplos + arquivos não versionados
  results/                    # JSON, CSV, PNG e relatórios
  configs/default.yaml
```

Cada execução gravará `run_id`, modelo/revisão, ambiente, semente, hashes de dados, política e parâmetros; saídas `results/{run_id}.json`, `{run_id}_tokens.csv`, `{run_id}_layers.csv` e gráficos de qualidade versus velocidade. Gerar relatório de resultados com intervalo de confiança quando houver amostra suficiente.

## 7. Testes obrigatórios (gate de qualidade de software)

1. **Equivalência:** com 100% dos MLPs executados e mesmos tokens/dtype/cache, o adaptador retorna logits iguais (ou erro dentro de tolerância numérica explicitada) ao modelo sem patch.
2. **Skip real:** inserir contador no MLP original, provocar `skip_decode` em entrada de um token e verificar **zero chamadas** ao MLP original; verificar que o tensor retornado tem mesmo shape, dtype e device.
3. **Ablate_last:** em entrada com T>1, saídas das posições anteriores mantidas e última zerada.
4. **Cache:** a política usa prefill completo e decode de um token com KV cache; confirmar comprimento e sem exceções; comparar o modo FULL com `model.generate` ou um loop de referência dentro das mesmas opções de geração.
5. **Congelamento:** nenhuma alteração nos pesos originais antes/depois da calibração/treino do roteador.
6. **Sem vazamento:** o roteador não recebe logits futuros, o rótulo, KL calculado com ablação ou token alvo durante inferência; splits por documento.
7. **Orçamento:** nunca omitir camadas protegidas; respeitar limite máximo de omissões; em falhas rodar FULL.
8. **Tempo:** nenhum `torch.mps.synchronize` por camada; benchmark considera execução real e overhead do guia.
9. **Ablation controls:** mesmos percentuais de omissão efetiva no RANDOM, STATIC e GUIDED.
10. **Resultados negativos:** relatório sempre mostra os dados crus, inclusive quando guided perde do baseline.

## 8. Roadmap e decisões

- **Milestone M0**: baseline + testes de igualdade. Prosseguir somente se reproduzir o modelo.
- **Milestone M1**: ablação por bloco e mapa de sensibilidade por camada; avaliar se há blocos removíveis.
- **Milestone M2**: comparação RANDOM/STATIC e ganho real sem roteador; se não houver ganho mesmo sem roteador, investigar backend antes de treinar guia.
- **Milestone M3**: guia treinado em cima de z[l,t], teste em dados novos, curvas qualidade/custo.
- **Milestone M4**: `route-once` com KV, benchmark MPS.
- **Milestone M5**: guia dinâmico K=1,4,8,16 e comparação direta com route-once.
- **Milestone M6**: replicação em modelo maior após resultado reprodutível, sem alterar os critérios retroativamente.

## 9. Riscos centrais

- O roteador pode errar quando blocos interagem: KL de omissões isoladas não prevê perfeitamente múltiplas omissões.
- O MLP ignorado modifica estados posteriores e, portanto, entradas do roteador. Treino só com estados FULL cria *distribution shift*. Se necessário, fazer coleta iterativa de estados do modelo guiado e retreino, sempre preservando conjunto de teste intocado.
- Máscaras granulares de GPU podem tornar execução mais lenta que o modelo denso. Este projeto começa com granularidade de **bloco completo**, não peso individual.
- Inferência MPS e a API Transformers mudam. Testar compatibilidade e registrar versões; não copiar implementação privada do LlamaDecoderLayer sem necessidade.
- Uma redução de FLOPs pode não reduzir latência de um modelo de 135M. O sinal importante é relação qualidade/custo, posteriormente validada em modelo maior.
- Sem treinamento do LLM e sem pesos extras na arquitetura original, não temos garantia de equivalência; otimização é aproximada.

## 10. Resultado esperado do estudo (não resultado predito)

Ao terminar, devemos ter uma tabela de pelo menos cinco políticas (`FULL`, `RANDOM`, `STATIC`, `GUIDED-S`, `GUIDED-D`) com: taxa de MLPs omitidos, delta de NLL, KL, concordância de tokens, prefill, tokens/s de decode, p95 e uso de memória. O valor científico estará em mostrar, com o mesmo conjunto de testes, se usar vetores internos para decidir caminhos supera a escolha aleatória ou fixa e se a economia chega ao relógio.
