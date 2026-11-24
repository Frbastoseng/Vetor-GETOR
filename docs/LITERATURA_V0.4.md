## Revisão de literatura: computação condicional e salto de camadas/blocos em LLMs pré-treinados

*Revisão feita em 24/11/2026. As fontes foram conferidas nas páginas de resumo do arXiv, no texto completo dos artigos (PDF/HTML) e nos repositórios oficiais. Todo ponto que não pôde ser conferido está marcado como **[não verificado]**. "Speedup real" quer dizer tempo de relógio medido. "Speedup estimado" quer dizer um valor derivado de FLOPs ou do número de camadas executadas.*

### 1. Trabalhos obrigatórios

#### 1.1 Mixture-of-Depths (MoD)
- **Título:** *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*
- **Autores:** D. Raposo, S. Ritter, B. Richards, T. Lillicrap, P. C. Humphreys, A. Santoro (Google DeepMind)
- **arXiv:** 2404.02258, v1 em 02/04/2024. Preprint; nenhum venue aparece na página do arXiv.
- **O que é pulado:** o bloco inteiro (atenção + MLP) nos tokens que não são selecionados. Esses tokens também deixam de ser "atendíveis" naquele bloco.
- **Base:** o modelo é treinado do zero junto com o roteador (60M a 3B, análise isoFLOP). Não se aplica a um modelo congelado.
- **Roteador:** projeção linear escalar (r = wᵀx) com top-k por sequência e capacidade fixa (a melhor configuração é 12,5% de capacidade em blocos alternados). O top-k não é causal. Para amostragem autoregressiva, os autores usam uma perda auxiliar BCE ou um preditor auxiliar pequeno, com acurácia de cerca de 97–99%.
- **KV cache:** o token pulado não gera K/V naquele bloco.
- **Qualidade:** iguala o baseline com FLOPs e tempo de treino equivalentes.
- **Speedup:** "até ~50% mais rápido por passo na amostragem" e ~60–66% no treino, medido em passos/s em TPU. O artigo não informa batch nem topologia exata **[detalhes não verificados]**.
- **Código oficial:** não há.
- **Baseline viável em Mac/1.7B?** Não. Exige pré-treino e não se aplica a um modelo congelado.

#### 1.2 LayerSkip
- **Título:** *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding*
- **Autores:** M. Elhoushi, A. Shrivastava, D. Liskovich, B. Hosmer, B. Wasti, L. Lai, A. Mahmoud, B. Acun, S. Agarwal, A. Roman, A. A. Aly, B. Chen, C.-J. Wu (Meta)
- **arXiv:** 2404.16710, v1 em 25/04/2024. Venue: ACL 2024 (long).
- **O que é pulado:** as camadas finais (saída antecipada), que depois são verificadas pelas camadas restantes (decodificação autoespeculativa).
- **Base:** **não é congelada**. O modelo é treinado ou ajustado com *layer dropout* crescente em profundidade e uma perda de saída antecipada (pré-treino, pré-treino continuado ou fine-tuning).
- **Roteador:** não tem roteador aprendido. A camada de saída é fixa (`--exit_layer`).
- **KV cache:** a implementação principal reutiliza o cache entre o rascunho e a verificação. O README observa que a variante nativa do HF *não* compartilha esse cache.
- **Speedup real:** sim, medido como tempo médio por token contra a decodificação autoregressiva em H100/A100: até 2,16× (CNN/DM), 1,82× (código) e 2,0× (TOPv2). Com greedy, a saída é equivalente à do modelo. Batch = 1 **[não verificado]**.
- **Código:** https://github.com/facebookresearch/LayerSkip (licença CC-BY-NC). Há checkpoints para Llama 2 7B/13B/70B, CodeLlama 7B/34B, Llama 3 8B e **Llama 3.2 1B**.
- **Baseline viável?** Só parcialmente. O speedup depende de um modelo treinado com a receita LayerSkip, e o código de treino é descrito como "work-in-progress". Também é outra família de método (sem perdas em greedy, via especulação). O checkpoint Llama-3.2-1B poderia rodar em Mac via HF/MPS **[não testado]**.

#### 1.3 DiffSkip
- **Título:** *DiffSkip: Differential Layer Skipping in Large Language Models*
- **Autores:** Xuan Luo, Weizhi Wang, Xifeng Yan (UC Santa Barbara)
- **Venue:** Findings of ACL 2025, pp. 7221–7231. **Não encontrei versão no arXiv.**
- **O que é pulado:** **somente o FFN/MLP**, por token. A atenção sempre roda.
- **Base:** **congelada**. Só o roteador e o adaptador são treinados.
- **Roteador:** MLP com gargalo, `W_head·(W_up·tanh(W_down z))` com sigmoide e limiar τ. A entrada é **ΔAttn = Attn(x) − x**, motivada pela alta similaridade de cosseno entre ΔAttn e ΔFFN. Perda de esparsidade em L2 sobre o número esperado de FFNs usados, ponderada por token conforme a perplexidade (tokens difíceis são menos penalizados).
- **Compensador:** um "adaptor" no lugar do FFN pulado, implementado como um FFN pequeno com dimensão intermediária bem reduzida. Na ablação, tirar o adaptador derruba a retenção para 23,2%, um adaptador **linear** retém 85,8% e o FFN pequeno retém 99,0%, numa configuração específica da Tabela 3.
- **KV cache:** não há problema, porque a atenção não é pulada.
- **Qualidade:** Llama-3-8B-Instruct com 8 FFNs pulados (25%) mantém 91,3% do desempenho original.
- **Speedup real:** o artigo diz explicitamente que **não há speedup na decodificação em GPU**: com batch, os tokens se dividem entre o FFN e o adaptador e o custo de E/S supera a economia. Há um ganho modesto no prefill, medido em 8×A6000 com batch 8 (ex.: 33,12 → 35,24 it/s).
- **Código:** só o checkpoint `xuan-luo/DiffSkip-Llama-3-8B-Instruct` no Hugging Face. **Não achei repositório de treino no GitHub [não verificado se existe]**.
- **Baseline viável?** A ideia é reimplementável em poucas linhas sobre a infraestrutura do VETOR (trocar a entrada do roteador por ΔAttn e o compensador por um FFN pequeno). O código oficial não serve diretamente para 1.7B.

#### 1.4 FlexiDepth
- **Título:** *Adaptive Layer-skipping in Pre-trained LLMs*
- **Autores:** Xuan Luo, Weizhi Wang, Xifeng Yan (UCSB)
- **arXiv:** 2503.23798, v1 em 31/03/2025 (v3 em 08/10/2025). Venue: COLM 2025.
- **O que é pulado:** a camada inteira por token. No caminho de salto, a atenção não calcula a query do token, mas **K/V continuam sendo calculados**, e o FFN é substituído por um adaptador.
- **Base:** **congelada**. Só o roteador e o adaptador são treinados.
- **Roteador:** MLP com gargalo d/16 sobre o estado normalizado (RMSNorm), com sigmoide, limiar τ e SparseMixer para o gradiente. O adaptador tem a mesma forma do FFN, com dimensão intermediária 16× menor (não é low-rank linear).
- **KV cache:** o K/V dos tokens pulados é calculado. Sem isso, a retenção cai para 84,3%.
- **Qualidade:** Llama-3-8B-Instruct pulando 8 de 32 camadas mantém 100,7% (Tab. 1). Também foi avaliado em Llama-2-13B e Qwen-2.5-3B.
- **Speedup real:** **nenhum**. Em 8×A6000 com batch 8 o throughput fica parecido com o vanilla (MMLU 33,43 vs 32,21 it/s; CoQA 7,78 vs 8,29). Os autores atribuem isso ao fluxo de controle e ao acesso irregular à memória. Um trabalho posterior (vSkipper, seção 2.8) mede o laço de geração do FlexiDepth **14,6–21,0% mais lento** que a base.
- **Código:** checkpoint `xuan-luo/FlexiDepth-Llama-3-8B-Instruct` (Apache-2.0) e o dataset FlexiPatterns. Repositório de treino **[não verificado]**.
- **Baseline viável?** Só como reimplementação. Para o VETOR, a parte relevante é a variante "MLP → adaptador".

#### 1.5 Dr.LLM
- **Título:** *Dr.LLM: Dynamic Layer Routing in LLMs*
- **Autores:** Ahmed Heakl, Martin Gubri, Salman Khan, Sangdoo Yun, Seong Joon Oh
- **arXiv:** 2510.12773, v1 em 10/2025. Venue: ICLR 2026.
- **O que é pulado:** a camada inteira, que pode ser pulada, executada ou **repetida**.
- **Base:** **congelada**. Treina só os roteadores, cerca de 11M parâmetros num modelo de 3B (0,14%).
- **Roteador:** um `RouterBlock` por camada (Linear-GELU-Linear, dimensão oculta 128) sobre o estado oculto com *mean-pooling* em janelas (8 janelas). Supervisão por rótulos obtidos com MCTS e focal loss.
- **Granularidade:** **por entrada/sequência, não por token** (uma configuração de tamanho L por exemplo).
- **KV cache:** o repositório não trata disso **[não verificado]**.
- **Qualidade:** +3,4 pontos em ARC/DART economizando cerca de 5 camadas por exemplo. Perda de 0,85% fora do domínio.
- **Speedup real:** **não reportado**. Só a economia de camadas.
- **Código:** https://github.com/parameterlab/dr-llm (MIT). Suporta Llama, Qwen2 e Qwen3. O treino usou 1×A100 40GB.
- **Baseline viável?** Fraco para o VETOR, porque a granularidade é por sequência e o foco é raciocínio.

#### 1.6 PoLar
- **Título:** *Skip a Layer or Loop It? Learning Program-of-Layers in LLMs*
- **Autores:** Ziyue Li, Yang Li, Tianyi Zhou
- **arXiv:** 2606.06574, v1 em 04/06/2026 (v2 em 08/08/2026). Venue: ICML 2026. Estende de forma substancial o arXiv 2507.07996 (*Skip a Layer or Loop it? Test-Time Depth Adaptation of Pretrained LLMs*, 10/07/2025, que busca a melhor "Chain-of-Layers" por amostra via MCTS, sem fine-tuning).
- **O que é pulado:** segmentos de camadas, que podem ser pulados ou repetidos ("programa" de camadas).
- **Base:** pré-treinada. O preditor (~2,1M parâmetros) é treinado. Se a base fica congelada **[não verificado]**; o resumo chama os programas de *training-free*.
- **Granularidade:** **por entrada**.
- **KV cache:** não discutido **[não verificado]**.
- **Qualidade:** melhora a acurácia em benchmarks de matemática em relação à inferência padrão e a métodos anteriores de profundidade dinâmica, muitas vezes com menos camadas.
- **Speedup real:** **não reportado**.
- **Código:** https://github.com/tianyi-lab/PoLar (Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, Qwen3-8B, Qwen1.5-MoE). Licença **[não verificada]**.
- **Baseline viável?** Não para o objetivo do VETOR (salto por token para eficiência).

### 2. Outros trabalhos relevantes

#### 2.1 CALM (saída antecipada)
- *Confident Adaptive Language Modeling*. Schuster, Fisch, Gupta, Dehghani, Bahri, Tran, Tay, Metzler (Google). arXiv 2207.07061, 14/07/2022. NeurIPS 2022 (oral).
- **O que faz:** saída antecipada por token no decodificador de um T5 encoder-decoder (8 camadas). O modelo é treinado ou ajustado. O critério de saída usa confiança (softmax, saturação do estado ou classificador treinado) com calibração estatística.
- **KV:** usa "state propagation": nos tokens que saíram, o estado da camada de saída é **copiado** para as camadas superiores para que os tokens futuros possam atender a eles.
- **Speedup:** "até ×3" (×3,15–×3,65), porém **estimado** a partir de benchmark em TPUv3. A métrica principal é o número médio de camadas.
- **Código:** https://github.com/google-research/t5x/tree/main/t5x/contrib/calm (JAX/T5X).

#### 2.2 SkipDecode
- *SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference*. Del Corro et al. (Microsoft Research). arXiv 2307.02628, 05/07/2023.
- **O que faz:** pula as camadas **inferiores e intermediárias** e concentra a computação no topo. O ponto de saída é único por posição dentro do batch e diminui monotonicamente com a posição, o que evita recalcular o KV de tokens anteriores. O modelo (OPT 1.3B/6.7B) é ajustado.
- **Speedup:** "2×–5×", mas é a **meta de computação configurada** (camadas executadas). Não é tempo de relógio medido.
- **Código:** **[não verificado]**.

#### 2.3 D-LLM
- *D-LLM: A Token Adaptive Computing Resource Allocation Strategy for Large Language Models*. Yikun Jiang, Huanyu Wang, Lei Xie, Hanbin Zhao, Chao Zhang, Hui Qian, John C.S. Lui (Huawei/ZJU/CUHK). NeurIPS 2024. arXiv **[não encontrado]**.
- **O que faz:** decide por token e por camada se a camada inteira é executada ou pulada. A base é ajustada com **LoRA** junto com o módulo de decisão (não fica congelada).
- **KV:** política de *eviction*: os tokens pulados em uma camada saem da atenção dessa camada por meio de uma máscara esparsa, preservando os m=2 primeiros tokens (efeito *attention sink*).
- **Resultado:** até 45–50% de redução de custo e de KV em Llama2-7B/Llama3-8B. Não reporta **tempo de relógio**; a métrica é FLOPs.
- **Código:** github.com/Jyk-122/D-LLM (citado no artigo).

#### 2.4 Router-Tuning / MindSkip
- *Router-Tuning: A Simple and Effective Approach for Dynamic Depth* (título atual da v6: "...for Enabling Dynamic-Depth in Transformers"). Shwai He, Tao Ge, Guoheng Sun, Bowei Tian, Xiaoyang Wang, Dong Yu (UMD / Tencent AI Lab). arXiv 2410.13184, v1 em 10/2024. EMNLP 2025 (main).
- **O que faz:** **backbone congelado**, só os roteadores são treinados (projeção linear de uma camada, `σ(Wx)`, W inicializado em zero, limiar 0,5 e straight-through estimator). Funciona por token ou por sequência (por sequência é o padrão). O alvo padrão (MindSkip) é **a atenção**. Os autores também testam blocos e MLP e relatam que **pular MLP prejudica bem mais** que pular atenção.
- **KV:** pular atenção reduz o KV (−8 GB com seq 2048 e batch 64 no Llama-3-8B).
- **Qualidade:** quedas de 0,1 a 1,1 ponto de média nos modelos 7–14B.
- **Speedup real:** **1,21×** medido em 1×RTX A6000 Ada, com seq 2048 e batch ajustado para saturar a GPU, ou seja, **não é batch 1**. Treino em menos de 15 minutos.
- **Código:** https://github.com/CASE-Lab-UMD/Router-Tuning

#### 2.5 FFN-SkipLLM
- *FFN-SkipLLM: A Hidden Gem for Autoregressive Decoding with Adaptive Feed Forward Skipping*. Ajay Jaiswal et al. (UT Austin e outros). arXiv 2404.03865, 04/2024. EMNLP 2024.
- **O que faz:** **sem treino**, pula **apenas FFNs** por token. Quando a similaridade de cosseno entre a entrada e a saída de um FFN passa de um limiar, pula de forma gulosa os FFNs das k camadas seguintes. As camadas iniciais e finais ("cold") e os primeiros tokens gerados (warm-up) sempre rodam completos. **Nada substitui o FFN pulado** (identidade).
- **KV:** intacto, porque a atenção não é pulada.
- **Qualidade:** cerca de 25–30% dos FFNs pulados com perda pequena (Llama-2 7B/13B, avaliação via GPT-4). Acima de ~35% a queda fica significativa.
- **Speedup real:** **não reportado**.
- **Código:** **[não encontrado]**.

#### 2.6 Deja Vu (esparsidade contextual)
- *Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time*. Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, A. Shrivastava, Ce Zhang, Yuandong Tian, C. Ré, Beidi Chen. ICML 2023 (arXiv 2310.17157).
- **O que faz:** preditores pequenos, treinados sobre a base sem retreiná-la, escolhem por token quais cabeças de atenção e neurônios do MLP ativar. Tem execução assíncrona "lookahead" e kernels próprios.
- **Speedup real:** latência **>2×** contra FasterTransformer e **>6×** contra a implementação do HF no OPT-175B, em GPU (A100, **[hardware não verificado na página]**).
- **Código:** https://github.com/FMInference/DejaVu
- É o principal precedente de que a economia **dentro** do MLP só vira tempo com kernels dedicados.

#### 2.7 CATS
- *CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models*. Je-Yong Lee, Donghyun Lee, Genghan Zhang, Mo Tiwari, Azalia Mirhoseini. arXiv 2404.08763, 12/04/2024. COLM 2024.
- **O que faz:** limiariza as ativações do MLP com gate (50% de esparsidade) e mantém ~99% do desempenho mesmo sem fine-tuning.
- **Speedup real:** cerca de **15%** na latência de geração por token, com kernel GPU dedicado.

#### 2.8 vSkipper (ligado diretamente à lacuna entre FLOPs e tempo)
- *vSkipper: Translating Dynamic Layer Skipping into LLM Serving Gains*. Wei Da, Yavuz Ferhatosmanoglu, Evangelia Kalyvianaki. arXiv 2609.37062, 29/09/2026 (preprint, cs.DC).
- **Ponto central:** os métodos de salto dinâmico usam laços de geração próprios, incompatíveis com engines de serving. O FlexiDepth pula cerca de 8 de 32 camadas e mesmo assim decodifica **14,6–21,0% mais devagar** que a base. O vSkipper (sobre SGLang) agrupa tokens por decisão e só usa execução roteada quando prevê ganho. Resultado: −36,8% de latência média ponta a ponta em GSM8K e −13,6% em BBH, no "joelho" da curva de carga, e +11,3% / +7,4% de throughput sob saturação. Ou seja, o ganho aparece **em serving com batch contínuo**, não em batch 1.
- **Código:** https://github.com/AKafakA/sglang-vskipper/tree/vskipper-ref

#### 2.9 Apple Silicon / dispositivos pessoais
- **Nenhum trabalho de salto de camada/MLP com roteador implementado em MLX foi encontrado [busca negativa, não exaustiva].**
- O mais próximo: *Practical activation sparsity for quantized LLMs on personal devices*. R. Wang, K. Shu, F. X. Lin. arXiv 2511.04477, v1 em 06/11/2025. Esparsidade de ativação dinâmica em **llama.cpp**, avaliada em 5 dispositivos, incluindo GPUs Apple (chip não especificado). Gera tokens 1,45–2,15× mais rápido que o llama.cpp com 50% de esparsidade.
- Contexto da Apple: *LLM in a flash* (Alizadeh et al., arXiv 2312.11514, ACL 2024) usa esparsidade de ativação para carregar pesos da flash, com 4–5× em CPU e 20–25× em GPU **contra carregamento ingênuo**. O cenário é de memória limitada e não é comparável ao do VETOR.

#### 2.10 Baselines estáticos de poda de camadas
- **ShortGPT**: *ShortGPT: Layers in Large Language Models are More Redundant Than You Expect*. Men et al. arXiv 2403.03853, 06/03/2024. Remove camadas inteiras pela métrica *Block Influence* (1 − cosseno entre a entrada e a saída da camada). Sem treino.
- **LaCo**: *LaCo: Large Language Model Pruning via Layer Collapse*. Yang, Cao, Zhao. arXiv 2402.11187, 17/02/2024. Findings of EMNLP 2024. Colapsa as camadas finais numa anterior e mantém >80% do desempenho médio com 25–30% de poda.
- Esses trabalhos dão speedup real "de graça", porque o grafo fica estático, e são o contraponto natural ao baseline "estático best-layer" do VETOR.

#### 2.11 Mais recentes, citados para completude (lidos só no resumo)
- **DASH** (arXiv 2505.17420, 23/05/2025; Yang et al.): salto por token formulado como MDP, com "mecanismo de compensação" e avaliação assíncrona da política. Sem números de speedup no resumo nem código.
- **LayerRoute** (arXiv 2609.13682, 12/09/2026; autor único): roteador por bloco com LoRA rank 8 em Qwen2.5-0.5B. Speedup real **1,02–1,06×** em A100. Granularidade aparentemente por amostra **[não verificado]**.

### 3. Tabela-resumo

| Trabalho | Ano/venue | O que pula | Base | Roteador | Granularidade | KV dos pulados | Speedup **real** reportado | Código |
|---|---|---|---|---|---|---|---|---|
| MoD | 2024, preprint | bloco inteiro | treinada do zero | linear + top-k | token | não gera | ~50% por passo de amostragem (TPU) | não |
| LayerSkip | ACL 2024 | camadas finais (saída + verificação) | ajustada | nenhum | token | reutiliza (especulação) | 1,8–2,16× (H100/A100) | sim (CC-BY-NC) |
| CALM | NeurIPS 2022 | camadas finais | ajustada | confiança | token | copia o estado | ×3 **estimado** | sim (T5X) |
| SkipDecode | 2023, preprint | camadas iniciais | ajustada | regra por posição | posição | monotônico, sem recálculo | 2–5× **configurado** | [não verif.] |
| D-LLM | NeurIPS 2024 | camada inteira | LoRA | módulo de decisão | token | eviction por máscara | só FLOPs | sim |
| Router-Tuning/MindSkip | EMNLP 2025 | atenção (padrão) | congelada | linear σ(Wx) | seq. (padrão) / token | atenção pulada → menos KV | 1,21× (A6000, batch saturado) | sim |
| FFN-SkipLLM | EMNLP 2024 | **só FFN** | congelada (sem treino) | cosseno (regra) | token | intacto | não | [não encontrado] |
| **DiffSkip** | Findings ACL 2025 | **só FFN** | **congelada** | MLP com gargalo sobre ΔAttn | **token** | intacto | **nenhum na decodificação**; prefill modesto | só checkpoint HF |
| **FlexiDepth** | COLM 2025 | camada (atenção sem query + FFN → adaptador) | **congelada** | MLP com gargalo | token | K/V calculados | **nenhum** (mais lento, segundo vSkipper) | só checkpoint HF |
| Dr.LLM | ICLR 2026 | camada (pular/repetir) | congelada | MLP pequeno, MCTS | sequência | [n.v.] | não | sim (MIT) |
| PoLar | ICML 2026 | segmentos (pular/repetir) | pré-treinada [cong. n.v.] | preditor de 2,1M | entrada | [n.v.] | não | sim |
| Deja Vu | ICML 2023 | cabeças/neurônios | congelada | preditores MLP | token | n/a | >2× vs FasterTransformer (GPU) | sim |
| CATS | COLM 2024 | ativações do MLP | opcional FT | limiar | token | n/a | ~15% (kernel GPU) | [n.v.] |
| vSkipper | 2026, preprint | (sistema para skippers) | — | — | token | paged KV | −13,6 a −36,8% de latência em serving | sim |

### 4. Síntese

**(1) O que já está estabelecido, e portanto não pode ser apresentado como original.**
O roteamento aprendido **por token e por camada** sobre um LLM pré-treinado **congelado**, com roteador leve e perda de esparsidade, já está bem documentado: FlexiDepth (COLM 2025), DiffSkip (Findings ACL 2025) e Router-Tuning (EMNLP 2025). Pular **somente o MLP/FFN** por token também já existe, tanto sem treino (FFN-SkipLLM, EMNLP 2024) quanto com roteador treinado (DiffSkip). Também já existe a substituição do bloco pulado por um **compensador/adaptador barato** (DiffSkip e FlexiDepth: um FFN com dimensão intermediária 16× menor). Há ainda evidência publicada de que o roteamento dinâmico supera a poda estática no mesmo orçamento (ShortGPT/LaCo como baselines em vários desses trabalhos). Por isso o VETOR **não pode** reivindicar originalidade em "roteador por token em modelo congelado", em "MLP-only skip" nem em "compensador no lugar do bloco pulado".

**(2) A lacuna entre FLOPs e tempo real em batch 1.**
É um problema conhecido e quase universal. FlexiDepth e DiffSkip relatam abertamente **nenhum speedup na decodificação** em GPU, por fluxo de controle, acesso irregular à memória e a necessidade de buscar os pesos do FFN *e* do adaptador quando o batch mistura decisões. O vSkipper mediu o FlexiDepth **14,6–21% mais lento** que a base. Os poucos ganhos reais publicados vêm de: (a) grafos estáticos ou decisões por sequência (Router-Tuning, 1,21× com batch saturado; poda estática); (b) especulação, que transforma a saída antecipada em verificação em lote (LayerSkip, ~2×); (c) kernels dedicados para esparsidade (Deja Vu, CATS, llama.cpp esparso); ou (d) engines de serving que reagrupam tokens (vSkipper). Em batch 1 a decodificação é limitada pela banda de memória. Pular um MLP só economiza tempo se os pesos daquele MLP **deixarem de ser lidos** e se a decisão não forçar uma sincronização host–device. A sincronização GPU→CPU em cada decisão, que é exatamente o gargalo observado no VETOR com PyTorch/MPS, anula o ganho. **Não encontrei nenhum trabalho que demonstre speedup real de salto dinâmico por token em Apple Silicon/MLX.** Isso é uma lacuna empírica, não uma garantia de que um resultado seria positivo.

**(3) Arte anterior mais próxima do VETOR.**
É o **DiffSkip** (Luo, Wang, Yan, Findings ACL 2025): base congelada, salto **só do FFN**, **por token**, roteador pequeno (MLP com gargalo), compensador ("adaptor") no lugar do FFN pulado e perda de esparsidade. As diferenças em relação ao VETOR são: (i) a **entrada do roteador** (ΔAttn no DiffSkip; no VETOR, o estado oculto antes do MLP, h = x + ΔAttn, que contém ΔAttn mas não isolado); (ii) a **forma do compensador** (FFN pequeno não linear no DiffSkip; residual **low-rank linear de rank 32** no VETOR); (iii) a escala (8B no DiffSkip; 135M/1.7B no VETOR). A ablação do próprio DiffSkip mostra o adaptador linear bem abaixo do FFN pequeno (85,8% vs 99,0% de retenção). Isso é um alerta direto para o compensador linear do VETOR e uma ablação natural a fazer. O FlexiDepth (mesmo grupo) é o segundo mais próximo. A diferença é que ele também mexe na atenção (omite a query) e decide na granularidade da camada. O FFN-SkipLLM é o precedente "MLP-only sem treino e sem compensador". Uma contribuição que o VETOR ainda pode defender honestamente é um estudo controlado em escala pequena (135M→1.7B), com comparação ao estático best-layer no mesmo orçamento e medição honesta de tempo em Apple Silicon. Não é o mecanismo em si.

**(4) Baselines mais viáveis.**
1. **DiffSkip reimplementado** dentro do pipeline do VETOR (SmolLM2-1.7B, mesma divisão DEV/FINAL e mesmo orçamento de ~23%): roteador sobre ΔAttn e adaptador FFN pequeno (d_ff/16), com a mesma perda. É a comparação mais informativa e barata, porque troca só dois componentes. Não depende do código oficial, que só existe como checkpoint de 8B.
2. **FFN-SkipLLM (sem treino)** como piso dinâmico sem compensador. Também é trivial de implementar (cosseno entrada/saída do MLP, k camadas, warm-up), complementando o baseline estático best-layer que o VETOR já tem.

Como alternativa com código oficial pronto, o **Router-Tuning** (https://github.com/CASE-Lab-UMD/Router-Tuning, roteador linear com base congelada) pode ser configurado para MLP por token. Porém os próprios autores relatam que pular MLP é o caso ruim do método, e o código pressupõe CUDA/HF **[compatibilidade com MPS não verificada]**. LayerSkip, MoD, Dr.LLM e PoLar não servem como baseline direto: exigem retreino, são por sequência ou têm outro objetivo.

### Fontes
- MoD: https://arxiv.org/abs/2404.02258
- LayerSkip: https://arxiv.org/abs/2404.16710 · https://github.com/facebookresearch/LayerSkip
- DiffSkip: https://aclanthology.org/2025.findings-acl.377 · https://huggingface.co/xuan-luo/DiffSkip-Llama-3-8B-Instruct
- FlexiDepth: https://arxiv.org/abs/2503.23798 · https://huggingface.co/xuan-luo/FlexiDepth-Llama-3-8B-Instruct
- Dr.LLM: https://arxiv.org/abs/2510.12773 · https://github.com/parameterlab/dr-llm
- PoLar: https://arxiv.org/abs/2606.06574 · https://github.com/tianyi-lab/PoLar · precursor https://arxiv.org/abs/2507.07996
- CALM: https://arxiv.org/abs/2207.07061
- SkipDecode: https://arxiv.org/abs/2307.02628
- D-LLM: https://proceedings.nips.cc/paper_files/paper/2024/file/03469b1a66e351b18272be23baf3b809-Paper-Conference.pdf
- Router-Tuning: https://arxiv.org/abs/2410.13184 · https://github.com/CASE-Lab-UMD/Router-Tuning
- FFN-SkipLLM: https://arxiv.org/abs/2404.03865
- Deja Vu: https://arxiv.org/abs/2310.17157 · https://github.com/FMInference/DejaVu
- CATS: https://arxiv.org/abs/2404.08763
- vSkipper: https://arxiv.org/abs/2609.37062
- Esparsidade em dispositivos pessoais: https://arxiv.org/abs/2511.04477 · LLM in a flash: https://arxiv.org/abs/2312.11514
- ShortGPT: https://arxiv.org/abs/2403.03853 · LaCo: https://arxiv.org/abs/2402.11187
- DASH: https://arxiv.org/abs/2505.17420 · LayerRoute: https://arxiv.org/abs/2609.13682
