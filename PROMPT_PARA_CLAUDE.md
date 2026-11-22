# PROMPT — Implementação do Projeto VETOR no Mac

Você será meu pesquisador de ML aplicado, engenheiro de inferência PyTorch/Apple Silicon e responsável por testes de software. Quero implementar do zero o experimento documentado em `ESPECIFICACAO_TECNICA.md`, dentro desta pasta. **Leia integralmente essa especificação antes de escrever código.**

## Contexto

A hipótese: adicionar um vetor-guia probabilístico, condicionado aos estados intermediários do token, para decidir quais blocos MLP/FFN de um modelo Transformer pré-treinado devem ser executados. A meta é diminuir o tempo real de decodificação com degradação controlada. O projeto é investigação experimental, não declare resultado positivo antes de medir.

Equipamento: MacBook Pro Apple Silicon M4 Max, 128 GB de memória unificada, execução inicial no PyTorch MPS com fallback CPU. Modelo inicial: `HuggingFaceTB/SmolLM2-135M-Instruct` (Apache-2.0, arquitetura Llama, 30 camadas). Não usar Ollama/GGUF como backend da primeira experiência; precisamos instrumentar `model.model.layers[i].mlp`.

## Regras de implementação

1. Trabalhe na pasta atual; mantenha todas as modificações dentro do projeto. Não alterar dados de outras pastas.
2. Use um ambiente Python isolado e registre as versões exatas das dependências. Verifique PyTorch/MPS e API real do Transformers instalado antes de desenvolver adaptadores.
3. Nunca reescreva ou modifique o checkpoint original. Congele pesos do LLM. Treine apenas a pequena rede-guia, depois de validação.
4. Preserve a atenção e o KV cache em todas as camadas; ignore somente MLPs inteiros quando o controle permitir.
5. A execução otimizada deve **pular a chamada do MLP original**; mascarar sua saída após calculá-la não é ganho de inferência.
6. No prefill, calcule todos os MLPs; capture as entradas normalizadas de MLP do último token de cada camada para alimentar o guia.
7. Use uma abordagem em etapas com testes executados e comprovados antes de prosseguir para a seguinte. Não implemente otimizações não verificadas.
8. Faça medidas reais com sincronização MPS apenas nos limites do intervalo medido. Proibidas 30 conversões `.item()` por token no caminho rápido.
9. Separe dados train/val/test por documento; calibre hiperparâmetros somente em train/val. Não usar o conjunto test durante decisões de arquitetura.
10. Registre logs, arquivos CSV/JSON e gere um relatório comparativo com os resultados medidos, mesmo que o guia fique mais lento ou reduza a qualidade.

## Plano de entrega passo a passo

**Etapa 0** — Leia e revise a especificação. Inspecione SO, Python, dispositivo, memória livre e diretório do projeto. Crie `pyproject.toml`, `README.md`, `.gitignore`, `.venv`, lock de dependências e estrutura de pacotes. Teste carregamento de modelo e tokenizer. Mostre `config` e path real de MLP.

**Etapa 1** — Implemente `FULL` reproduzível, geração greedy com KV cache, dataset pequeno, métricas de prefill e decode, testes de equivalência com `generate` quando aplicável. Garanta tempo e saída reproduzíveis em CPU e MPS (tolerância numérica adequada).

**Etapa 2** — Implemente wrapper MLP reversível, captura de entrada, `ablate_last` e `skip_decode`. Faça testes que confirmem zero chamadas ao MLP durante skip; verifique que prefill continua completo e que não há alteração de pesos. Calcule curva de sensibilidade dos 30 MLPs em amostras de texto.

**Etapa 3** — Gere dataset de professor com vetor `z[l,t]`, KL e DeltaNLL para omissão isolada do MLP no último token. Tenha split, seed e exportações. O label safe inicial usa KL <= 0.02 e DeltaNLL <= 0.05 nats, ajustável pela validação.

**Etapa 4** — Implemente `RANDOM` e `STATIC`; compare sob o mesmo percentual de MLPs realmente omitidos. Se ignorar MLPs não reduzir o tempo mesmo em máscara estática, diagnostique o gargalo de kernel/sincronização antes de construir um roteador complexo.

**Etapa 5** — Treine a rede-guia (projetor 576->16 + embedding de camada + cabeça sigmoide; arquitetura exata na especificação). Restrinja parâmetros e regularize. Teste separadamente generalização de previsão de risco e qualidade da geração com múltiplas omissões. Garanta que o roteador só veja dados disponíveis antes da decisão.

**Etapa 6** — `GUIDED-S`: decidir a máscara uma vez no prefill, aplicar em todos os tokens decodificados. `GUIDED-D`: atualizar máscara a cada K tokens a partir dos estados capturados no token anterior, usando K em 1,4,8,16. Teste a segurança causal: nenhuma decisão usa o token futuro.

**Etapa 7** — Compare `FULL`, `RANDOM`, `STATIC`, `GUIDED-S` e `GUIDED-D` em texto/continuação held-out. Métricas obrigatórias: MLPs omitidos, FLOPs teóricos evitados, delta NLL/perplexidade, KL, concordância de tokens, tok/s, tempo de prefill, p95, memória, overhead do guia e relatório da curva qualidade/velocidade.

## Requisitos de comunicação

- Antes de mudar um componente crítico, explique a escolha técnica e o teste que comprova a correção, mas evite pedir aprovação para detalhes triviais.
- Execute cada teste possível na máquina e informe resultado verdadeiro; **não escreva testes 'passando' sem executar**.
- Não afirme speedup até apresentar medidas sincronizadas e replicáveis.
- Faça commits locais pequenos com mensagens descritivas, caso exista repositório Git inicializado e isso seja permitido.
- Ao término de cada etapa, entregue: arquivos alterados, comandos reproduzíveis, testes PASS/FAIL e próximos pontos técnicos.
- Se a API do Transformers estiver diferente do plano, adapte às classes reais instaladas e registre exatamente o que mudou.

**Comece agora pela Etapa 0 e Etapa 1**, garantindo que o modelo original pode ser reproduzido e cronometrado de forma confiável antes de implementar a intervenção. Depois avance pelas etapas conforme o resultado dos testes, preservando todo o trabalho anterior.
