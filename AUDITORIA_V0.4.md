# VETOR v0.4 — Auditoria da base experimental (Fase 0)

**Data:** 24/11/2026 · **Escopo:** implementação da v0.2 (motor, avaliação, professor, roteador, decode real), infraestrutura da v0.3 e reprodutibilidade pública.

**Método:**

- leitura do código;
- 52 testes automáticos existentes, todos passando;
- script novo `scripts/v4_00_audit.py`, executado nos **dois** modelos.

Resultados brutos: `results/v0.4/audit_135M.json` e `results/v0.4/audit_1.7B.json`.

## 1. Verificações automáticas

| # | Verificação | 135M | 1.7B | Situação |
|---|---|---|---|---|
| 1 | Modelo instrumentado sem omissões == original (logits) | dif. máx. 0,0 | 0,0 | ✅ |
| 2 | MLP omitido **não é executado**: contador no MLP original durante decode real com 7 (135M) ou 5 (1.7B) camadas omitidas | só a chamada do prefill; demais camadas, 1 + 44 | idem | ✅ |
| 3a | Modo mascarado em lote == decode real com KV (omissão estática, compensador LR) | 6,5e-5 | 1,3e-4 | ✅ (log-prob, float32) |
| 3b | Idem com roteador GUIDED-T / GUIDED-DL (decisões idênticas, camadas protegidas e prefill nunca omitidos) | 7,9e-5 / 2,4e-4 | 2,0e-4 / 2,0e-4 | ✅ |
| 4 | KV cache: decode FULL com cache == forward completo | 6,3e-5 | 2,0e-4 | ✅ |
| 5 | Alinhamento de tokens: NLL nas posições de decode == perda do Hugging Face com rótulos deslocados | dif. 0,0 | 0,0 | ✅ |
| 6 | FLOPs do MLP contados pelo PyTorch == fórmula `6·d·d_ff` | 5,31 M | 100,7 M | ✅ |

A igualdade dos itens 3a e 3b é o que sustenta medir qualidade no modo em lote. Nesse modo o MLP **é** calculado e depois substituído, e a v0.4 nunca o usa para medir velocidade (regra 9).

## 2. Problemas encontrados

| # | Problema | Impacto | Correção |
|---|---|---|---|
| A1 | **FLOPs do compensador subcontados na v0.2.** A fórmula `2·d·r + 2·d` conta só uma das duas multiplicações (`·V` e `·U`). O valor contado é `4·d·r` (73.728 no 135M com r = 32, contra 38.016 relatados). | O saldo de operações evitadas relatado (31,3 M FLOPs/token, 11,6% do modelo) passa a ~31,1 M (~11,4%). O impacto é pequeno, mas o número relatado estava errado. | A v0.4 conta FLOPs com `torch.utils.flop_counter`, não por fórmula. O relatório da v0.2 **não** é alterado e a correção fica registrada aqui. |
| A2 | **`n_el = 26` fixo no treino do roteador** (`train2.py`). A feature `n/|E|` seria normalizada por 26 no treino e por \|E\| no uso. | Nenhum na v0.2, porque no 135M \|E\| = 26. No 1.7B (\|E\| = 20) seria uma inconsistência silenciosa. | Virou parâmetro (padrão 26, que preserva a v0.2). Os scripts da v0.4 passam `len(E)`. |
| A3 | **Compensadores da v0.2 fora do Git.** O arquivo tinha 248 MB porque o `torch.save` gravava visões de matrizes d×d float64 inteiras. Os roteadores publicados dependem dele. | A v0.2 não era reproduzível publicamente. | `scripts/v4_01_export_v02_comps.py` exporta uma cópia float32 compacta (8,3 MB), verificada elemento a elemento, em `results/v0.2/compensators_f32.pt`. O carregador usa essa cópia quando o original não existe. |
| A4 | **Dependências não congeladas por inteiro.** O `requirements-lock.txt` não tinha `lm_eval` nem dependências da v0.3. | Reprodução da v0.3 imprecisa. | Lock regenerado por `pip freeze` (103 pacotes). |
| A5 | **Seleção de checkpoint da v0.3 inconsistente com a perda.** O critério de validação era `task + distill`, sem os pesos α e `task_weight` e sem o termo de estabilidade γ que entrava na perda. | Pode ter escolhido checkpoints diferentes dos que minimizam o objetivo treinado. Não muda a conclusão negativa da v0.3, que foi consistente em todas as variantes. | Registrado. A v0.4 seleciona o roteador pela própria métrica do critério (ΔNLL real na validação, no orçamento-alvo), como a v0.2. O código da v0.3 é preservado. |
| A6 | **Checkpoints da v0.3 fora do Git.** | Reprodução pública impossível. | Os checkpoints pequenos (A, A₀, B, C: ~4 MB no total) são versionados. Os de LoRA (146 MB cada, acima do limite do GitHub) ficam para anexo de release. |
| A7 | **TTFT e memória inconclusivos na v0.3.** O TTFT saiu menor que o do FULL, e o driver MPS não tinha resolução. | Os critérios P2 e P3 foram julgados por argumento, não por medida. | Na v0.4: aquecimento **por configuração** antes de cada bloco; ordem alternada; ≥ 15 repetições com sincronização; IC por bootstrap. Memória medida por `torch.mps.current_allocated_memory()` (ou `mx.get_peak_memory()` no MLX) **e** pela soma dos bytes dos tensores adicionados. |
| A8 | **Pouca cobertura de variabilidade.** A v0.2 tinha 1 semente de roteador; a v0.3, 2 sementes. | Os intervalos de confiança cobriam só a amostragem de janelas, não o treino. | A v0.4 usa 3 sementes por roteador e RANDOM com 3 sementes. A variabilidade dos rótulos do professor continua não coberta (limitação declarada). |
| A9 | **Critério de superioridade da v0.2 por "IC sem sobreposição".** | É conservador, mas não é um teste da diferença. | A v0.4 usa bootstrap pareado da diferença. |
| A10 | **τ calibrado em 64 das 200 janelas de validação**, que também entram na avaliação de validação. | Pequeno otimismo na validação, sem efeito no conjunto final. | Mantido, como na v0.2, e declarado. |
| A11 | **Taxa de omissão diferente entre regimes:** 6,06 por token em teacher forcing e 8,04 na geração livre. | A qualidade relatada na v0.2 vale para 6,06, e a velocidade foi medida a 8,04. | Investigada na Fase 1 (`REPRODUCAO_V0.2.md`). |

## 3. Itens verificados sem problema

- **Métrica ΔNLL relativo:** `Σ NLL / Σ NLL_FULL − 1` sobre as mesmas posições de decode. O bootstrap por janela respeita a dependência entre os tokens de uma janela.
- **Professor:** `label_window` inverte exatamente uma célula (t, l) e mantém as demais decisões. Há teste que compara o rótulo com a avaliação direta (`test_teacher_label_matches_direct_evaluation`).
- **Causalidade do roteador:** mudar tokens futuros não altera decisões nem logits passados (`test_traversal_router_is_causal`).
- **Bloqueio do conjunto final da v0.2:** só lido com `allow_final=True`, com hash por artigo verificado. A avaliação final exige a decisão commitada.
- **GUIDED-T no decode real da v0.2:** faz 1 sincronização GPU→CPU por camada elegível (`.item()`). Foi relatado corretamente como inviável para velocidade e não é usado como implementação final.
