# Como contribuir / opinar no VETOR

Este é um projeto de **pesquisa aberta**. Resultados negativos são tão bem-vindos quanto positivos, e críticas ao método são a contribuição mais valiosa.

## Formas de participar

| Quero... | Onde |
|---|---|
| Apontar falha de método, vazamento de dados ou métrica enganosa | Issue **"Questionamento metodológico"** |
| Propor um experimento ou melhoria | Issue **"Proposta de experimento"** |
| Reportar bug ou divergência ao reproduzir | Issue **"Bug / reprodução"** |
| Discutir ideias abertas | aba **Discussions** |
| Enviar código | Pull Request (veja abaixo) |

## Reproduzir os resultados

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # ou: pip install -r requirements-lock.txt && pip install -e .
python -m pytest -q                # 36+ testes; -m "not model" pula os que baixam o SmolLM2
python scripts/00_sanity.py        # ambiente e modelo
python scripts/01_baseline.py      # FULL: equivalencia e tempos
python scripts/02_ablate.py        # professor (≈7 min em M4 Max)
python scripts/03_train.py         # guia (selecao so na validacao)
python scripts/04_evaluate.py --split test
python scripts/05_benchmark.py --kinds STATIC RANDOM GUIDED-S GUIDED-D --guide results/guide.pt
python scripts/06_report.py        # gera results/REPORT.md e graficos
```

Em outra máquina (CUDA, CPU, outro Mac), os números de **tempo** vão mudar, e isso é informação útil. Abra uma issue de reprodução com `results/00_sanity.json` e o CSV do benchmark.

## Regras do experimento (não negociáveis em PRs)

1. **O conjunto de teste não orienta decisões.** Toda escolha (alvo, limiar, pool, hiperparâmetros) é feita na validação. Um PR que ajuste algo olhando o teste será recusado.
2. **Pular de verdade.** Calcular o MLP e zerar depois não conta como otimização.
3. **Mesma taxa de omissão** ao comparar GUIDED com RANDOM e STATIC.
4. **Sincronização só nas bordas** dos trechos cronometrados; nada de `.item()` por camada.
5. **Relatar sempre os dados crus**, inclusive quando o guia perde.
6. Mudou algo que afeta resultados? Rode de novo e versione os novos JSON/CSV em `results/`. Não edite números à mão.

## Pull requests

- Uma mudança por PR, com descrição de *o que* muda, *por que* e *qual teste prova*.
- `python -m pytest -q` deve passar. O CI roda os testes em CPU automaticamente.
- Novos componentes críticos (wrapper, política, decode) precisam de teste novo.
