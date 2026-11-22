# VETOR — Experimento de inferência com guia probabilístico

Este diretório é o pacote inicial da pesquisa VETOR: especificação, prompt de implementação para o Claude e um núcleo matemático testável que **não depende de baixar modelos**.

## Arquivos principais

- `ESPECIFICACAO_TECNICA.md` — protocolo completo e formulação matemática.
- `PROMPT_PARA_CLAUDE.md` — instruções de engenharia e execução para o Claude Code.
- `src/vetor/core.py` — exemplo inicial de wrapper MLP e classificador-guia em PyTorch.
- `tests/test_core.py` — testes isolados do núcleo, sem modelo da Hugging Face.

## Pré-requisitos no Mac

```bash
mkdir -p ~/Projetos/vetor
cd ~/Projetos/vetor
# Descompacte o pacote aqui. Depois:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch pytest
PYTHONPATH=src python -m pytest -q
```

Esses comandos testam somente o núcleo, **não demonstram speedup em LLM**. Para iniciar o experimento real, o Claude deve instalar e travar também `transformers`, `datasets`, `numpy`, `pandas`, `scikit-learn`, `safetensors` e dependências de experimentação, conforme compatibilidade do Mac.

## Importante

A integração completa com SmolLM2, rotulagem, treinamento, KV-cache e benchmark deve ser programada e verificada no seu Mac. O código incluído é um ponto de partida isolado, não uma implementação pronta de inference routing validada em modelo real.

## Atalho para preparar o ambiente completo no Mac

```bash
cd ~/Projetos/vetor
sh scripts/mac_setup.sh
source .venv/bin/activate
python scripts/02_ablate_smoke.py --max-layers 6
```

O script `02_ablate_smoke.py` baixa o modelo, compara a distribuição do próximo token com o modelo completo e com cada MLP testado isoladamente. Ele gera `results/ablation_smoke.json`. **Esse teste mede sensibilidade, não velocidade.** Não foi executado no Mac aqui; o Claude deve verificar a compatibilidade das bibliotecas antes de avançar.
