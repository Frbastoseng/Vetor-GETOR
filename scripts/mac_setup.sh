#!/bin/sh
set -eu

if [ ! -f "pyproject.toml" ]; then
  echo "Execute este script a partir da raiz do projeto VETOR."
  exit 1
fi
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers numpy pandas scikit-learn safetensors datasets tqdm pytest
printf '\nAmbiente instalado. Verificando MPS...\n'
python -c 'import torch; print("PyTorch:", torch.__version__, "| MPS disponivel:", torch.backends.mps.is_available())'
printf '\nExecutando testes unitarios...\n'
PYTHONPATH=src python -m pytest -q
printf '\nPara iniciar a ablacao em modelo real:\n'
printf 'source .venv/bin/activate && python scripts/02_ablate_smoke.py\n'
