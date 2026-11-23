#!/bin/bash
# Executa uma fila de treinos + avaliacoes DEV da v0.3, um por linha: "tag | argumentos do v3_03_train.py"
# Uso: bash scripts/v3_queue.sh fila.txt
set -u
cd "$(dirname "$0")/.."
export SSL_CERT_FILE="$(.venv/bin/python -m certifi)" HF_ALLOW_CODE_EVAL=1
while IFS='|' read -r tag args; do
  tag="$(echo "$tag" | xargs)"; [ -z "$tag" ] && continue; case "$tag" in \#*) continue;; esac
  if [ ! -f "results/v0.3/runs/${tag}.pt" ]; then
    /usr/bin/caffeinate -i -s .venv/bin/python -u scripts/v3_03_train.py $args --tag "$tag" > "results/v0.3/runs_${tag}.log" 2>&1
  fi
  /usr/bin/caffeinate -i -s .venv/bin/python -u scripts/v3_04_eval_run.py --tag "$tag" < /dev/null > "results/v0.3/eval_${tag}.log" 2>&1
  grep -hE "params treinaveis|macro DEV|Traceback|Error" "results/v0.3/runs_${tag}.log" "results/v0.3/eval_${tag}.log"
done < "$1"
