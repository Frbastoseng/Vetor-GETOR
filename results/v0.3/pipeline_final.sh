#!/bin/bash
# Pipeline final da v0.3: espera a fila 4 (por PID), mede desempenho (Fase 5) e roda a avaliacao FINAL (Fase 6).
cd "$(dirname "$0")/../.."
while ps -p "$1" >/dev/null 2>&1; do sleep 30; done
export SSL_CERT_FILE="$(.venv/bin/python -m certifi)" HF_ALLOW_CODE_EVAL=1
/usr/bin/caffeinate -i -s .venv/bin/python -u scripts/v3_05_perf.py --repeats 15 --tags A0_r8_task_g1 A_r8_task_g1 C_r8_task_g1 B_r8_rep4_task_g1 B0_r8_rep4_task_g1 LORA_r1_task_g1_lr1e3 > results/v0.3/perf.log 2>&1
/usr/bin/caffeinate -i -s .venv/bin/python -u scripts/v3_06_final.py --only FULL A A0 LORA < /dev/null > results/v0.3/final.log 2>&1
echo PIPELINE_FIM >> results/v0.3/final.log
