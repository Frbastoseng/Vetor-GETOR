"""v0.4 / Fase 2: FULL (1.7B) nas tarefas, metade DEV -> lista de tarefas aplicaveis (regra pre-registrada).
Saida: results/v0.4/full_dev_tasks.json, results/v0.4/applicability.json
"""
from __future__ import annotations

import os

from vetor.model_adapter import env_info, pick_device, seed_all, write_json
from vetor.v4.common import OUT, load
from vetor.v4.tasks4 import SUITE, run


def main():
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    seed_all(1234)
    device = pick_device("auto")
    model, tok = load("1.7B", device)
    res = run(model, tok, list(SUITE), "dev", device=device)
    res["env"] = env_info(device)
    write_json(OUT / "full_dev_tasks.json", res)
    appl = []
    for n, v in res["tasks"].items():
        s = SUITE[n]
        val = v["value"]
        thr = s["chance"] + (0.03 if not s.get("generative") else 0.02)
        appl.append({"task": n, "cat": s["cat"], "value": val, "chance": s["chance"], "threshold": thr,
                     "applicable": val is not None and val >= thr})
        print(f"{n:14s} {val:.4f}  limiar {thr:.3f}  aplicavel={val >= thr}")
    write_json(OUT / "applicability.json", appl)


if __name__ == "__main__":
    main()
