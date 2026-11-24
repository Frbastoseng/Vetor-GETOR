"""v0.3 / Fase 2: manifesto dev/final da bateria e baseline de qualidade do FULL (somente metade DEV).

  HF_ALLOW_CODE_EVAL=1 python scripts/v3_00_baseline.py [--tasks ...]
Saidas: configs/v3_eval_manifest.json (uma unica vez), results/v0.3/baseline_full_dev.json
"""
from __future__ import annotations

import argparse
import os
import time

from vetor.model_adapter import ROOT, env_info, load_config, load_model, pick_device, seed_all, write_json
from vetor.v3.bench import MANIFEST, SUITE, build_manifest, run, task_sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(SUITE))
    ap.add_argument("--model", default=None, help="outro modelo (ex.: professor) para referencia")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    if not MANIFEST.exists():
        sizes = task_sizes(list(SUITE))
        man = build_manifest(sizes)
        print("manifesto criado:", {t: (len(v["dev"]), len(v["final"])) for t, v in man["tasks"].items()})
    if args.model:
        cfg = {**cfg, "model": {"id": args.model, "revision": None, "dtype": "float32"}}
    model, tok = load_model(cfg, device)
    out_path = ROOT / "results" / "v0.3" / f"baseline_{args.tag}_dev.json"
    res = {"model": cfg["model"]["id"], "env": env_info(device), "split": "dev", "tasks": {}}
    if out_path.exists():
        import json
        res = json.loads(out_path.read_text())
    for t in args.tasks:
        if t in res["tasks"]:
            continue
        t0 = time.time()
        r = run(model, tok, [t], "dev", device=device)
        res["tasks"].update(r["tasks"])
        res["tasks"][t]["seconds"] = time.time() - t0
        v = r["tasks"][t]
        print(f"{t:45s} {SUITE[t]['cat']:13s} {v['metric']:32s} {v['value']}  n={v['n']}  ({time.time() - t0:.0f}s)",
              flush=True)
        write_json(out_path, res)


if __name__ == "__main__":
    main()
