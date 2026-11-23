"""v0.3: avalia uma execucao treinada nas tarefas APLICAVEIS da metade DEV e compara com o FULL (pareado).

  HF_ALLOW_CODE_EVAL=1 python scripts/v3_04_eval_run.py --tag A0_r8_kd
Saida: results/v0.3/runs/{tag}_dev.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from vetor.model_adapter import ROOT, load_config, load_model, pick_device, seed_all, write_json
from vetor.v3.bench import SUITE, run
from vetor.v3.modules import merge_lora
from vetor.v3.stats import by_category, paired_macro

sys.path.insert(0, str(ROOT / "scripts"))
RUNS = ROOT / "results" / "v0.3" / "runs"


def load_run(model, tag):
    from v3_03_train import build  # noqa: E402
    ck = torch.load(RUNS / f"{tag}.pt", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    b = build(model, a)
    with torch.no_grad():
        for i, p in enumerate(b["params"]):
            p.copy_(ck["best_params"][f"p{i}"].to(p.device))
            p.requires_grad_(False)
    if a.arch == "LORA":
        merge_lora(model)  # avaliado JA FUNDIDO (como seria usado)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", nargs="+", default=None)
    args = ap.parse_args()
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    a, b = load_run(model, args.tag)
    model.eval()
    appl = [r["task"] for r in json.loads((ROOT / "results" / "v0.3" / "applicability.json").read_text()) if r["applicable"]]
    tasks = args.tasks or appl
    res = run(model, tok, tasks, "dev", device=device)
    full = json.loads((ROOT / "results" / "v0.3" / "baseline_full_dev.json").read_text())
    cmp = paired_macro(res, full, tasks)
    cats = {t: SUITE[t]["cat"] for t in tasks}
    out = {"tag": args.tag, "args": vars(a), "n_params": b["n_params"], "split": "dev", "tasks": res["tasks"],
           "vs_full": {k: v for k, v in cmp.items()}, "vs_full_by_category": by_category(cmp, cats)}
    write_json(RUNS / f"{args.tag}_dev.json", out)
    print(f"[{args.tag}] macro DEV = {cmp['macro_a']:.2f} vs FULL {cmp['macro_b']:.2f}  "
          f"diff = {cmp['macro_diff_pp']:+.2f} p.p. IC95 [{cmp['ci95'][0]:+.2f}, {cmp['ci95'][1]:+.2f}]")
    for t, v in cmp["per_task"].items():
        print(f"   {t:16s} {v['a']:6.2f} vs {v['b']:6.2f}  ({v['diff']:+.2f})")


if __name__ == "__main__":
    main()
