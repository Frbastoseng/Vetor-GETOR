"""v0.3 / Fase 6: avaliacao UNICA na metade FINAL, com a decisao congelada em configs/v3_decision.json.

Recusa rodar se a decisao nao estiver commitada e sem alteracoes. Avalia cada modelo nas tarefas
aplicaveis (metade FINAL) e grava os acertos por exemplo em results/v0.3/final/{modelo}.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import torch

from vetor.model_adapter import ROOT, env_info, load_config, load_model, pick_device, seed_all, write_json
from vetor.v3.bench import run
from vetor.v3.modules import detach_all

sys.path.insert(0, str(ROOT / "scripts"))
from v3_04_eval_run import load_run  # noqa: E402

DEC = ROOT / "configs" / "v3_decision.json"
OUT = ROOT / "results" / "v0.3" / "final"


def require_committed():
    rel = str(DEC.relative_to(ROOT))
    st = subprocess.run(["git", "status", "--porcelain", rel], cwd=ROOT, capture_output=True, text=True).stdout
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout
    if not tracked.strip() or st.strip():
        raise SystemExit("configs/v3_decision.json precisa estar commitado e sem alteracoes")
    return subprocess.run(["git", "log", "-1", "--format=%H %cI", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def attach_v02(model, device):
    """VETOR v0.2 (GUIDED-T + LR32, k=6) no forward de sequencia inteira: roteador em todas as posicoes."""
    from v2_03_teacher import load_comps, load_router
    from vetor.v2.engine import Ctrl, patch_v2, set_comps, set_ctrl, zero_comps
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    set_comps(wraps, load_comps("LR32", L, d, device))
    r, ck = load_router(ROOT / "results" / "v0.2" / "router_LR32_r2_T.pt", device)
    r.configure(list(range(2, L - 2)), k_max=9, tau=ck["tau_by_k"]["6.0"])
    ctrl = Ctrl(start=0, L=L, router=r.masked, eligible=r.eligible)
    set_ctrl(wraps, ctrl)
    model.model.register_forward_pre_hook(lambda m, a, kw: setattr(ctrl, "bits", None), with_kwargs=True)
    return ctrl


def main():
    commit = require_committed()
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    dec = json.loads(DEC.read_text())
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    tasks = [r["task"] for r in json.loads((ROOT / "results" / "v0.3" / "applicability.json").read_text()) if r["applicable"]]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("FULL", None)] + [(f"{m}__{t}", t) for m, tags in dec["models"].items() if isinstance(tags, list) for t in tags] + [("V02", "V02")]
    for name, tag in jobs:
        path = OUT / f"{name}.json"
        if path.exists():
            continue
        t0 = time.time()
        model, tok = load_model(cfg, device)
        if tag == "V02":
            attach_v02(model, device)
        elif tag:
            load_run(model, tag)
        model.eval()
        res = run(model, tok, tasks, "final", allow_final=True, device=device)
        res.update({"name": name, "tag": tag, "decision_commit": commit, "env": env_info(device),
                    "seconds": time.time() - t0})
        write_json(path, res)
        print(f"{name}: " + "  ".join(f"{t}={v['value']:.4f}" for t, v in res["tasks"].items() if v["value"] is not None)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.mps.empty_cache() if device == "mps" else None


if __name__ == "__main__":
    main()
