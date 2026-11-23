"""v0.2 / avaliacao FINAL (uma unica vez) no conjunto final pre-registrado.

Le a configuracao congelada em configs/v2_decision.json, que precisa estar COMMITADO e sem alteracoes
locais (o script recusa rodar caso contrario). Avalia todas as politicas em todos os k com o mesmo
protocolo da validacao. Politicas historicas da v0.1 (GUIDED-S, GUIDED-D1, ZERO) via decode real.
Saida: results/v0.2/final_eval.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import torch

from vetor.model_adapter import ROOT, env_info, load_config, load_model, pick_device, seed_all, write_json
from vetor.v2.data import load_split
from vetor.v2.engine import patch_v2, set_comps, unpatch_v2, zero_comps
from vetor.v2.search import eval_batch, make_windows, masks_from_sets, summarize

sys.path.insert(0, str(ROOT / "scripts"))
from v2_03_teacher import load_comps, load_router  # noqa: E402

DEC = ROOT / "configs" / "v2_decision.json"
P, NW = 64, 64


def require_committed():
    rel = str(DEC.relative_to(ROOT))
    st = subprocess.run(["git", "status", "--porcelain", rel], cwd=ROOT, capture_output=True, text=True).stdout
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout
    if not tracked.strip() or st.strip():
        raise SystemExit("configs/v2_decision.json precisa estar commitado e sem alteracoes antes da avaliacao final")
    return subprocess.run(["git", "log", "-1", "--format=%H %cI", "--", rel], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def v01_historical(model, wraps2, wins, device, ks):
    """GUIDED-S e GUIDED-D1 da v0.1 (guia v0.1, ZERO) com o codigo da v0.1 (decode real)."""
    from vetor.decode import run_decode
    from vetor.guide import VectorGuide
    from vetor.policies import FullPolicy, GuidedPolicy
    from vetor.wrappers import patch_model, unpatch_model
    unpatch_v2(model)
    wraps = patch_model(model)
    ck = torch.load(ROOT / "results" / "guide.pt", weights_only=False)
    g = VectorGuide(model.config.hidden_size, len(wraps), ck["dim"])
    g.load_state_dict(ck["state_dict"])
    g = g.to(device).eval()
    out = {}
    for k in ks:
        for name, K in (("v0.1-GUIDED-S", None), ("v0.1-GUIDED-D1", 1)):
            nll, nf = [], []
            for w in wins:
                w = w.to(device)
                full = run_decode(model, wraps, w[:P].view(1, -1), NW, FullPolicy(len(wraps)), forced=w[P:], collect_logp=True)
                pol = GuidedPolicy(len(wraps), k, g, K=K)
                r = run_decode(model, wraps, w[:P].view(1, -1), NW, pol, forced=w[P:], collect_logp=True)
                t = w[P + 1:].view(-1, 1)
                nll.append(float(-r["logp"][1:].gather(-1, t).sum()))
                nf.append(float(-full["logp"][1:].gather(-1, t).sum()))
            nll, nf = np.array(nll), np.array(nf)
            idx = np.random.default_rng(0).integers(0, len(nll), (2000, len(nll)))
            boot = nll[idx].sum(1) / nf[idx].sum(1) - 1
            out[f"{name}-ZERO@k{k}"] = {"rel_dnll_pct": float(100 * (nll.sum() / nf.sum() - 1)),
                                       "ci95": [float(100 * np.percentile(boot, 2.5)), float(100 * np.percentile(boot, 97.5))],
                                       "skips_per_token": float(k), "n_windows": len(wins)}
    unpatch_model(model)
    return out


def main():
    commit = require_committed()
    dec = json.loads(DEC.read_text())
    cfg = load_config()
    seed_all(cfg["seed"])
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    E = list(range(2, L - 2))
    T = P + NW - 1
    wins = make_windows(tok, load_split("final", allow_final=True), P, NW, dec["n_final_windows"], dec["window_seed"])
    ks = dec["ks"]
    orc = json.loads((ROOT / "results" / "v0.2" / "oracle_val.json").read_text())
    v01_rank = [l for l in torch.load(ROOT / "results" / "guide.pt", weights_only=False)["static_rank"] if l in E]
    rng = np.random.default_rng(dec["random_seed"])
    rand_sets = {k: [sorted(rng.choice(E, k, replace=False).tolist()) for _ in wins] for k in ks}
    rep = {"decision_commit": commit, "decision": dec, "env": env_info(device), "n_windows": len(wins),
           "policies": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    for comp in dec["comps"]:
        set_comps(wraps, load_comps(comp, L, d, device))
        sg = orc["comps"][comp]["static_greedy_order"]
        for k in ks:
            for pname, sets in (("STATIC-GREEDY", [sg[:k]] * len(wins)), ("STATIC", [v01_rank[:k]] * len(wins)),
                                ("RANDOM", rand_sets[k])):
                r = eval_batch(model, wraps, wins, P, device, masks=masks_from_sets(sets, T, P, L))
                rep["policies"][f"{pname}-{comp}@k{k}"] = summarize(r)
        for path in dec["routers"].get(comp, []):
            router, ck = load_router(ROOT / path, device)
            nm = ("GUIDED-DL" if ck["lagged"] else "GUIDED-T") + f"-{comp}"
            for k in ks:
                router.configure(E, k_max=int(-(-1.5 * k // 1)), tau=ck["tau_by_k"][str(float(k))])
                r = eval_batch(model, wraps, wins, P, device, router=router.masked, eligible=router.eligible)
                rep["policies"][f"{nm}@k{k}"] = summarize(r)
        print(f"{comp}: " + "  ".join(f"{n}={v['rel_dnll_pct']:.2f}%" for n, v in rep["policies"].items()
                                      if n.endswith(f"-{comp}@k6") or n == f"GUIDED-T-{comp}@k6"))
        write_json(ROOT / "results" / "v0.2" / "final_eval.json", rep)
    rep["policies"].update(v01_historical(model, wraps, wins, device, dec["v01_ks"]))
    rep["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(ROOT / "results" / "v0.2" / "final_eval.json", rep)
    for n, v in sorted(rep["policies"].items()):
        if "@k6" in n:
            print(f"{n:28s} {v['rel_dnll_pct']:7.2f}%  IC95 [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]  omissoes/token {v['skips_per_token']:.2f}")


if __name__ == "__main__":
    main()
