"""v0.4 / Fase 2: avaliacao das politicas no 1.7B (mesmo codigo para DEV e FINAL).

  python scripts/v4_15_eval.py --split dev     (validacao/val-eval + metade DEV das tarefas + prompts dev)
  python scripts/v4_15_eval.py --split final   (exige configs/v4_decision.json commitado; avaliacao UNICA)
Partes: --parts nll tasks gen (padrao: todas). Saida: results/v0.4/eval_{split}_{part}.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

from vetor.model_adapter import ROOT, env_info, pick_device, seed_all, write_json
from vetor.v2.engine import Ctrl, patch_v2, set_comps, zero_comps
from vetor.v2.fasteval import logprobs_D
from vetor.v2.search import eval_batch, masks_from_sets
from vetor.v4.attach import Policy
from vetor.v4.common import OUT, eligible, load
from vetor.v4.data4 import load_prompts
from vetor.v4.pipe import NW, P, domain_of, load_comps, random_sets, windows
from vetor.v4.routers import load_router4
from vetor.v4.stats4 import boot_rel, rep4
from vetor.v4.tasks4 import SUITE, run

DEC = ROOT / "configs" / "v4_decision.json"


def require_committed():
    rel = str(DEC.relative_to(ROOT))
    st = subprocess.run(["git", "status", "--porcelain", rel], cwd=ROOT, capture_output=True, text=True).stdout
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout
    if not tracked.strip() or st.strip():
        raise SystemExit("configs/v4_decision.json precisa estar commitado e sem alteracoes antes da avaliacao final")
    return subprocess.run(["git", "log", "-1", "--format=%H %cI", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def config(split):
    """DEV: configuracao provisoria (dos arquivos de validacao). FINAL: so a decisao congelada."""
    if split == "final":
        return json.loads(DEC.read_text())
    return json.loads((OUT / "dev_config.json").read_text())


def layer_profile(model, wraps, router, wins, device, L):
    bits = []
    for i in range(0, len(wins), 16):
        x = torch.stack(wins[i:i + 16]).to(device)[:, :-1]
        c = Ctrl(start=P, L=L, router=router.masked, eligible=router.eligible)
        logprobs_D(model, wraps, x, P, c)
        bits.append(c.bits[:, P:].float().cpu())
    b = torch.cat(bits)
    per_tok = b.sum(-1)
    return {"by_layer": b.mean((0, 1)).tolist(), "by_position_quartile": [float(q.sum(-1).mean()) for q in b.chunk(4, 1)],
            "per_token_hist": np.bincount(per_tok.long().flatten().numpy(), minlength=12).tolist(),
            "per_window_mean": per_tok.mean(1).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "final"])
    ap.add_argument("--parts", nargs="+", default=["nll", "tasks", "gen"])
    args = ap.parse_args()
    fin = args.split == "final"
    commit = require_committed() if fin else None
    cfg = config(args.split)
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    seed_all(1234)
    device = pick_device("auto")
    model, tok = load("1.7B", device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    E = eligible(L)
    K, C = cfg["k"], cfg["comp"]
    kmax = int(-(-1.5 * K // 1))
    wraps = patch_v2(model, zero_comps(L, d, device))
    comps = {c: load_comps(c, L, d, device) for c in {C, "ZERO"}}
    T = P + NW - 1
    meta = {"split": args.split, "decision_commit": commit, "config": cfg, "env": env_info(device)}

    def routers(key):
        out = []
        for path in cfg["routers"][key]:
            r, ck = load_router4(ROOT / path, device)
            out.append((r, ck))
        return out

    if "nll" in args.parts:
        wins = windows(tok, "final", allow_final=True, seed=2026) if fin else windows(tok, "val", part="eval", seed=12)
        doms = domain_of(tok, "final", allow_final=True, seed=2026) if fin else domain_of(tok, "val", part="eval", seed=12)
        assert len(doms) == len(wins)
        rep = {**meta, "n_windows": len(wins), "domains": doms, "policies": {}}
        t0 = time.time()
        for comp in (C, "ZERO"):
            set_comps(wraps, comps[comp])
            sg = cfg["static_greedy_order"][comp]
            for k in cfg["ks_report"]:
                rr = eval_batch(model, wraps, wins, P, device, masks=masks_from_sets([sg[:k]] * len(wins), T, P, L), chunk=16)
                rep["policies"][f"STATIC-GREEDY-{comp}@k{k}"] = {"nll": rr["nll"].tolist(), "nll_full": rr["nll_full"].tolist(),
                                                                "skips": float(rr["skips"].mean())}
                for s in cfg["random_seeds"]:
                    rr = eval_batch(model, wraps, wins, P, device, masks=masks_from_sets(random_sets(E, len(wins), k, s), T, P, L), chunk=16)
                    rep["policies"][f"RANDOM-{comp}@k{k}_s{s}"] = {"nll": rr["nll"].tolist(), "skips": float(rr["skips"].mean())}
            for key in ([f"GUIDED-T-{comp}", f"GUIDED-DL-{comp}"] if comp == C else [f"GUIDED-T-{comp}"]):
                for i, (r, ck) in enumerate(routers(key)):
                    for k in cfg["ks_report"]:
                        if str(float(k)) not in ck["tau_tf"]:
                            continue
                        r.configure(E, k_max=int(-(-1.5 * k // 1)), tau=ck["tau_tf"][str(float(k))])
                        rr = eval_batch(model, wraps, wins, P, device, router=r.masked, eligible=r.eligible, chunk=16)
                        rep["policies"][f"{key}@k{k}_s{ck['seed']}"] = {"nll": rr["nll"].tolist(), "skips": float(rr["skips"].mean()),
                                                                     "skips_per_window": rr["skips"].mean(1).tolist()}
                    if key == f"GUIDED-T-{C}":
                        r.configure(E, k_max=kmax, tau=ck["tau_tf"][str(float(K))])
                        rep.setdefault("profiles", {})[f"s{ck['seed']}"] = layer_profile(model, wraps, r, wins, device, L)
                        r.configure(E, k_max=kmax, tau=ck["tau_gen"][str(float(K))])
                        rr = eval_batch(model, wraps, wins, P, device, router=r.masked, eligible=r.eligible, chunk=16)
                        rep["policies"][f"{key}@k{K}_tau_gen_s{ck['seed']}"] = {"nll": rr["nll"].tolist(), "skips": float(rr["skips"].mean())}
            print(f"nll {comp} ({time.time() - t0:.0f}s)", flush=True)
        full = np.array(rep["policies"][f"STATIC-GREEDY-{C}@k{K}"]["nll_full"])
        rep["summary"] = {n: {**boot_rel(v["nll"], full, n_boot=2000), "skips": v["skips"]} for n, v in rep["policies"].items()}
        write_json(OUT / f"eval_{args.split}_nll.json", rep)
        for n, v in rep["summary"].items():
            print(f"  {n:34s} {v['rel_dnll_pct']:7.2f}%  [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]  omissoes {v['skips']:.2f}")

    set_comps(wraps, comps[C])
    pols = [("FULL", dict(kind="FULL"))]
    pols.append((f"STATIC-GREEDY-{C}@k{K}", dict(kind="STATIC", static_set=cfg["static_greedy_order"][C][:K])))
    for r, ck in routers(f"GUIDED-T-{C}"):
        pols.append((f"GUIDED-T-{C}@k{K}_s{ck['seed']}", dict(kind="GUIDED", router=r, tau_tf=ck["tau_tf"][str(float(K))],
                                                             tau_gen=ck["tau_gen"][str(float(K))], k_max=kmax)))

    if "tasks" in args.parts:
        names = [t["task"] for t in json.loads((OUT / "applicability.json").read_text()) if t["applicable"]]
        rep = {**meta, "tasks": names, "policies": {}}
        for name, kw in pols:
            t0 = time.time()
            with torch.inference_mode(), Policy(model, wraps, L, **kw) as pol:
                res = run(model, tok, names, args.split, allow_final=fin, device=device, policy=pol)
            rep["policies"][name] = res["tasks"]
            print(f"tarefas {name} ({time.time() - t0:.0f}s): " + "  ".join(
                f"{t}={v['value']:.3f}" for t, v in res["tasks"].items() if v["value"] is not None), flush=True)
            write_json(OUT / f"eval_{args.split}_tasks.json", rep)

    if "gen" in args.parts:
        groups = ["chat_en:final", "chat_pt:final"] if fin else ["chat_en:dev", "chat_pt:dev"]
        prompts = [(g.split(":")[0], p["prompt"]) for g in groups for p in load_prompts(g, allow_final=fin)]
        enc = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for _, p in prompts]
        tok.padding_side = "left"
        rep = {**meta, "n_prompts": len(prompts), "langs": [l for l, _ in prompts], "policies": {}}
        outs = {}
        for name, kw in pols:
            gen = []
            with torch.inference_mode(), Policy(model, wraps, L, **kw) as pol:
                pol.regime = "gen"
                for i in range(0, len(enc), 16):
                    b = tok(enc[i:i + 16], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                    g = model.generate(**b, max_new_tokens=128, do_sample=False, pad_token_id=2)
                    gen.extend([row[b["input_ids"].shape[1]:].cpu() for row in g])
                rate = pol.rate() if kw["kind"] != "FULL" else 0.0
            outs[name] = gen
            reps = [rep4(t.tolist()) for t in gen]
            rep["policies"][name] = {"skips_per_generated_token": rate, "rep4": reps,
                                     "degenerate_frac": float(np.mean([x >= 0.5 for x in reps])),
                                     "texts": [tok.decode(t, skip_special_tokens=True) for t in gen]}
        full = outs["FULL"]
        for name, gen in outs.items():
            n = [min(len(a), len(b)) for a, b in zip(gen, full)]
            rep["policies"][name]["agree_with_FULL"] = float(np.mean([(a[:m] == b[:m]).float().mean() for a, b, m in zip(gen, full, n)]))
            print(f"geracao {name}: omissoes/token {rep['policies'][name]['skips_per_generated_token']:.2f}  "
                  f"degeneradas {rep['policies'][name]['degenerate_frac']:.3f}  concordancia {rep['policies'][name]['agree_with_FULL']:.3f}",
                  flush=True)
        write_json(OUT / f"eval_{args.split}_gen.json", rep)


if __name__ == "__main__":
    main()
