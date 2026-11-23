"""v0.3: treina UMA configuracao (LLM congelado) e salva os modulos.

  python scripts/v3_03_train.py --arch A  --layers spaced8 --rank 8 --alpha 1 --tag A_r8_kd
  python scripts/v3_03_train.py --arch A0 --layers spaced8 --rank 8 --alpha 1 --tag A0_r8_kd
  python scripts/v3_03_train.py --arch LORA --rank 1 --alpha 1 --tag lora_kd
  python scripts/v3_03_train.py --arch B  --layers spaced8 --rank 8 --replace 4 --alpha 1 --tag B_kd
Arquiteturas: A (correcao + portao do guia), A0 (correcao, portao = 1), LORA (fundivel),
B / B0 (substitui k MLPs por MEAN [ordem STATIC-GREEDY da v0.2] + correcao com/sem guia).
Saidas: results/v0.3/runs/{tag}.pt (modulos) e {tag}.json (configuracao, curvas, custo).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from vetor.model_adapter import ROOT, env_info, load_config, load_model, pick_device, seed_all, write_json
from vetor.v3.modules import attach_lora, attach_vetor, n_params
from vetor.v3.train import batches, collate, distill_loss, gate_means, load_jsonl, set_modules_enabled

RUNS = ROOT / "results" / "v0.3" / "runs"
LAYER_SETS = {"spaced8": [3, 6, 9, 12, 15, 18, 21, 24], "spaced4": [6, 12, 18, 24],
              "late8": [20, 21, 22, 23, 24, 25, 26, 27], "early8": [2, 3, 4, 5, 6, 7, 8, 9]}


def mean_replacements(k: int, d: int) -> dict[int, torch.Tensor]:
    if k == 0:
        return {}
    orc = json.loads((ROOT / "results" / "v0.2" / "oracle_val.json").read_text())
    order = orc["comps"]["MEAN"]["static_greedy_order"][:k]
    ck = torch.load(ROOT / "results" / "v0.2" / "compensators.pt", weights_only=False)
    return {l: ck["mean"][l]["ybar"].float() for l in order}


def build(model, a):
    if a.arch == "LORA":
        lora = attach_lora(model, r=a.rank, seed=a.seed)
        return {"params": list(lora.parameters()), "n_params": n_params(lora), "lora": lora}
    layers = LAYER_SETS[a.layers]
    rep = mean_replacements(a.replace, model.config.hidden_size) if a.arch in ("B", "B0") else {}
    layers = [l for l in layers if l not in rep]
    mods = attach_vetor(model, layers, a.rank, guided=a.arch in ("A", "B"), replace=rep, seed=a.seed)
    ps = list(mods["corrections"].parameters()) + (list(mods["guide"].parameters()) if mods["guide"] else [])
    return {"params": ps, "n_params": n_params(mods["corrections"], mods["guide"]), "mods": mods,
            "layers": layers, "replaced": sorted(rep)}


def teacher_cache(split):
    p = ROOT / "data" / f"v3_teacher_top64_{split}.pt"
    return torch.load(p, weights_only=False)["examples"]


def step_losses(model, ids, att, lab, tcache, idx, a, device):
    pred = torch.zeros_like(lab)
    pred[:, :-1] = lab[:, 1:]
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    logits = model.lm_head(h[pred]).float()          # [N,V] so nas posicoes que preveem resposta
    tgt = ids[:, 1:][pred[:, :-1]]
    out = {"task": F.cross_entropy(logits, tgt)}
    if a.alpha > 0 and tcache is not None:
        ti = torch.cat([tcache[j][0] for j in idx]).long().to(device)
        tl = torch.cat([tcache[j][1] for j in idx]).float().to(device)
        tl = tl - torch.logsumexp(tl, -1, keepdim=True)   # renormaliza no top-K
        lsm = F.log_softmax(logits, -1)
        out["distill"] = (tl.exp() * (tl - lsm.gather(-1, ti))).sum(-1).mean()
    return out


def ref_batches(tok, n_tokens=256, seed=0):
    from vetor.v2.compensate import token_batches
    from vetor.v2.data import load_split
    ids = [tok(x["text"], add_special_tokens=False).input_ids for x in load_split("train")[:400]]
    return token_batches(ids, n_tokens, 2, 10 ** 9, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["A", "A0", "LORA", "B", "B0"])
    ap.add_argument("--layers", default="spaced8")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--replace", type=int, default=0)
    ap.add_argument("--task-weight", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    cfg = load_config()
    seed_all(a.seed)
    device = pick_device(cfg["device"])
    model, tok = load_model(cfg, device)
    b = build(model, a)
    for p in b["params"]:
        p.requires_grad_(True)
    tr = load_jsonl(ROOT / "data" / "v3_train.jsonl")[: a.max_train]
    va = load_jsonl(ROOT / "data" / "v3_val.jsonl")
    tc_tr = teacher_cache("train") if a.alpha > 0 else None
    tc_va = teacher_cache("val")
    enc, bl = batches(tr, tok, a.max_tokens, seed=a.seed)
    venc, vbl = batches(va, tok, a.max_tokens, shuffle=False)
    total = int(len(bl) * a.epochs)
    opt = torch.optim.AdamW(b["params"], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / total)))
    refs = ref_batches(tok, seed=a.seed) if a.gamma > 0 else None
    pad = tok.pad_token_id or 0

    def validate():
        model.eval()
        tot, n = {"task": 0.0, "distill": 0.0}, 0
        with torch.no_grad():
            for idx in vbl:
                ids, att, lab = collate(venc, idx, pad, device)
                ls = step_losses(model, ids, att, lab, tc_va, idx, argparse.Namespace(alpha=1.0), device)
                for k in tot:
                    tot[k] += float(ls[k])
                n += 1
        model.train()
        return {k: v / n for k, v in tot.items()}

    hist, best, best_state = [], None, None
    v0 = validate()
    hist.append({"step": 0, **{f"val_{k}": v for k, v in v0.items()}})
    print(f"[{a.tag}] params treinaveis={b['n_params']}  passos={total}  val inicial={v0}", flush=True)
    t0, step = time.time(), 0
    model.train()
    while step < total:
        for idx in bl:
            if step >= total:
                break
            ids, att, lab = collate(enc, idx, pad, device)
            ls = step_losses(model, ids, att, lab, tc_tr, idx, a, device)
            loss = a.task_weight * ls["task"] + a.alpha * ls.get("distill", 0.0)
            if a.gamma > 0:
                r = next(refs).to(device)
                with torch.no_grad():
                    set_modules_enabled(model, False)
                    lf = F.log_softmax(model(input_ids=r).logits.float(), -1)
                    set_modules_enabled(model, True)
                lp = F.log_softmax(model(input_ids=r).logits.float(), -1)
                loss = loss + a.gamma * (lf.exp() * (lf - lp)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(b["params"], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % a.eval_every == 0 or step == total:
                v = validate()
                gm = {k: float(x) for k, x in gate_means(model).items()}
                hist.append({"step": step, **{f"val_{k}": x for k, x in v.items()}, "gates": gm,
                             "elapsed_s": time.time() - t0})
                crit = v["task"] + v["distill"]
                print(f"[{a.tag}] passo {step}/{total} val task={v['task']:.4f} distill={v['distill']:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                if best is None or crit < best:
                    best = crit
                    best_state = {k: t.detach().clone().cpu() for k, t in
                                  ((f"p{i}", p) for i, p in enumerate(b["params"]))}
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save({"args": vars(a), "best_params": best_state, "layers": b.get("layers"),
                "replaced": b.get("replaced")}, RUNS / f"{a.tag}.pt")
    write_json(RUNS / f"{a.tag}.json", {"args": vars(a), "n_params": b["n_params"], "layers": b.get("layers"),
                                        "replaced": b.get("replaced"), "history": hist, "best_val": best,
                                        "env": env_info(device), "train_seconds": time.time() - t0})
    print(f"[{a.tag}] fim. melhor val (task+distill) = {best:.4f}")


if __name__ == "__main__":
    main()
