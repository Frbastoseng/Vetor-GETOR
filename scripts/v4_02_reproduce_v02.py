"""v0.4 / Fase 1: reproducao controlada da v0.2 no SmolLM2-135M.

  --part eval    : reavalia os artefatos CONGELADOS da v0.2 no conjunto final da v0.2 (compensadores da copia
                   compacta publicada) e compara com results/v0.2/final_eval.json
  --part seeds   : retreina GUIDED-T e GUIDED-DL com 3 sementes a partir dos rotulos salvos; compara com
                   STATIC-GREEDY e RANDOM (mesmo compensador, k=6) por bootstrap pareado (val e final v0.2)
  --part labels  : regera os rotulos do professor (rodada 0) em 16 janelas e compara com os salvos
  --part regime  : investiga 6,06 (teacher forcing) vs 8,04 (geracao livre) omissoes/token
Saida: results/v0.4/repro/{part}.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from vetor.model_adapter import ROOT, env_info, pick_device, seed_all, write_json
from vetor.v2.compensate import make_comps
from vetor.v2.data import load_split
from vetor.v2.decode2 import decode_real
from vetor.v2.engine import Ctrl, patch_v2, set_comps, zero_comps
from vetor.v2.fasteval import kl_rows, logprobs_D
from vetor.v2.search import eval_batch, make_windows, masks_from_sets
from vetor.v2.teacher2 import concat
from vetor.v2.train2 import calibrate_tau, train_router
from vetor.v4.common import OUT, eligible, load
from vetor.v4.stats4 import boot_rel, paired_diff, rep4

sys.path.insert(0, str(ROOT / "scripts"))
from v2_03_teacher import load_router  # noqa: E402

P, NW, K = 64, 64, 6
R = OUT / "repro"
V2 = ROOT / "results" / "v0.2"


def comps_f32(name, L, d, device):
    if name == "ZERO":
        return zero_comps(L, d, device)
    ck = torch.load(V2 / "compensators_f32.pt", weights_only=False)
    r = int(name[2:])
    return make_comps("lr", ck["lr"][r], L, d, r, device)


def setup():
    seed_all(1234)
    device = pick_device("auto")
    model, tok = load("135M", device)
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    return device, model, tok, wraps, L, d, eligible(L)


def windows(tok, split):
    if split == "final":
        return make_windows(tok, load_split("final", allow_final=True), P, NW, 200, 2027)
    return make_windows(tok, load_split("val"), P, NW, 200, 12)


def static_sets(E, n, k, seed=7):
    o = json.loads((V2 / "oracle_val.json").read_text())["comps"]
    rng = np.random.default_rng(seed)
    # mesma ordem de sorteio da v0.2 (v2_07_final.py): conjuntos de todos os k em sequencia, k=3,5,6,8,10
    sets = {kk: [sorted(rng.choice(E, kk, replace=False).tolist()) for _ in range(n)] for kk in (3, 5, 6, 8, 10)}
    return o, sets[k]


def per_window(r):
    return {"nll": r["nll"].tolist(), "nll_full": r["nll_full"].tolist(), "skips": float(r["skips"].mean())}


def part_eval(device, model, tok, wraps, L, d, E):
    wins = windows(tok, "final")
    T = P + NW - 1
    ref = json.loads((V2 / "final_eval.json").read_text())["policies"]
    rep = {"n_windows": len(wins), "rows": {}}
    o, rsets = static_sets(E, len(wins), K)
    for comp in ("ZERO", "LR32"):
        set_comps(wraps, comps_f32(comp, L, d, device))
        pols = {f"STATIC-GREEDY-{comp}@k6": dict(masks=masks_from_sets([o[comp]["static_greedy_order"][:K]] * len(wins), T, P, L)),
                f"RANDOM-{comp}@k6": dict(masks=masks_from_sets(rsets, T, P, L))}
        for v in ("T", "DL"):
            rt, ck = load_router(V2 / f"router_{comp}_r{'2' if comp == 'LR32' else '0'}_{v}.pt", device)
            rt.configure(E, k_max=int(-(-1.5 * K // 1)), tau=ck["tau_by_k"]["6.0"])
            pols[f"GUIDED-{v}-{comp}@k6"] = dict(router=rt.masked, eligible=rt.eligible)
        for name, kw in pols.items():
            r = eval_batch(model, wraps, wins, P, device, **kw)
            s = boot_rel(r["nll"].numpy(), r["nll_full"].numpy(), n_boot=2000)
            rep["rows"][name] = {"reproduced": s["rel_dnll_pct"], "v02_reported": ref[name]["rel_dnll_pct"],
                                 "abs_diff_pp": abs(s["rel_dnll_pct"] - ref[name]["rel_dnll_pct"]),
                                 "skips": float(r["skips"].mean()), "v02_skips": ref[name]["skips_per_token"]}
            print(name, rep["rows"][name], flush=True)
    rep["max_abs_diff_pp"] = max(v["abs_diff_pp"] for v in rep["rows"].values())
    rep["gate_ok"] = rep["max_abs_diff_pp"] <= 0.05
    return rep


def part_seeds(device, model, tok, wraps, L, d, E):
    set_comps(wraps, comps_f32("LR32", L, d, device))
    data = concat([{k: v for k, v in torch.load(ROOT / "data" / f"v2_teacher_LR32_r{r}_train.pt", weights_only=False).items()
                    if torch.is_tensor(v)} for r in (0, 1, 2)])
    T = P + NW - 1
    rep = {"config": {"T": "N=16 mse", "DL": "N=8 pinball q=0.3", "rounds": [0, 1, 2], "k": K}, "splits": {}}
    for split in ("val", "final"):
        wins = windows(tok, split)
        o, rsets = static_sets(E, len(wins), K)
        base = {"STATIC-GREEDY": eval_batch(model, wraps, wins, P, device,
                                            masks=masks_from_sets([o["LR32"]["static_greedy_order"][:K]] * len(wins), T, P, L))}
        for s in (7, 8, 9):
            _, rs = static_sets(E, len(wins), K, seed=s)
            base[f"RANDOM_s{s}"] = eval_batch(model, wraps, wins, P, device, masks=masks_from_sets(rs, T, P, L))
        rep["splits"][split] = {"controls": {k: {**boot_rel(v["nll"].numpy(), v["nll_full"].numpy()), "skips": float(v["skips"].mean())}
                                             for k, v in base.items()}, "routers": {}}
        rep["_wins_" + split] = wins
        rep["_base_" + split] = base
    calib = windows(tok, "val")[:64]
    for v, lagged, N, loss, q in (("T", False, 16, "mse", 0.5), ("DL", True, 8, "pinball", 0.3)):
        for seed in (0, 1, 2):
            t0 = time.time()
            r = train_router(data, d=d, L=L, N=N, lagged=lagged, loss=loss, q=q, device=device, seed=seed, n_el=len(E))
            tau, _ = calibrate_tau(model, wraps, r, calib, P, device, float(K), E)
            for split in ("val", "final"):
                res = eval_batch(model, wraps, rep["_wins_" + split], P, device, router=r.masked, eligible=r.eligible)
                base = rep["_base_" + split]
                row = {**boot_rel(res["nll"].numpy(), res["nll_full"].numpy()), "skips": float(res["skips"].mean()), "tau": tau,
                       "vs_STATIC-GREEDY": paired_diff(res["nll"].numpy(), base["STATIC-GREEDY"]["nll"].numpy(), res["nll_full"].numpy())}
                best_rand = min((k for k in base if k.startswith("RANDOM")), key=lambda k: float(base[k]["nll"].sum()))
                row["vs_best_RANDOM"] = paired_diff(res["nll"].numpy(), base[best_rand]["nll"].numpy(), res["nll_full"].numpy())
                rep["splits"][split]["routers"][f"GUIDED-{v}_s{seed}"] = row
                print(split, v, seed, {k: row[k] for k in ("rel_dnll_pct", "ci95", "skips")}, "vs SG", row["vs_STATIC-GREEDY"],
                      f"({time.time() - t0:.0f}s)", flush=True)
    for k in [k for k in rep if k.startswith("_")]:
        del rep[k]
    return rep


def part_labels(device, model, tok, wraps, L, d, E):
    from vetor.v2.teacher2 import label_window, sample_positions
    set_comps(wraps, comps_f32("LR32", L, d, device))
    stored = torch.load(ROOT / "data" / "v2_teacher_LR32_r0_train.pt", weights_only=False)
    rng = np.random.default_rng(100)
    wins = make_windows(tok, load_split("train"), P, NW, 480, 21)[:16]
    parts = []
    for w in wins:
        pos = sample_positions(P, NW, 12, rng)
        parts.append(label_window(model, wraps, w, P, torch.zeros(w.shape[0] - 1, L, dtype=torch.bool), pos, E, device))
    new = concat(parts)
    n = new["d16"].shape[0]
    old = stored["d16"][:n]
    rel_err = ((new["d16"] - old).abs() / (old.abs() + 1e-6))
    return {"cells": n, "same_cells": bool(torch.equal(new["lid"], stored["lid"][:n])),
            "d16_spearman": float(np.corrcoef(np.argsort(np.argsort(new["d16"].numpy())), np.argsort(np.argsort(old.numpy())))[0, 1]),
            "d16_median_rel_err": float(rel_err.median()), "d16_p99_rel_err": float(rel_err.quantile(0.99)),
            "d16_max_abs_err": float((new["d16"] - old).abs().max())}


@torch.inference_mode()
def gen_policy(model, wraps, prompts, device, n, **kw):
    out = []
    for p in prompts:
        r = decode_real(model, wraps, p.to(device).view(1, -1), n, **kw)
        out.append((r["tokens"].cpu(), r["mlp_skipped"] / max(r["forwards"], 1)))
    return out


def part_regime(device, model, tok, wraps, L, d, E):
    set_comps(wraps, comps_f32("LR32", L, d, device))
    rt, ck = load_router(V2 / "router_LR32_r2_T.pt", device)
    tau_tf = ck["tau_by_k"]["6.0"]
    rt.configure(E, k_max=int(-(-1.5 * K // 1)), tau=tau_tf)
    val = windows(tok, "val")
    use = val[64:192]  # nao usadas na calibracao de tau
    prompts = [w[:P] for w in use]
    rep = {"n_prompts": len(use), "tau_tf": tau_tf}
    # (1) mesmo texto natural: decode real (tokens forcados) == lote; (2) geracao livre
    nat = eval_batch(model, wraps, use, P, device, router=rt.masked, eligible=rt.eligible)
    forced = [decode_real(model, wraps, w[:P].to(device).view(1, -1), NW, router=rt, forced=w[P:].to(device)) for w in use[:32]]
    rep["skips_natural_batch"] = float(nat["skips"].mean())
    rep["skips_natural_real_forced_32"] = float(np.mean([f["mlp_skipped"] / f["forwards"] for f in forced]))
    full_gen = gen_policy(model, wraps, prompts, device, NW)
    full_seqs = [torch.cat([p, g]) for p, (g, _) in zip(prompts, full_gen)]
    gen_tf = gen_policy(model, wraps, prompts, device, NW, router=rt)
    rep["skips_free_generation_real"] = float(np.mean([s for _, s in gen_tf]))
    seqs_tf = [torch.cat([p, g]) for p, (g, _) in zip(prompts, gen_tf)]
    on_own = eval_batch(model, wraps, seqs_tf, P, device, router=rt.masked, eligible=rt.eligible)
    rep["skips_batch_on_policy_own_generation"] = float(on_own["skips"].mean())
    on_full = eval_batch(model, wraps, full_seqs, P, device, router=rt.masked, eligible=rt.eligible)
    rep["skips_batch_on_FULL_generation"] = float(on_full["skips"].mean())

    # diagnostico: entropia do FULL e taxa por camada / posicao nos dois tipos de texto
    def profile(seqs):
        ent, bits = [], []
        for i in range(0, len(seqs), 32):
            x = torch.stack(seqs[i:i + 32]).to(device)[:, :-1]
            lpf = logprobs_D(model, wraps, x, P, None)
            ent.append((-(lpf.exp() * lpf).sum(-1)).mean(1).cpu())
            c = Ctrl(start=P, L=L, router=rt.masked, eligible=rt.eligible)
            logprobs_D(model, wraps, x, P, c)
            bits.append(c.bits[:, P:].float().cpu())
        b = torch.cat(bits)
        return {"full_entropy_nats": float(torch.cat(ent).mean()), "skips_by_layer": b.mean((0, 1)).tolist(),
                "skips_by_position_quartile": [float(q.sum(-1).mean()) for q in b.chunk(4, dim=1)],
                "skips_per_token_hist": np.bincount(b.sum(-1).long().flatten().numpy(), minlength=10).tolist()}
    rep["profile_natural"] = profile(use)
    rep["profile_FULL_generation"] = profile(full_seqs)

    # politica de orcamento: tau calibrado em texto gerado pelo FULL (64 janelas de calibracao, nao avaliadas)
    calib_gen = [torch.cat([w[:P], g]) for w, (g, _) in zip(val[:64], gen_policy(model, wraps, [w[:P] for w in val[:64]], device, NW))]
    tau_gen, _ = calibrate_tau(model, wraps, rt, calib_gen, P, device, float(K), E)
    rep["tau_gen"] = tau_gen
    rt.configure(E, k_max=int(-(-1.5 * K // 1)), tau=tau_gen)
    gen_g = gen_policy(model, wraps, prompts, device, NW, router=rt)
    rep["skips_free_generation_real_tau_gen"] = float(np.mean([s for _, s in gen_g]))
    nat_g = eval_batch(model, wraps, use, P, device, router=rt.masked, eligible=rt.eligible)
    rep["natural_text_at_tau_gen"] = {**boot_rel(nat_g["nll"].numpy(), nat_g["nll_full"].numpy()), "skips": float(nat_g["skips"].mean())}

    # qualidade na geracao livre real (k=6 efetivo em cada politica)
    o, rsets = static_sets(E, len(prompts), K)
    sg = set(o["LR32"]["static_greedy_order"][:K])
    pols = {"GUIDED-T(tau_gen)": gen_g, "GUIDED-T(tau_tf)": gen_tf,
            "STATIC-GREEDY": gen_policy(model, wraps, prompts, device, NW, skip=sg),
            "RANDOM": [gen_policy(model, wraps, [p], device, NW, skip=set(s))[0] for p, s in zip(prompts, rsets)]}
    rep["free_generation"] = {}
    for name, g in pols.items():
        seqs = [torch.cat([p, t]) for p, (t, _) in zip(prompts, g)]
        x = torch.stack(seqs).to(device)
        # KL(FULL || politica) ao longo da propria trajetoria da politica (= o que o decode real experimenta)
        if name.startswith("GUIDED"):
            rt.configure(E, k_max=int(-(-1.5 * K // 1)), tau=tau_gen if "gen" in name else tau_tf)
            kw = dict(router=rt.masked, eligible=rt.eligible)
        elif name == "STATIC-GREEDY":
            kw = dict(masks=masks_from_sets([sorted(sg)] * len(seqs), P + NW - 1, P, L))
        else:
            kw = dict(masks=masks_from_sets(rsets, P + NW - 1, P, L))
        r = eval_batch(model, wraps, seqs, P, device, **kw)
        agree = [float((t == f).float().mean()) for (t, _), (f, _) in zip(g, full_gen)]
        prefix = [int((t != f).nonzero()[0]) if (t != f).any() else NW for (t, _), (f, _) in zip(g, full_gen)]
        reps = [rep4(t.tolist()) for t, _ in g]
        rep["free_generation"][name] = {
            "skips_real": float(np.mean([s for _, s in g])), "kl_on_policy_mean": float(r["kl"].mean()),
            "agree_with_FULL": float(np.mean(agree)), "identical_prefix_tokens": float(np.mean(prefix)),
            "rep4": float(np.mean(reps)), "degenerate_frac": float(np.mean([x >= 0.5 for x in reps]))}
        print(name, rep["free_generation"][name], flush=True)
    rep["free_generation"]["FULL"] = {"rep4": float(np.mean([rep4(t.tolist()) for t, _ in full_gen])),
                                      "degenerate_frac": float(np.mean([rep4(t.tolist()) >= 0.5 for t, _ in full_gen]))}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["eval", "seeds", "labels", "regime"])
    a = ap.parse_args()
    ctx = setup()
    t0 = time.time()
    rep = globals()[f"part_{a.part}"](*ctx)
    rep.update({"env": env_info(ctx[0]), "seconds": time.time() - t0})
    write_json(R / f"{a.part}.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, (dict, list))}, indent=1))


if __name__ == "__main__":
    main()
