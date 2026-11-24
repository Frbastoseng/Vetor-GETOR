"""v0.4 / Fase 0: auditoria automatica da base experimental (v0.2) nos dois modelos.

Verifica: (1) modelo instrumentado sem omissoes == original; (2) MLP omitido NAO e executado (contador
no MLP original); (3) modo mascarado em lote == decode real com KV (estatico e roteador); (4) KV cache:
decode com cache == forward completo; (5) alinhamento de tokens da NLL; (6) FLOPs contados pelo PyTorch
vs formulas usadas na v0.2. Saida: results/v0.4/audit_{modelo}.json
  python scripts/v4_00_audit.py --model 135M
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from vetor.model_adapter import env_info, pick_device, seed_all, write_json
from vetor.v2.decode2 import decode_real
from vetor.v2.engine import Compensator, Ctrl, patch_v2, unpatch_v2, zero_comps
from vetor.v2.fasteval import logprobs_D, seq_mask, window_io
from vetor.v2.router2 import TraversalRouter
from vetor.v4.common import OUT, eligible, flops_of, load

TEXT = ("The history of the city begins in the early medieval period, when a small settlement grew around the "
        "river crossing. Over the following centuries the town expanded, its markets attracted merchants from "
        "distant regions, and a cathedral was built on the hill overlooking the harbour. In the nineteenth century "
        "the arrival of the railway transformed the local economy, and factories replaced many of the old workshops. "
        "Today the city is known for its university, its museums and the annual music festival held every summer.")


def rand_comps(kind, L, d, device, rank=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(L):
        c = Compensator(kind, d, rank=rank if kind == "lr" else 0)
        c.ybar.copy_(torch.randn(d, generator=g) * 0.05)
        if kind == "lr":
            c.zbar.copy_(torch.randn(d, generator=g) * 0.1)
            c.V.copy_(torch.randn(d, rank, generator=g) / d ** 0.5)
            c.U.copy_(torch.randn(rank, d, generator=g) * 0.05)
        out.append(c.to(device))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["135M", "1.7B"])
    args = ap.parse_args()
    seed_all(1234)
    device = pick_device("auto")
    model, tok = load(args.model, device)
    L, d, dff = model.config.num_hidden_layers, model.config.hidden_size, model.config.intermediate_size
    E = eligible(L)
    ids = torch.tensor(tok(TEXT).input_ids[:96], device=device)
    P = 48
    rep = {"model": args.model, "env": env_info(device), "L": L, "d": d, "d_ff": dff, "eligible": E, "checks": {}}
    ck = rep["checks"]

    with torch.inference_mode():
        lg_orig = model(input_ids=ids.view(1, -1), use_cache=False).logits.float()
    wraps = patch_v2(model, zero_comps(L, d, device))

    # (1) instrumentado sem omissoes == original (caminho real e caminho mascarado sem decisoes)
    with torch.inference_mode():
        lg_p = model(input_ids=ids.view(1, -1), use_cache=False).logits.float()
    lp_m = logprobs_D(model, wraps, ids.view(1, -1), 0, Ctrl(start=0, L=L, static_mask=torch.zeros(1, ids.numel(), L, dtype=torch.bool, device=device)))
    ck["1_instrumented_equals_original"] = {
        "max_abs_logit_diff_real_path": float((lg_p - lg_orig).abs().max()),
        "max_abs_logprob_diff_masked_path": float((lp_m[0] - F.log_softmax(lg_orig[0], -1)).abs().max())}

    # (2) MLP omitido nao e executado: contador no MLP ORIGINAL (forward hook em w.inner)
    calls = [0] * L
    hooks = [w.inner.register_forward_hook(lambda m, i, o, l=w.layer_idx: calls.__setitem__(l, calls[l] + 1)) for w in wraps]
    skip = set(E[1::4])
    n = ids.numel() - P
    for w, c in zip(wraps, rand_comps("lr", L, d, device)):
        w.comp = c
    real = decode_real(model, wraps, ids[:P].view(1, -1), n, skip=skip, forced=ids[P:], collect_logp=True)
    for h in hooks:
        h.remove()
    ck["2_skipped_mlp_not_executed"] = {
        "skip_layers": sorted(skip), "decode_forwards": n - 1,
        "inner_calls_skipped_layers": {l: calls[l] for l in sorted(skip)},
        "inner_calls_other_layers_min_max": [min(calls[l] for l in range(L) if l not in skip),
                                             max(calls[l] for l in range(L) if l not in skip)],
        "expected": "camadas omitidas: 1 chamada (so o prefill); demais: 1 + (n-1)",
        "ok": all(calls[l] == 1 for l in skip) and all(calls[l] == n for l in range(L) if l not in skip)}

    # (3) mascarado == real (estatico, LR) e (roteador GUIDED-T e GUIDED-DL)
    x, tgt = window_io(ids, P)
    m = seq_mask(sorted(skip), x.shape[1], P, L, device).unsqueeze(0)
    lp = logprobs_D(model, wraps, x, P, Ctrl(start=P, L=L, static_mask=m))[0]
    ck["3a_masked_equals_real_static"] = {"max_abs_logprob_diff": float((real["logp"][1:] - lp).abs().max()),
                                          "mlp_skipped_real": real["mlp_skipped"], "expected": len(skip) * (n - 1)}
    for lagged in (False, True):
        r = TraversalRouter(d, L, lagged=lagged, seed=1).to(device)
        with torch.no_grad():
            r.b.weight.normal_(0, 1)
            r.w.weight.normal_(0, 1)
        r.n_el = len(E)
        r.configure(E, k_max=6, tau=0.0)
        rr = decode_real(model, wraps, ids[:P].view(1, -1), n, router=r, forced=ids[P:], collect_logp=True)
        ctrl = Ctrl(start=P, L=L, router=r.masked, eligible=r.eligible)
        lpr = logprobs_D(model, wraps, x, P, ctrl)[0]
        ck[f"3b_masked_equals_real_router_{'DL' if lagged else 'T'}"] = {
            "max_abs_logprob_diff": float((rr["logp"][1:] - lpr).abs().max()),
            "skips_real": rr["mlp_skipped"], "skips_masked": int(ctrl.bits[0, P:].sum()),
            "decisions_identical": rr["mlp_skipped"] == int(ctrl.bits[0, P:].sum()),
            "protected_never_skipped": not bool(ctrl.bits[0, :, [0, 1, L - 2, L - 1]].any()),
            "prefill_never_skipped": not bool(ctrl.bits[0, :P].any())}

    # (4) KV cache: decode FULL com cache == forward completo sem cache
    full = decode_real(model, wraps, ids[:P].view(1, -1), n, forced=ids[P:], collect_logp=True)
    lpf = logprobs_D(model, wraps, x, P, None)[0]
    ck["4_kv_cache_full_decode_equals_forward"] = {"max_abs_logprob_diff": float((full["logp"][1:] - lpf).abs().max())}

    # (5) alinhamento: NLL do modo em lote nas posicoes D == perda do HF (labels deslocados) nas mesmas posicoes
    with torch.inference_mode():
        lg = model(input_ids=x, use_cache=False).logits[0].float()
    hf = F.cross_entropy(lg[P:], tgt, reduction="sum")
    ours = -lpf.gather(-1, tgt.view(-1, 1)).sum()
    ck["5_token_alignment"] = {"nll_ours": float(ours), "nll_hf_shifted": float(hf), "abs_diff": float((ours - hf).abs()),
                               "n_positions": int(tgt.numel())}

    # (6) FLOPs por token contados pelo PyTorch vs formulas da v0.2
    unpatch_v2(model)
    mlp = model.model.layers[E[0]].mlp.cpu()
    z = torch.randn(1, 1, d)
    f_mlp = flops_of(mlp, z)
    mlp.to(device)
    comp = rand_comps("lr", 1, d, "cpu", rank=32)[0]
    f_lr32 = flops_of(comp, z)
    r = TraversalRouter(d, L)
    f_rt = flops_of(lambda: r(z[0], torch.tensor([3]), torch.zeros(1, L, dtype=torch.bool), torch.tensor([0])))
    mc = model.cpu()
    with torch.inference_mode():
        o = mc(input_ids=ids[:P].cpu().view(1, -1), use_cache=True)
    f_tok = flops_of(lambda: mc(input_ids=ids[P:P + 1].cpu().view(1, 1), cache=o.cache, use_cache=True))
    f_head = 2 * d * model.config.vocab_size
    ck["6_flops"] = {
        "mlp_counted": f_mlp, "mlp_formula_6_d_dff": 6 * d * dff,
        "lr32_counted": f_lr32, "lr32_formula_v02": 2 * d * 32 + 2 * d,
        "lr32_note": "a v0.2 contou so 1 das 2 multiplicacoes (V e U): o custo real e ~4*d*r",
        "router_decision_counted_matmul_only": f_rt,
        "decode_token_total_counted_ctx48": f_tok, "lm_head": f_head,
        "mlp_share_of_token_flops": 6 * d * dff * L / f_tok}
    model.to(device)
    write_json(OUT / f"audit_{args.model}.json", rep)
    for k, v in ck.items():
        print(k, v)


if __name__ == "__main__":
    main()
