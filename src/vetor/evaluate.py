"""Qualidade: teacher forcing autoregressivo com cache + geracao greedy livre.

Para cada janela (prompt, continuacao verdadeira) e politica:
- teacher forcing passo a passo com KV e a politica ativa no decode: NLL do token verdadeiro,
  KL(p_FULL || p_pol) no MESMO prefixo de referencia, concordancia top-1;
- geracao greedy livre: taxa de tokens iguais ao FULL e posicao da 1a divergencia.
Metricas usam os passos 1..n-1 (o passo 0 vem do prefill, sempre completo).
Usa o caminho de skip REAL (mesmo codigo do benchmark), nao um atalho mascarado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .collect import ablation_teacher
from .decode import run_decode
from .policies import FullPolicy, Policy
from .wrappers import counters, reset_counters


@torch.inference_mode()
def _tf_stats(lp_full: torch.Tensor, lp: torch.Tensor, tgt: torch.Tensor) -> dict:
    s = slice(1, None)
    lpf, lpp, t = lp_full[s], lp[s], tgt[s]
    ar = torch.arange(t.shape[0], device=t.device)
    nll = -lpp[ar, t]
    kl = (lpf.exp() * (lpf - lpp)).sum(-1).clamp_min(0)
    top1 = (lpp.argmax(-1) == lpf.argmax(-1)).float()
    return {"nll": nll.cpu().numpy(), "kl": kl.cpu().numpy(), "top1": top1.cpu().numpy()}


def evaluate(model, wraps, windows: list[dict], policies: list[Policy], device: str,
             greedy: bool = True, oracle: bool = True, progress=None) -> tuple[pd.DataFrame, dict]:
    """Retorna (linhas por amostra x politica, kl por token por politica)."""
    L = len(wraps)
    full = FullPolicy(L)
    rows, kl_tokens = [], {}
    it = enumerate(windows)
    if progress:
        it = progress(list(it))
    for si, w in it:
        prompt = torch.tensor([w["prompt"]], device=device)
        cont = torch.tensor(w["continuation"], device=device)
        n = cont.shape[0]
        ctx = {"sample_idx": si}
        if oracle:
            ctx["oracle_kl"] = ablation_teacher(model, wraps, w["prompt"], None, device)["kl"].cpu()
        ref = run_decode(model, wraps, prompt, n, full, forced=cont, collect_logp=True)
        lp_full = ref["logp"]
        nll_full = _tf_stats(lp_full, lp_full, cont)["nll"]
        g_full = run_decode(model, wraps, prompt, n, full)["tokens"].tolist() if greedy else None
        for pol in policies:
            reset_counters(wraps)
            r = run_decode(model, wraps, prompt, n, pol, forced=cont, collect_logp=True, ctx=ctx)
            inner, skipped = counters(wraps)
            st = _tf_stats(lp_full, r["logp"], cont)
            row = {"sample": si, "doc_id": w["doc_id"], "policy": pol.name, "k": pol.k,
                   "nll_full": float(nll_full.mean()), "nll": float(st["nll"].mean()),
                   "kl_mean": float(st["kl"].mean()), "top1_agree": float(st["top1"].mean()),
                   "mlp_skipped": skipped, "mlp_decode_total": (n - 1) * L,
                   "skip_rate_all": skipped / ((n - 1) * L),
                   "mask_updates": len(r["masks"])}
            if greedy:
                g = run_decode(model, wraps, prompt, n, pol, ctx=ctx)["tokens"].tolist()
                eq = [a == b for a, b in zip(g, g_full)]
                row["greedy_match"] = float(np.mean(eq))
                row["greedy_first_div"] = eq.index(False) if False in eq else n
            rows.append(row)
            kl_tokens.setdefault((pol.name, pol.k), []).append(st["kl"])
    return pd.DataFrame(rows), {k: np.concatenate(v) for k, v in kl_tokens.items()}


def summarize(df: pd.DataFrame, kl_tokens: dict, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for (name, k), g in df.groupby(["policy", "k"], sort=False):
        rel = g.nll.values / g.nll_full.values - 1
        d = g.nll.values - g.nll_full.values
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        # IC bootstrap do aumento relativo de NLL agregado (razao das medias)
        boot = g.nll.values[idx].mean(1) / g.nll_full.values[idx].mean(1) - 1
        klt = kl_tokens[(name, k)]
        r = {"policy": name, "k": k, "n_samples": len(g),
             "skip_rate_all": g.skip_rate_all.mean(),
             "nll_full": g.nll_full.mean(), "nll": g.nll.mean(),
             "delta_nll": d.mean(),
             "rel_delta_nll_pct": 100 * (g.nll.mean() / g.nll_full.mean() - 1),
             "rel_delta_nll_ci95_lo": 100 * np.percentile(boot, 2.5),
             "rel_delta_nll_ci95_hi": 100 * np.percentile(boot, 97.5),
             "rel_delta_nll_median_sample_pct": 100 * np.median(rel),
             "ppl_full": float(np.exp(g.nll_full.mean())), "ppl": float(np.exp(g.nll.mean())),
             "kl_mean": float(klt.mean()), "kl_p50": float(np.percentile(klt, 50)),
             "kl_p90": float(np.percentile(klt, 90)), "kl_p99": float(np.percentile(klt, 99)),
             "top1_agree": g.top1_agree.mean()}
        if "greedy_match" in g:
            r["greedy_match"] = g.greedy_match.mean()
            r["greedy_first_div_median"] = g.greedy_first_div.median()
        out.append(r)
    return pd.DataFrame(out)
