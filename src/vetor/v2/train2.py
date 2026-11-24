"""Treino do roteador v0.2 (so o roteador; LLM e compensador congelados) e calibracao de tau."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .router2 import TraversalRouter
from .search import eval_batch, summarize

EPS = 1e-4


def train_router(data: dict, *, d: int, L: int, N: int, lagged: bool, loss: str = "mse", q: float = 0.5,
                 epochs: int = 40, lr: float = 3e-3, wd: float = 1e-2, bs: int = 4096, seed: int = 0,
                 device: str = "cpu", n_el: int = 26) -> TraversalRouter:
    """n_el: nº de camadas elegiveis (normaliza a feature n). 26 = SmolLM2-135M (v0.2); passar len(E) em outros modelos."""
    torch.manual_seed(seed)
    r = TraversalRouter(d, L, lagged=lagged, seed=seed)
    z = (data["z_lag"] if lagged else data["z"]).float()
    y = torch.log(data[f"d{N}"].clamp_min(0) + EPS)
    r.y_mu, r.y_sd = float(y.mean()), float(y.std())
    y = (y - r.y_mu) / r.y_sd
    lid, bits, n = data["lid"], data["bits"], data["n"]
    r = r.to(device)
    z, y, lid, bits, n = (t.to(device) for t in (z, y, lid, bits, n))
    opt = torch.optim.AdamW(r.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    r.n_el = n_el
    for _ in range(epochs):
        perm = torch.randperm(z.shape[0], device=device)
        for i in range(0, z.shape[0], bs):
            b = perm[i:i + bs]
            s = r(z[b], lid[b], bits[b], n[b])
            if loss == "mse":
                ls = F.mse_loss(s, y[b])
            else:  # pinball: estimativa avessa a risco (quantil q do risco)
                e = y[b] - s
                ls = torch.maximum(q * e, (q - 1) * e).mean()
            opt.zero_grad()
            ls.backward()
            opt.step()
        sched.step()
    return r.eval()


@torch.inference_mode()
def calibrate_tau(model, wraps, router: TraversalRouter, windows, P: int, device, k_target: float,
                  eligible: list[int], iters: int = 14) -> tuple[float, float]:
    """Bissecao em tau para que a media de omissoes por token de decode ~= k_target (dados de VALIDACAO)."""
    router.configure(eligible, k_max=int(-(-1.5 * k_target // 1)))
    lo, hi = -20.0, 20.0
    el = router.eligible
    rate = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        router.tau.fill_(mid)
        r = eval_batch(model, wraps, windows, P, device, router=router.masked, eligible=el, chunk=64)
        rate = float(r["skips"].mean())
        if rate < k_target:
            lo = mid
        else:
            hi = mid
    router.tau.fill_((lo + hi) / 2)
    return (lo + hi) / 2, rate


def eval_router(model, wraps, router, windows, P, device) -> dict:
    r = eval_batch(model, wraps, windows, P, device, router=router.masked, eligible=router.eligible)
    return summarize(r)
