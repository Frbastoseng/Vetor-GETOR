"""Treino EXCLUSIVO do vetor-guia sobre dados do professor (LLM congelado, fora do grafo).

Entradas: z[l,t] e indice da camada (somente o que existe antes do MLP).
Alvos:
  classify: safe_l = 1[KL<=kl_max & dNLL<=dnll_max]  (BCEWithLogits + pos_weight)
  regress : -log(KL + eps)  (score alto = omissao mais segura; MSE)
  quantile: quantil `q` de -log(KL + eps) (pinball) — estimativa AVESSA A RISCO da seguranca
Selecao de modelo/alvo SO na validacao. O teste nunca entra aqui.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .guide import VectorGuide, eligible_layers

EPS = 1e-6


def load_teacher(path) -> dict:
    return torch.load(path, weights_only=False)


def flatten(t: dict, layers: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = t["z"].shape[0]
    lid = torch.tensor(layers)
    Z = t["z"][:, lid, :].float().reshape(-1, t["z"].shape[-1])
    ids = lid.repeat(N)
    return Z, ids, t["kl"][:, lid].reshape(-1), t["safe"][:, lid].reshape(-1).float()


def static_ranking(t: dict, stat: str = "mean") -> list[int]:
    """Camadas ordenadas da MENOS para a MAIS sensivel (KL medio no conjunto dado)."""
    v = t["kl"].mean(0) if stat == "mean" else t["kl"].median(0).values
    return torch.argsort(v).tolist()


def train_guide(train: dict, val: dict, *, n_layers: int, hidden: int, pf: int, pl: int,
                target: str, dim: int = 16, epochs: int = 60, lr: float = 3e-3,
                weight_decay: float = 1e-2, batch_size: int = 2048, seed: int = 0,
                q: float = 0.2, device: str = "cpu", log=print) -> tuple[VectorGuide, dict]:
    torch.manual_seed(seed)
    layers = eligible_layers(n_layers, pf, pl)
    Z, ids, kl, safe = flatten(train, layers)
    Zv, idsv, klv, safev = flatten(val, layers)
    y = safe if target == "classify" else -torch.log(kl + EPS)
    yv = safev if target == "classify" else -torch.log(klv + EPS)
    if target in ("regress", "quantile"):  # normaliza o alvo (estatisticas do treino)
        mu, sd = y.mean(), y.std()
        y, yv = (y - mu) / sd, (yv - mu) / sd
    pos_weight = ((1 - safe).sum() / safe.sum().clamp_min(1)).to(device)
    g = VectorGuide(hidden, n_layers, dim, seed=seed).to(device)
    opt = torch.optim.AdamW(g.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Z, ids, y = Z.to(device), ids.to(device), y.to(device)
    Zv, idsv, yv = Zv.to(device), idsv.to(device), yv.to(device)

    def loss_fn(s, t):
        if target == "classify":
            return F.binary_cross_entropy_with_logits(s, t, pos_weight=pos_weight)
        if target == "quantile":
            e = t - s
            return torch.maximum(q * e, (q - 1) * e).mean()
        return F.mse_loss(s, t)

    best, best_state, hist = float("inf"), None, []
    for ep in range(epochs):
        g.train()
        perm = torch.randperm(Z.shape[0], device=device)
        tot = 0.0
        for i in range(0, Z.shape[0], batch_size):
            b = perm[i:i + batch_size]
            loss = loss_fn(g(Z[b], ids[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        g.eval()
        with torch.no_grad():
            vl = float(loss_fn(g(Zv, idsv), yv))
        hist.append({"epoch": ep, "train_loss": tot / Z.shape[0], "val_loss": vl})
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in g.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  [{target}] ep {ep:3d} train {tot / Z.shape[0]:.4f} val {vl:.4f}")
    g.load_state_dict(best_state)
    g.eval()
    n_params = sum(p.numel() for p in g.parameters())
    return g.cpu(), {"history": hist, "best_val_loss": best, "n_params": n_params,
                     "target": target, "q": q if target == "quantile" else None,
                    "pos_weight": float(pos_weight)}


@torch.no_grad()
def offline_metrics(g: VectorGuide, t: dict, *, n_layers: int, pf: int, pl: int,
                    ks: list[int], static_rank: list[int], seed: int = 0, pool_extra: int | None = None) -> dict:
    """Metricas de previsao (sem rodar o LLM): AUC do rotulo safe, Spearman intra-amostra e
    'arrependimento' da selecao top-k sob o proxy aditivo de KLs isolados."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    layers = eligible_layers(n_layers, pf, pl)
    N = t["z"].shape[0]
    lid = torch.tensor(layers)
    Z = t["z"][:, lid, :].float()
    S = g(Z.reshape(-1, Z.shape[-1]), lid.repeat(N)).reshape(N, len(layers))
    KL = t["kl"][:, lid]
    safe = t["safe"][:, lid]
    out = {"auc_safe": float(roc_auc_score(safe.reshape(-1).numpy(), S.reshape(-1).numpy()))}
    rho = [spearmanr(S[i].numpy(), -KL[i].numpy()).statistic for i in range(N)]
    out["spearman_within_sample_mean"] = float(np.nanmean(rho))
    # AUC "intra-camada": o guia distingue contextos seguros dentro da MESMA camada?
    aucs = []
    for j in range(len(layers)):
        yj = safe[:, j].numpy()
        if 0 < yj.mean() < 1:
            aucs.append(roc_auc_score(yj, S[:, j].numpy()))
    out["auc_safe_within_layer_mean"] = float(np.mean(aucs))
    rng = np.random.default_rng(seed)
    st_pos = [layers.index(l) for l in static_rank if l in layers]
    for k in ks:
        if k == 0:
            continue
        Sk = S
        if pool_extra is not None:  # guia escolhe so entre as k+m camadas mais seguras do treino
            Sk = torch.full_like(S, float("-inf"))
            pool = st_pos[: k + pool_extra]
            Sk[:, pool] = S[:, pool]
        gk = KL.gather(1, Sk.topk(k, dim=1).indices).sum(1)
        ok = KL.topk(k, dim=1, largest=False).values.sum(1)
        sk = KL[:, st_pos[:k]].sum(1)
        rk = torch.stack([KL[i, rng.choice(len(layers), k, replace=False)].sum() for i in range(N)])
        out[f"k{k}"] = {"guided_sumKL": float(gk.mean()), "static_sumKL": float(sk.mean()),
                        "random_sumKL": float(rk.mean()), "oracle_sumKL": float(ok.mean()),
                        "guided_beats_static_frac": float((gk < sk).float().mean())}
    return out
