"""Politicas de roteamento: FULL, RANDOM, STATIC, ORACLE-DIAG, GUIDED-S, GUIDED-D.

Interface usada pelo loop de decode:
    start(Z, ctx)          -> mascara `active` [L] (bool) apos o prefill. Z = z[l, ultimo token]
    update(step, wraps)    -> nova mascara ou None, ANTES do forward de decode `step` (0-based).
                              So pode usar capturas de forwards anteriores (causal).
    capture_decode         -> se o loop deve capturar z[l,t] durante o decode.

Toda politica respeita: camadas protegidas, maximo de k omissoes, fallback FULL em erro.
"""
from __future__ import annotations

import numpy as np
import torch

from .guide import VectorGuide, eligible_layers, topk_mask, validate_mask
from .wrappers import RoutedMLP, stack_captured


class Policy:
    name = "BASE"
    capture_decode = False

    def __init__(self, n_layers: int, k: int, protected_first: int = 2, protected_last: int = 2):
        self.L = n_layers
        self.k = k
        self.pf = protected_first
        self.pl = protected_last
        self.eligible = eligible_layers(n_layers, protected_first, protected_last)
        self.n_updates = 0
        self.n_fallbacks = 0

    def full(self) -> list[bool]:
        return [True] * self.L

    def safe(self, active: list[bool]) -> list[bool]:
        if not validate_mask(active, self.k, self.pf, self.pl):
            self.n_fallbacks += 1
            return self.full()
        return active

    def start(self, Z: torch.Tensor, ctx: dict) -> list[bool]:
        return self.full()

    def update(self, step: int, wraps: list[RoutedMLP]) -> list[bool] | None:
        return None


class FullPolicy(Policy):
    name = "FULL"

    def __init__(self, n_layers: int, **kw):
        super().__init__(n_layers, 0, **kw)


class StaticPolicy(Policy):
    """Omite sempre as k camadas de menor impacto medio no TREINO."""
    name = "STATIC"

    def __init__(self, n_layers: int, k: int, ranking: list[int], **kw):
        super().__init__(n_layers, k, **kw)
        chosen = [l for l in ranking if l in self.eligible][:k]
        self.mask = [l not in chosen for l in range(n_layers)]

    def start(self, Z, ctx):
        return self.safe(list(self.mask))


class RandomPolicy(Policy):
    """k camadas elegiveis sorteadas por sequencia (semente por amostra)."""
    name = "RANDOM"

    def __init__(self, n_layers: int, k: int, seed: int = 0, **kw):
        super().__init__(n_layers, k, **kw)
        self.seed = seed

    def start(self, Z, ctx):
        rng = np.random.default_rng([self.seed, int(ctx.get("sample_idx", 0))])
        chosen = set(rng.choice(self.eligible, size=self.k, replace=False).tolist()) if self.k else set()
        return self.safe([l not in chosen for l in range(self.L)])


class OraclePolicy(Policy):
    """DIAGNOSTICO, NAO IMPLEMENTAVEL: usa o KL de ablacao isolada do proprio prompt."""
    name = "ORACLE-DIAG"

    def start(self, Z, ctx):
        kl = torch.as_tensor(ctx["oracle_kl"], dtype=torch.float32)
        active = topk_mask(-kl, self.k, protected_first=self.pf, protected_last=self.pl)
        return self.safe(active.tolist())


class GuidedPolicy(Policy):
    """GUIDED-S (K=None: decide uma vez no prefill) ou GUIDED-D (reavalia a cada K tokens).

    Em GUIDED-D, a decisao do forward `step` usa z capturado no forward `step-1`
    (atraso de 1 a K tokens, nunca informacao futura). Uma unica sincronizacao
    (`.tolist()`) por atualizacao de mascara.
    """

    def __init__(self, n_layers: int, k: int, guide: VectorGuide, K: int | None = None,
                 min_score: float | None = None, pool: list[int] | None = None, **kw):
        super().__init__(n_layers, k, **kw)
        self.guide = guide
        # pool: camadas candidatas (ex.: k+m mais seguras no TREINO). Fora do pool -> nunca omite.
        self.pool_bias = None
        if pool is not None:
            b = torch.full((n_layers,), float("-inf"))
            b[[l for l in pool if l in self.eligible]] = 0.0
            self.pool_bias = b
        self.K = K
        self.min_score = min_score
        self.capture_decode = K is not None
        self.name = "GUIDED-S" if K is None else f"GUIDED-D{K}"

    def _decide(self, Z: torch.Tensor) -> list[bool]:
        try:
            scores = self.guide.score_layers(Z)
            if self.pool_bias is not None:
                if self.pool_bias.device != scores.device:
                    self.pool_bias = self.pool_bias.to(scores.device)
                scores = scores + self.pool_bias
            active = topk_mask(scores, self.k, protected_first=self.pf, protected_last=self.pl,
                               min_score=self.min_score).tolist()
            self.n_updates += 1
            return self.safe(active)
        except Exception:
            self.n_fallbacks += 1
            return self.full()

    def start(self, Z, ctx):
        if self.k == 0:
            return self.full()
        return self._decide(Z)

    def update(self, step, wraps):
        if self.K is None or self.k == 0 or step == 0 or step % self.K:
            return None
        return self._decide(stack_captured(wraps))


def build_policies(n_layers: int, ks: list[int], *, kinds: list[str], static_rank: list[int] | None = None,
                   guide: VectorGuide | None = None, Ks: tuple[int, ...] = (1, 4, 8, 16),
                   seed: int = 0, protected_first: int = 2, protected_last: int = 2,
                   pool_extra: int | None = None) -> list[Policy]:
    """kinds: subconjunto de RANDOM, STATIC, ORACLE, GUIDED-S, GUIDED-D. k=0 vira so FULL."""
    kw = {"protected_first": protected_first, "protected_last": protected_last}
    out: list[Policy] = [FullPolicy(n_layers, **kw)]
    for k in ks:
        if k == 0:
            continue
        if "RANDOM" in kinds:
            out.append(RandomPolicy(n_layers, k, seed=seed, **kw))
        if "STATIC" in kinds:
            out.append(StaticPolicy(n_layers, k, static_rank, **kw))
        if "ORACLE" in kinds:
            out.append(OraclePolicy(n_layers, k, **kw))
        pool = None
        if pool_extra is not None:
            pool = [l for l in static_rank if protected_first <= l < n_layers - protected_last][: k + pool_extra]
        if "GUIDED-S" in kinds:
            out.append(GuidedPolicy(n_layers, k, guide, pool=pool, **kw))
        if "GUIDED-D" in kinds:
            out.extend(GuidedPolicy(n_layers, k, guide, K=K, pool=pool, **kw) for K in Ks)
    return out
