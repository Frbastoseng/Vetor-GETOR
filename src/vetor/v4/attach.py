"""Acopla uma politica (FULL / STATIC / GUIDED-T) ao modelo instrumentado para lm-eval e geracao livre.

Modo mascarado (qualidade; equivalente ao decode real, testado). Regras:
- forward com T>1 em regime 'loglik' (tarefas de verossimilhanca): decisoes em TODAS as posicoes, tau_tf;
- forward com T>1 em regime 'gen' (prefill do prompt): sem omissoes;
- forward com T==1 (passo de decode da geracao): decisoes com tau_gen.
GUIDED-DL nao e suportado aqui (precisa de z do token anterior entre passos); avaliado so em NLL.
"""
from __future__ import annotations

import torch

from ..v2.engine import Ctrl, set_ctrl


class Policy:
    def __init__(self, model, wraps, L, *, kind="FULL", router=None, tau_tf=None, tau_gen=None, k_max=None,
                 static_set=None):
        self.model, self.wraps, self.L, self.kind = model, wraps, L, kind
        self.router, self.tau_tf, self.tau_gen, self.k_max = router, tau_tf, tau_gen, k_max
        self.static_set = set(static_set or [])
        self.regime = "loglik"
        self.skips = 0.0
        self.tokens = 0
        self.ctrl = None
        self.hook = None

    def _static(self, z, l, c):
        return torch.full(z.shape[:2], l in self.static_set, dtype=torch.bool, device=z.device)

    def _post(self, mod, args, kwargs, out):
        c = self.ctrl
        if c is not None and c.bits is not None:
            b = c.bits[:, c.start:]
            self.skips += float(b.sum())
            self.tokens += int(b.shape[0] * b.shape[1])

    def _pre(self, mod, args, kwargs):
        x = kwargs.get("input_ids")
        if x is None:
            x = kwargs.get("inputs_embeds")
        T = x.shape[1]
        if T > 1 and self.regime == "gen":
            start, tau = T, self.tau_gen
        elif T == 1:
            start, tau = 0, self.tau_gen
        else:
            start, tau = 0, self.tau_tf
        if self.kind == "GUIDED":
            self.router.configure(self.router.eligible.nonzero().flatten().tolist(), k_max=self.k_max, tau=tau)
            self.ctrl = Ctrl(start=start, L=self.L, router=self.router.masked, eligible=self.router.eligible)
        elif self.kind == "STATIC":
            self.ctrl = Ctrl(start=start, L=self.L, router=self._static)
        else:
            self.ctrl = None
        set_ctrl(self.wraps, self.ctrl)

    def __enter__(self):
        if self.kind != "FULL":
            self.hook = (self.model.model.register_forward_pre_hook(self._pre, with_kwargs=True),
                         self.model.model.register_forward_hook(self._post, with_kwargs=True))
        return self

    def __exit__(self, *a):
        if self.hook:
            for h in self.hook:
                h.remove()
        set_ctrl(self.wraps, None)
        self.ctrl = None

    def rate(self) -> float:
        return self.skips / max(self.tokens, 1)
