"""Loop autoregressivo com KV cache e roteamento de MLP por politica.

Convencao: o prefill (prompt inteiro, todos os MLPs) produz os logits do 1o token novo.
Cada forward de decode `step` (0-based) recebe 1 token e produz os logits do token seguinte.
Para `n_new` tokens produzidos ha `n_new - 1` forwards de decode; os logits afetados
pelo roteamento sao os de indices 1..n_new-1.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model_adapter import Throttle
from .policies import Policy
from .wrappers import RoutedMLP, apply_mask, reset, set_capture, stack_captured


@torch.inference_mode()
def run_decode(
    model,
    wraps: list[RoutedMLP],
    prompt_ids: torch.Tensor,
    n_new: int,
    policy: Policy,
    *,
    forced: torch.Tensor | None = None,
    collect_logp: bool = False,
    ctx: dict | None = None,
) -> dict:
    """Greedy (forced=None) ou teacher forcing (forced=[n_new] tokens alvo).

    Nao ha `.item()` nem sincronizacao no loop, exceto as da propria politica.
    """
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("prompt_ids deve ser [1, T]")
    reset(wraps)
    set_capture(wraps, True)
    out = model(input_ids=prompt_ids, use_cache=True)
    cache = out.cache
    Z = stack_captured(wraps)
    active = policy.start(Z, ctx or {})
    set_capture(wraps, policy.capture_decode)
    apply_mask(wraps, active)
    masks = [active]

    logits = out.logits[:, -1, :]
    toks, logps = [], []
    throttle = Throttle(str(prompt_ids.device.type))
    for i in range(n_new):
        if collect_logp:
            logps.append(F.log_softmax(logits.float(), dim=-1)[0])
        nxt = forced[i].view(1, 1) if forced is not None else logits.argmax(-1, keepdim=True)
        toks.append(nxt.view(-1))
        if i == n_new - 1:
            break
        new = policy.update(i, wraps)
        if new is not None:
            apply_mask(wraps, new)
            masks.append(new)
        out = model(input_ids=nxt, cache=cache, use_cache=True)
        cache = out.cache
        logits = out.logits[:, -1, :]
        throttle.step()

    reset(wraps)
    return {
        "tokens": torch.cat(toks),
        "logp": torch.stack(logps) if collect_logp else None,
        "masks": masks,
        "cache_len": cache.get_seq_length(),
    }
