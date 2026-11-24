"""v0.4: a politica acoplada (modo mascarado, usada em lm-eval e geracao livre) reproduz o decode real.

Geracao greedy via model.generate + Policy == decode_real (omissao real, MLP nao executado), com omissao
estatica e com GUIDED-T: mesmos tokens e mesmo numero de omissoes.
"""
import pytest
import torch

from vetor.v2.decode2 import decode_real
from vetor.v2.engine import Compensator, patch_v2, unpatch_v2, zero_comps
from vetor.v2.router2 import TraversalRouter
from vetor.v4.attach import Policy

pytestmark = pytest.mark.model


def _comps(L, d, device):
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(L):
        c = Compensator("mean", d)
        c.ybar.copy_(torch.randn(d, generator=g) * 0.05)
        out.append(c.to(device))
    return out


@pytest.fixture
def v2(lm, device):
    model, _ = lm
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    for w, c in zip(wraps, _comps(L, d, device)):
        w.comp = c
    yield model, wraps, L, d
    unpatch_v2(model)


def _gen(model, prompt, n):
    return model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=n,
                          min_new_tokens=n, do_sample=False, pad_token_id=2)[0, prompt.shape[1]:]


@pytest.mark.parametrize("kind", ["STATIC", "GUIDED"])
def test_policy_generation_equals_real_decode(v2, prompt, device, kind):
    model, wraps, L, d = v2
    P, n = 16, 20
    p = prompt[:, :P]
    if kind == "STATIC":
        S = {3, 9, 14, 20}
        real = decode_real(model, wraps, p, n + 1, skip=S)
        pol = Policy(model, wraps, L, kind="STATIC", static_set=S)
    else:
        r = TraversalRouter(d, L, seed=3).to(device)
        with torch.no_grad():
            r.b.weight.normal_(0, 1)
            r.w.weight.normal_(0, 1)
        r.configure(list(range(2, L - 2)), k_max=6, tau=0.0)
        real = decode_real(model, wraps, p, n + 1, router=r)
        pol = Policy(model, wraps, L, kind="GUIDED", router=r, tau_tf=0.0, tau_gen=0.0, k_max=6)
    pol.regime = "gen"
    with torch.inference_mode(), pol:
        toks = _gen(model, p, n + 1)
    assert torch.equal(toks.cpu(), real["tokens"].cpu())
    # decode_real faz n forwards de decode (o 1o token vem do prefill); generate faz o mesmo
    assert pol.skips == real["mlp_skipped"], (pol.skips, real["mlp_skipped"])
