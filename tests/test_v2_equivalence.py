"""v0.2: o modo mascarado em lote (qualidade) e identico ao decode com execucao condicional real.

Sem essa equivalencia, nenhum numero de qualidade da v0.2 seria valido.
"""
import pytest
import torch

from vetor.v2.decode2 import decode_real
from vetor.v2.engine import Compensator, patch_v2, unpatch_v2, zero_comps
from vetor.v2.fasteval import logprobs_D, seq_mask, window_io
from vetor.v2.engine import Ctrl
from vetor.v2.router2 import TraversalRouter

pytestmark = pytest.mark.model
TOL = 2e-3


def _comps(kind, L, d, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    cs = []
    for _ in range(L):
        c = Compensator(kind, d, rank=8 if kind == "lr" else 0)
        c.ybar.copy_(torch.randn(d, generator=g) * 0.05)
        if kind == "lr":
            c.zbar.copy_(torch.randn(d, generator=g) * 0.1)
            c.V.copy_(torch.randn(d, 8, generator=g) / d ** 0.5)
            c.U.copy_(torch.randn(8, d, generator=g) * 0.05)
        cs.append(c.to(device))
    return cs


@pytest.fixture
def v2(lm, device):
    model, _ = lm
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    yield model, wraps, L, d
    unpatch_v2(model)


@pytest.mark.parametrize("kind", ["zero", "mean", "lr"])
def test_masked_equals_real_skip_static(v2, prompt, device, kind):
    model, wraps, L, d = v2
    comps = _comps(kind, L, d, device)
    for w, c in zip(wraps, comps):
        w.comp = c
    P = 16
    window = prompt[0]
    n = window.shape[0] - P
    skip = {3, 7, 12, 20}
    real = decode_real(model, wraps, window[:P].view(1, -1), n, skip=skip, forced=window[P:], collect_logp=True)
    x, _ = window_io(window, P)
    m = seq_mask(sorted(skip), x.shape[1], P, L, device).unsqueeze(0)
    lp = logprobs_D(model, wraps, x, P, Ctrl(start=P, L=L, static_mask=m))[0]
    # real["logp"][1:] = logits dos forwards de decode (passos 1..n-1)
    diff = (real["logp"][1:] - lp).abs().max().item()
    assert diff < TOL, diff
    assert real["mlp_skipped"] == len(skip) * (n - 1)


@pytest.mark.parametrize("lagged", [False, True])
def test_masked_equals_real_traversal_router(v2, prompt, device, lagged):
    model, wraps, L, d = v2
    for w, c in zip(wraps, _comps("mean", L, d, device)):
        w.comp = c
    r = TraversalRouter(d, L, lagged=lagged, seed=1).to(device)
    with torch.no_grad():
        r.b.weight.normal_(0, 1)
        r.w.weight.normal_(0, 1)
    r.configure(list(range(2, L - 2)), k_max=6, tau=0.0)
    P = 16
    window = prompt[0]
    n = window.shape[0] - P
    real = decode_real(model, wraps, window[:P].view(1, -1), n, router=r, forced=window[P:], collect_logp=True)
    x, _ = window_io(window, P)
    ctrl = Ctrl(start=P, L=L, router=r.masked, eligible=r.eligible)
    lp = logprobs_D(model, wraps, x, P, ctrl)[0]
    assert real["mlp_skipped"] == int(ctrl.bits[0, P:].sum()), "decisoes diferentes"
    assert real["mlp_skipped"] > 0
    assert int(ctrl.count[0, P:].max()) <= 6
    assert not ctrl.bits[0, :P].any() and not ctrl.bits[0, :, [0, 1, 28, 29]].any()
    diff = (real["logp"][1:] - lp).abs().max().item()
    assert diff < TOL, diff


def test_traversal_router_is_causal(v2, prompt, device):
    """Mudar tokens FUTUROS nao altera decisoes nem logits das posicoes anteriores."""
    model, wraps, L, d = v2
    r = TraversalRouter(d, L, seed=2).to(device)
    with torch.no_grad():
        r.b.weight.normal_(0, 1)
    r.configure(list(range(2, L - 2)), k_max=6, tau=0.0)
    P, cut = 16, 24
    a = prompt[0].clone()
    b = a.clone()
    b[cut + 1:] = (b[cut + 1:] + 17) % 49152
    outs = []
    for w_ in (a, b):
        x, _ = window_io(w_, P)
        ctrl = Ctrl(start=P, L=L, router=r.masked, eligible=r.eligible)
        lp = logprobs_D(model, wraps, x, P, ctrl)
        outs.append((ctrl.bits[0, : cut + 1].clone(), lp[0, : cut + 1 - P].clone()))
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.allclose(outs[0][1], outs[1][1], atol=1e-4)
