"""Gate 2: skip real — zero chamadas ao MLP original em decode; prefill nunca pulado."""
import pytest
import torch

from vetor.decode import run_decode
from vetor.policies import StaticPolicy
from vetor.wrappers import RoutedMLP

from conftest import CountingMLP


def test_unit_skip_zero_calls_same_shape_dtype_device():
    base = CountingMLP()
    w = RoutedMLP(base, 0)
    w.active = False
    w.capture = True
    x = torch.randn(1, 1, 4, dtype=torch.float64)
    base.double()
    out = w(x)
    assert base.calls == 0
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
    assert torch.count_nonzero(out) == 0
    assert w.calls_skipped == 1
    assert torch.equal(w.captured, x[:, -1, :])  # captura mesmo quando pula


def test_unit_never_skip_prefill():
    base = CountingMLP()
    w = RoutedMLP(base, 0)
    w.active = False
    w(torch.randn(1, 4, 4))
    assert base.calls == 1


@pytest.mark.model
def test_model_skip_counts(patched, prompt):
    model, _, wraps = patched
    calls = [0] * len(wraps)
    hooks = [w.inner.register_forward_hook(lambda m, i, o, l=l: calls.__setitem__(l, calls[l] + 1))
             for l, w in enumerate(wraps)]
    skip = [5, 9, 14]
    ranking = skip + [l for l in range(len(wraps)) if l not in skip]
    n = 10
    try:
        run_decode(model, wraps, prompt, n, StaticPolicy(len(wraps), 3, ranking))
    finally:
        for h in hooks:
            h.remove()
    for l in range(len(wraps)):
        expected = 1 if l in skip else 1 + (n - 1)  # prefill sempre + decode se ativo
        assert calls[l] == expected, (l, calls[l])
