import torch
from torch import nn
import pytest
from vetor.core import InstrumentedMLP, VectorGuide, choose_mask, patch_llama_like_model, unpatch_llama_like_model


class CountingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.linear(x)


def test_original_equivalence_and_capture():
    torch.manual_seed(7)
    base = CountingMLP()
    wrapped = InstrumentedMLP(base)
    x = torch.randn(2, 5, 4)
    expected = base(x)
    wrapped.capture = True
    actual = wrapped(x)
    assert torch.equal(expected, actual)
    assert wrapped.last_input.shape == (2, 4)
    assert torch.equal(wrapped.last_input, x[:, -1, :])


def test_real_skip_does_not_call_inner_mlp():
    base = CountingMLP()
    wrapped = InstrumentedMLP(base)
    wrapped.skip_decode = True
    wrapped.capture = True
    x = torch.randn(1, 1, 4)
    before = base.calls
    out = wrapped(x)
    assert base.calls == before
    assert torch.equal(out, torch.zeros_like(x))
    assert wrapped.skipped_calls == 1
    assert torch.equal(wrapped.last_input, x[:, -1, :])


def test_never_skip_prefill():
    base = CountingMLP()
    wrapped = InstrumentedMLP(base)
    wrapped.skip_decode = True
    before = base.calls
    wrapped(torch.randn(1, 4, 4))
    assert base.calls == before + 1


def test_ablation_zeros_only_last_token():
    torch.manual_seed(8)
    base = CountingMLP()
    wrapped = InstrumentedMLP(base)
    x = torch.randn(1, 4, 4)
    ref = base(x)
    wrapped.ablate_last = True
    out = wrapped(x)
    assert torch.equal(out[:, :-1], ref[:, :-1])
    assert torch.equal(out[:, -1], torch.zeros_like(out[:, -1]))


def test_guide_logits_and_training():
    guide = VectorGuide(hidden_dim=4, num_layers=6, guide_dim=3)
    z = torch.randn(12, 4)
    layer_ids = torch.tensor([i % 6 for i in range(12)])
    logits = guide(z, layer_ids)
    assert logits.shape == (12,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.rand(12))
    loss.backward()
    assert guide.project.weight.grad is not None


def test_policy_protection_and_budget():
    mask = choose_mask([0.99] * 10, max_skips=3, threshold=0.9, protected_first=2, protected_last=2)
    assert mask[:2] == [True, True]
    assert mask[-2:] == [True, True]
    assert mask.count(False) == 3


def test_policy_rejects_bad_probabilities():
    with pytest.raises(ValueError):
        choose_mask([0.5, float('nan')], max_skips=1)


def test_patch_and_unpatch():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = CountingMLP()
    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(), Layer()])
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Core()
    m = Model()
    ids_before = [id(x.mlp) for x in m.model.layers]
    wraps = patch_llama_like_model(m)
    assert all(isinstance(x, InstrumentedMLP) for x in wraps)
    unpatch_llama_like_model(m)
    assert [id(x.mlp) for x in m.model.layers] == ids_before
