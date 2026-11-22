"""Gates 5 e reprodutibilidade: pesos congelados, saidas deterministicas."""
import pytest
import torch

from vetor.decode import run_decode
from vetor.guide import VectorGuide
from vetor.model_adapter import weights_fingerprint
from vetor.policies import GuidedPolicy, RandomPolicy, StaticPolicy

pytestmark = pytest.mark.model


def test_weights_frozen_through_routing_and_guide_training(patched, prompt, device):
    model, _, wraps = patched
    before = weights_fingerprint(model)
    L = len(wraps)
    run_decode(model, wraps, prompt, 8, StaticPolicy(L, 8, list(range(2, 28))))
    g = VectorGuide(model.config.hidden_size, L).to(device)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-2)
    Z = torch.randn(64, model.config.hidden_size, device=device)
    ids = torch.randint(0, L, (64,), device=device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(g(Z, ids), torch.rand(64, device=device))
    loss.backward()
    opt.step()
    assert all(p.grad is None for p in model.parameters())
    assert all(not p.requires_grad for p in model.parameters())
    assert weights_fingerprint(model) == before


def test_deterministic_decode(patched, prompt):
    model, _, wraps = patched
    L = len(wraps)
    pol = StaticPolicy(L, 5, list(range(2, 28)))
    a = run_decode(model, wraps, prompt, 16, pol)["tokens"].tolist()
    b = run_decode(model, wraps, prompt, 16, pol)["tokens"].tolist()
    assert a == b


def test_random_policy_seeded():
    Z = torch.zeros(30, 4)
    p = RandomPolicy(30, 6, seed=7)
    assert p.start(Z, {"sample_idx": 3}) == p.start(Z, {"sample_idx": 3})
    assert p.start(Z, {"sample_idx": 3}) != p.start(Z, {"sample_idx": 4})
    assert p.start(Z, {"sample_idx": 3}).count(False) == 6


def test_guided_deterministic(device):
    g = VectorGuide(576, 30, seed=5).to(device)
    Z = torch.randn(30, 576, device=device)
    pol = GuidedPolicy(30, 6, g)
    assert pol.start(Z, {}) == pol.start(Z, {})
