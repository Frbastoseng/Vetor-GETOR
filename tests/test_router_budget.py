"""Gates 7, 8 e 9: orcamento, camadas protegidas, fallback FULL, sincronizacoes e taxas iguais."""
import pytest
import torch

from vetor.decode import run_decode
from vetor.guide import VectorGuide, topk_mask, validate_mask
from vetor.policies import GuidedPolicy, RandomPolicy, StaticPolicy
from vetor.wrappers import counters, reset_counters


@pytest.mark.parametrize("k", [0, 1, 5, 26, 40])
def test_topk_respects_protection_and_budget(k):
    torch.manual_seed(k)
    s = torch.randn(30)
    s[[0, 1, 28, 29]] = 100.0  # protegidas com score alto nao podem ser escolhidas
    a = topk_mask(s, k, protected_first=2, protected_last=2).tolist()
    assert a[:2] == [True, True] and a[-2:] == [True, True]
    assert a.count(False) == min(k, 26)


def test_threshold_reduces_skips():
    s = torch.linspace(-1, 1, 30)
    a = topk_mask(s, 10, protected_first=2, protected_last=2, min_score=0.5).tolist()
    assert 0 < a.count(False) <= 10
    assert all(s[i] >= 0.5 for i, v in enumerate(a) if not v)


def test_validate_mask():
    m = [True] * 30
    assert validate_mask(m, 0, 2, 2)
    m[0] = False
    assert not validate_mask(m, 5, 2, 2)
    m = [True] * 30
    for i in range(2, 9):
        m[i] = False
    assert not validate_mask(m, 6, 2, 2)


def test_guide_error_falls_back_to_full():
    class Broken(VectorGuide):
        def score_layers(self, Z):
            raise RuntimeError("falha simulada")
    pol = GuidedPolicy(30, 6, Broken(576, 30))
    assert pol.start(torch.randn(30, 576), {}) == [True] * 30
    assert pol.n_fallbacks == 1


def test_guide_nan_falls_back_to_full():
    g = VectorGuide(576, 30)
    pol = GuidedPolicy(30, 6, g)
    out = pol.start(torch.full((30, 576), float("nan")), {})
    # NaN em score nao pode resultar em omissao de camada protegida nem acima do orcamento
    assert validate_mask(out, 6, 2, 2)


@pytest.mark.model
def test_equal_effective_skip_rates(patched, prompt, device):
    model, _, wraps = patched
    L, k, n = len(wraps), 5, 9
    g = VectorGuide(model.config.hidden_size, L).to(device)
    pols = [RandomPolicy(L, k, seed=1), StaticPolicy(L, k, list(range(L))),
            GuidedPolicy(L, k, g), GuidedPolicy(L, k, g, K=2)]
    for p in pols:
        reset_counters(wraps)
        run_decode(model, wraps, prompt, n, p)
        _, skipped = counters(wraps)
        assert skipped == k * (n - 1), (p.name, skipped)


@pytest.mark.model
def test_no_global_sync_and_one_host_read_per_update(patched, prompt, device, monkeypatch):
    model, _, wraps = patched
    L, n = len(wraps), 17
    g = VectorGuide(model.config.hidden_size, L).to(device)
    n_sync, n_tolist = [0], [0]
    if device == "mps":
        real = torch.mps.synchronize
        monkeypatch.setattr(torch.mps, "synchronize", lambda: (n_sync.__setitem__(0, n_sync[0] + 1), real())[1])
    real_tolist = torch.Tensor.tolist

    def counting_tolist(self):
        n_tolist[0] += 1
        return real_tolist(self)

    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)
    pol = GuidedPolicy(L, 5, g, K=4)
    run_decode(model, wraps, prompt, n, pol)
    assert n_sync[0] == 0
    # 1 no prefill + atualizacoes nos steps 4, 8, 12 (forwards 0..15)
    assert pol.n_updates == 4
    assert n_tolist[0] == pol.n_updates
