"""Gate 6: o roteador nao recebe alvo/KL/logits futuros; decisoes causais; splits por documento."""
import pytest
import torch

from vetor.decode import run_decode
from vetor.guide import VectorGuide
from vetor.policies import GuidedPolicy


def _guide(L=30, d=576):
    torch.manual_seed(3)
    g = VectorGuide(d, L, 16, seed=3)
    with torch.no_grad():
        g.b.weight.normal_()
    return g


def test_guided_ignores_context_with_labels():
    g = _guide()
    Z = torch.randn(30, 576)
    pol = GuidedPolicy(30, 6, g)
    a = pol.start(Z, {})
    b = pol.start(Z, {"oracle_kl": torch.rand(30), "target": 123, "sample_idx": 9})
    assert a == b


@pytest.mark.model
def test_dynamic_decision_uses_previous_token_only(patched, prompt, device):
    """A mascara aplicada no forward i deve ser funcao de z capturado no forward i-1."""
    model, _, wraps = patched
    g = _guide(len(wraps), model.config.hidden_size).to(device)
    seen = []
    orig = g.score_layers

    def spy(Z):
        seen.append(Z.clone())
        return orig(Z)

    g.score_layers = spy
    n = 6
    pol = GuidedPolicy(len(wraps), 5, g, K=1)
    r = run_decode(model, wraps, prompt, n, pol, collect_logp=False)
    # 1 decisao no prefill + 1 por forward de decode a partir do 2o (step>=1)
    assert len(seen) == 1 + (n - 2)
    toks = r["tokens"]
    # Reproduzir: z do prefill = entradas MLP do ultimo token do prompt
    from vetor.wrappers import reset, set_capture, stack_captured
    with torch.inference_mode():
        reset(wraps)
        set_capture(wraps, True)
        model(input_ids=prompt)
        z_prefill = stack_captured(wraps).clone()
        reset(wraps)
    assert torch.equal(seen[0], z_prefill)
    # A decisao do forward 1 usa z do forward 0 (token toks[0]), que nao depende de toks[1].
    assert not torch.equal(seen[1], seen[0])
    assert toks.shape == (n,)


def test_splits_are_disjoint_by_document(cfg):
    from vetor.collect import load_articles
    ids = {s: {d["title"] for d in load_articles(cfg, s)} for s in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])
