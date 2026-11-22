"""Gate 1: com 100% dos MLPs, o modelo instrumentado reproduz o original."""
import pytest
import torch

from vetor.decode import run_decode
from vetor.policies import FullPolicy
from vetor.wrappers import RoutedMLP, patch_model, unpatch_model

from conftest import TinyModel

pytestmark = pytest.mark.model


def test_unit_wrapper_identity():
    torch.manual_seed(0)
    m = TinyModel()
    x = torch.randn(2, 5, 4)
    ref = [l.mlp(x) for l in m.model.layers]
    wraps = patch_model(m)
    for w, r in zip(wraps, ref):
        assert torch.equal(w(x), r)
    unpatch_model(m)
    assert not any(isinstance(l.mlp, RoutedMLP) for l in m.model.layers)


def test_patched_logits_bitwise_equal(lm, prompt):
    model, _ = lm
    with torch.inference_mode():
        ref = model(input_ids=prompt).logits
        wraps = patch_model(model)
        try:
            out = model(input_ids=prompt).logits
        finally:
            unpatch_model(model)
    assert torch.equal(ref, out)


def test_loop_matches_generate(patched, prompt):
    model, _, wraps = patched
    n = 24
    r = run_decode(model, wraps, prompt, n, FullPolicy(len(wraps)))
    with torch.inference_mode():
        g = model.generate(prompt, max_new_tokens=n, min_new_tokens=n, do_sample=False)
    assert r["tokens"].tolist() == g[0, prompt.shape[1]:].tolist()
