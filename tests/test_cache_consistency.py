"""Gate 4: prefill completo + decode de 1 token com KV cache."""
import pytest
import torch
import torch.nn.functional as F

from vetor.decode import run_decode
from vetor.policies import FullPolicy, StaticPolicy

pytestmark = pytest.mark.model


def test_teacher_forced_cache_matches_full_sequence(patched, prompt):
    model, _, wraps = patched
    P = 16
    p, cont = prompt[:, :P], prompt[0, P:]
    n = cont.shape[0]
    r = run_decode(model, wraps, p, n, FullPolicy(len(wraps)), forced=cont, collect_logp=True)
    assert r["cache_len"] == P + n - 1
    with torch.inference_mode():
        ref = F.log_softmax(model(input_ids=prompt).logits[0, P - 1:-1].float(), -1)
    assert torch.allclose(r["logp"], ref, atol=1e-3), float((r["logp"] - ref).abs().max())


def test_cache_length_with_skips(patched, prompt):
    model, _, wraps = patched
    n = 12
    r = run_decode(model, wraps, prompt, n, StaticPolicy(len(wraps), 8, list(range(2, 28))))
    assert r["cache_len"] == prompt.shape[1] + n - 1
    assert r["tokens"].shape == (n,)
