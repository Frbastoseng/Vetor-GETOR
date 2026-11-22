"""Gate 3: ablate_last zera somente o ultimo token das linhas indicadas."""
import pytest
import torch
import torch.nn.functional as F

from vetor.collect import ablation_teacher
from vetor.wrappers import RoutedMLP

from conftest import CountingMLP


def test_unit_ablate_rows_last_token_only():
    torch.manual_seed(1)
    base = CountingMLP()
    w = RoutedMLP(base, 0)
    x = torch.randn(3, 5, 4)
    ref = base(x)
    w.ablate_rows = torch.tensor([1])
    out = w(x)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[2], ref[2])
    assert torch.equal(out[1, :-1], ref[1, :-1])
    assert torch.count_nonzero(out[1, -1]) == 0


@pytest.mark.model
def test_teacher_batch_matches_individual_runs(patched, prompt, device):
    model, _, wraps = patched
    prefix = prompt[0, :-1]
    target = int(prompt[0, -1])
    t = ablation_teacher(model, wraps, prefix, target, device)
    assert t["z"].shape == (len(wraps), model.config.hidden_size)
    assert torch.all(t["kl"] >= 0)
    with torch.inference_mode():
        full = model(input_ids=prefix.view(1, -1)).logits[0, -1].float()
        for l in (0, 7, 29):
            wraps[l].ablate_rows = torch.tensor([0], device=device)
            alt = model(input_ids=prefix.view(1, -1)).logits[0, -1].float()
            wraps[l].ablate_rows = None
            lp0, lpl = F.log_softmax(full, -1), F.log_softmax(alt, -1)
            kl = float((lp0.exp() * (lp0 - lpl)).sum())
            assert abs(kl - float(t["kl"][l])) < 1e-3 + 1e-2 * kl, (l, kl, float(t["kl"][l]))
