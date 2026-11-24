"""v0.3: modulos comecam identicos ao FULL, LoRA fundido e identico, portao sem sincronizacao, pesos congelados."""
import pytest
import torch

from vetor.model_adapter import weights_fingerprint
from vetor.v3.modules import attach_lora, attach_vetor, detach_all, merge_lora, n_params

pytestmark = pytest.mark.model


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_vetor_starts_identical_and_detaches(lm, prompt):
    model, _ = lm
    ref = _logits(model, prompt)
    mods = attach_vetor(model, [4, 8, 12], r=8, guided=True)
    try:
        assert torch.equal(_logits(model, prompt), ref)  # B = 0 -> identico
        assert n_params(mods["corrections"], mods["guide"]) > 0
    finally:
        detach_all(model)
    assert torch.equal(_logits(model, prompt), ref)


def test_lora_merge_is_exact(lm, prompt):
    model, _ = lm
    before = weights_fingerprint(model)
    mods = attach_lora(model, r=1)
    try:
        with torch.no_grad():
            for m in mods:
                m.B.normal_(0, 0.02)
        unmerged = _logits(model, prompt)
        n = merge_lora(model)
        assert n == len(mods) == 60
        merged = _logits(model, prompt)
        assert torch.allclose(unmerged, merged, atol=2e-3), (unmerged - merged).abs().max()
    finally:
        detach_all(model)
    # pesos originais intactos (LoRA fundido cria Linear novo; o original nao e tocado)
    assert weights_fingerprint(model) == before


def test_gate_has_no_host_sync(lm, prompt, monkeypatch):
    model, _ = lm
    mods = attach_vetor(model, [4, 8, 12, 16], r=8, guided=True)
    calls = {"item": 0, "tolist": 0}
    ri, rt = torch.Tensor.item, torch.Tensor.tolist
    monkeypatch.setattr(torch.Tensor, "item", lambda s: (calls.__setitem__("item", calls["item"] + 1), ri(s))[1])
    monkeypatch.setattr(torch.Tensor, "tolist", lambda s: (calls.__setitem__("tolist", calls["tolist"] + 1), rt(s))[1])
    try:
        with torch.no_grad():
            out = model(input_ids=prompt[:, :1], use_cache=False)
        assert calls == {"item": 0, "tolist": 0}
        assert mods["guide"] is not None
    finally:
        detach_all(model)


def test_only_modules_receive_gradients(lm, prompt):
    model, _ = lm
    mods = attach_vetor(model, [6, 10], r=4, guided=True)
    try:
        with torch.no_grad():
            for c in mods["corrections"].values():
                c.B.normal_(0, 0.01)
        loss = model(input_ids=prompt, labels=prompt).loss
        loss.backward()
        assert all(p.grad is None for n, p in model.named_parameters() if ".corr." not in n and "guide" not in n)
        assert all(c.A.grad is not None for c in mods["corrections"].values())
        assert mods["guide"].w.grad is not None
    finally:
        detach_all(model)
