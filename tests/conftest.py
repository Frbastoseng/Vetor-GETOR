import pytest
import torch
from torch import nn

from vetor.model_adapter import load_config, load_model, pick_device


class CountingMLP(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.linear(x)


class TinyLayer(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.mlp = CountingMLP(d)


class TinyModel(nn.Module):
    def __init__(self, n=6, d=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(d) for _ in range(n)])


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def device(cfg):
    return pick_device(cfg["device"])


@pytest.fixture(scope="session")
def lm(cfg, device):
    model, tok = load_model(cfg, device)
    return model, tok


@pytest.fixture
def patched(lm):
    from vetor.wrappers import patch_model, unpatch_model
    model, tok = lm
    wraps = patch_model(model)
    yield model, tok, wraps
    unpatch_model(model)


@pytest.fixture(scope="session")
def prompt(lm, device):
    _, tok = lm
    text = ("The history of the city begins in the Roman period, when a small settlement was "
            "founded on the river bank. Over the following centuries the town grew into")
    return tok(text, return_tensors="pt").input_ids.to(device)
