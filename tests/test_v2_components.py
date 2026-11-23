"""v0.2: compensadores, dados pre-registrados, professor acumulado e orcamento do roteador."""
import json

import pytest
import torch

from vetor.v2.compensate import StatsAccumulator, fit_layer, make_comps, r2_layer
from vetor.v2.data import MANIFEST, load_split
from vetor.v2.engine import Compensator, Ctrl, patch_v2, unpatch_v2, zero_comps
from vetor.v2.fasteval import eval_masks, seq_mask
from vetor.v2.router2 import TraversalRouter
from vetor.v2.teacher2 import label_window


# ----------------------------------------------------------------------------- compensadores
def _acc_from(Z, Y):
    acc = StatsAccumulator(1, Z.shape[1])
    acc.add(0, Z, Y)
    acc.n = Z.shape[0]
    return acc


def test_rrr_recovers_low_rank_map_and_r2_matches_empirical():
    torch.manual_seed(0)
    d, r, n = 24, 3, 6000
    A = torch.randn(d, r) @ torch.randn(r, d) / 4
    Z = torch.randn(n, d) + 0.5
    Y = Z @ A + 0.3 + 0.01 * torch.randn(n, d)
    acc = _acc_from(Z, Y)
    fit = fit_layer(acc, 0, r, 1e-8)
    comp = Compensator("lr", d, r)
    comp.ybar.copy_(fit["ybar"].float()); comp.zbar.copy_(fit["zbar"].float())
    comp.V.copy_(fit["V"].float()); comp.U.copy_(fit["U"].float())
    pred = comp(Z.unsqueeze(0))[0]
    emp_r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    assert emp_r2 > 0.99
    assert abs(r2_layer(acc, 0, fit) - float(emp_r2)) < 1e-3
    # rank menor que o verdadeiro perde explicacao
    assert r2_layer(acc, 0, fit_layer(acc, 0, 1, 1e-8)) < 0.9


def test_mean_and_zero_r2():
    torch.manual_seed(1)
    Z, Y = torch.randn(4000, 8), torch.randn(4000, 8) + 2.0
    acc = _acc_from(Z, Y)
    mean_fit = {"ybar": acc.moments(0, acc.n)[1]}
    assert abs(r2_layer(acc, 0, mean_fit)) < 1e-6
    assert r2_layer(acc, 0, None) < -1  # zerar com media 2 e pior que a media


def test_compensator_costs():
    assert Compensator("zero", 576).flops() == 0
    assert Compensator("mean", 576).flops() == 576
    assert Compensator("lr", 576, 32).flops() == 2 * 576 * 32 + 2 * 576


# ----------------------------------------------------------------------------- dados
def test_final_split_is_locked_and_disjoint():
    with pytest.raises(PermissionError):
        load_split("final")
    man = json.loads(MANIFEST.read_text())
    ids = {k: {e["id"] for e in v} for k, v in man["splits"].items()}
    keys = list(ids)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not (ids[a] & ids[b]), (a, b)
    assert man["counts"]["final"] == 200


# ----------------------------------------------------------------------------- modelo real
@pytest.fixture
def v2(lm, device):
    model, _ = lm
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    wraps = patch_v2(model, zero_comps(L, d, device))
    yield model, wraps, L, d
    unpatch_v2(model)


@pytest.mark.model
def test_teacher_label_matches_direct_evaluation(v2, prompt, device):
    model, wraps, L, d = v2
    P = 12
    window = prompt[0]
    T = window.shape[0] - 1
    base = torch.zeros(T, L, dtype=torch.bool)
    base[P:, 9] = True  # base ja omite a camada 9
    pos, layers = [P + 2], [5, 9]
    lab = label_window(model, wraps, window, P, base, pos, layers, device)
    # celula (P+2, 5): base executa -> linha omite; Delta = sum KL(base || base+(t,5))
    m_base = base.unsqueeze(0).to(device)
    alt = base.clone(); alt[P + 2, 5] = True
    from vetor.v2.fasteval import Ctrl as _C, logprobs_D, kl_rows
    x = window[:-1].view(1, -1)
    lpb = logprobs_D(model, wraps, x, P, _C(start=P, L=L, static_mask=m_base))
    lpa = logprobs_D(model, wraps, x, P, _C(start=P, L=L, static_mask=alt.unsqueeze(0).to(device)))
    kl = kl_rows(lpb, lpa)[0]
    p0 = 2
    D = kl.shape[0]
    expect8 = float(kl[p0:min(p0 + 8, D)].sum())
    assert abs(float(lab["d8"][0]) - expect8) < 1e-3 + 1e-3 * expect8
    assert not bool(lab["was_skipped"][0]) and bool(lab["was_skipped"][1])
    assert lab["bits"][0].sum() == 0          # camadas >= 5 zeradas nas features
    assert bool(lab["bits"][1, 9]) is False   # a propria camada nao entra nas features
    assert torch.all(lab["d32"] >= lab["d8"] - 1e-6)  # acumulado cresce com N (KL >= 0)


@pytest.mark.model
def test_router_budget_and_protection_masked(v2, prompt, device):
    model, wraps, L, d = v2
    r = TraversalRouter(d, L, seed=3).to(device)
    with torch.no_grad():
        r.b.weight.fill_(-5.0)  # roteador "quer" omitir tudo
    r.configure(list(range(2, L - 2)), k_max=4, tau=0.0)
    P = 10
    x = prompt[:, :-1]
    ctrl = Ctrl(start=P, L=L, router=r.masked, eligible=r.eligible)
    from vetor.v2.fasteval import logprobs_D
    logprobs_D(model, wraps, x, P, ctrl)
    assert int(ctrl.count.max()) == 4
    assert not ctrl.bits[:, :P].any()
    assert not ctrl.bits[:, :, [0, 1, L - 2, L - 1]].any()
