"""Probability and gradient contracts for Gaussian memory, without benchmark runs."""

import pytest
import torch
import torch.nn.functional as F

from src.gmf.model import GaussianMemoryBank, GMFModel


def _reference(query, centers, sigma, mode):
    # Explicit pairwise subtraction is independent of the cdist kernel.
    distance2 = (query[:, None, :] - centers[None, :, :]).square().sum(-1)
    log_kernel = -distance2 / (2 * sigma**2)
    cosine = F.normalize(query, dim=-1) @ F.normalize(centers, dim=-1).t()
    if mode == "cosine":
        return torch.softmax(cosine, dim=-1)
    if mode == "gaussian":
        return torch.softmax(log_kernel, dim=-1)
    return torch.softmax(log_kernel.exp() + 0.5 * (cosine + 1), dim=-1)


@pytest.mark.parametrize("mode", ["gaussian", "cosine", "hybrid"])
@pytest.mark.parametrize("sigma", [0.2, 1., 2.])
def test_probabilities_match_independent_kernel_reference(mode, sigma):
    torch.manual_seed(3)
    bank = GaussianMemoryBank(3, cells=5, sigma=sigma, mode=mode).double()
    query = torch.randn(4, 3, dtype=torch.float64)
    weights = bank._similarity(query)
    expected = _reference(query, bank.centers, sigma, mode)
    torch.testing.assert_close(weights, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(weights.sum(-1), torch.ones(4, dtype=torch.float64))
    assert (weights >= 0).all()


def test_far_gaussian_query_keeps_mass_and_recovers_nearest_value():
    bank = GaussianMemoryBank(2, cells=2, sigma=1., mode="gaussian")
    with torch.no_grad():
        bank.centers.copy_(torch.tensor([[0., 0.], [1., 0.]]))
        bank.values.copy_(torch.eye(2))
    query = torch.tensor([[50., 0.]])
    weights = bank._similarity(query)
    torch.testing.assert_close(weights, torch.softmax(torch.tensor([[-1250., -1200.5]]), -1))
    assert weights.sum().item() == pytest.approx(1.)
    torch.testing.assert_close(bank.read(query), torch.tensor([[0., 1.]]))


def test_almost_equal_large_vectors_do_not_overflow_hybrid_kernel():
    torch.manual_seed(53)
    bank = GaussianMemoryBank(8, cells=8, sigma=1., mode="hybrid")
    with torch.no_grad():
        bank.centers.copy_(torch.randn_like(bank.centers) * 1e5)
    query = bank.centers.detach().clone() + 0.01
    weights = bank._similarity(query)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights, _reference(query, bank.centers, 1., "hybrid"))
    torch.testing.assert_close(weights.sum(-1), torch.ones(8))


@pytest.mark.parametrize("mode", ["gaussian", "cosine", "hybrid"])
def test_query_and_center_gradients_match_kernel_reference(mode):
    torch.manual_seed(17)
    bank = GaussianMemoryBank(2, cells=3, sigma=0.7, mode=mode).double()
    query = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    coefficients = torch.tensor([[1., -2., 3.], [2., 1., -4.]], dtype=torch.float64)
    actual = (bank._similarity(query) * coefficients).sum()
    expected = (_reference(query, bank.centers, 0.7, mode) * coefficients).sum()
    actual_grads = torch.autograd.grad(actual, (query, bank.centers), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (query, bank.centers))
    for a, e in zip(actual_grads, expected_grads):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)
    assert torch.autograd.gradcheck(bank._similarity, (query,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["gaussian", "hybrid"])
def test_half_precision_kernel_has_finite_unit_mass(dtype, mode):
    torch.manual_seed(8)
    bank = GaussianMemoryBank(4, cells=6, mode=mode).to(dtype=dtype)
    query = torch.randn(2, 4, dtype=dtype)
    weights = bank._similarity(query)
    assert weights.dtype == dtype
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, dtype=dtype))
    assert torch.isfinite(bank(query)).all()
    assert torch.isfinite(bank.values).all()


@pytest.mark.parametrize("mode", ["gaussian", "cosine", "hybrid"])
def test_write_matches_declared_weighted_update(mode):
    torch.manual_seed(6)
    bank = GaussianMemoryBank(3, cells=4, mode=mode, decay=0.7, lr=0.2).double()
    bank.values.copy_(torch.randn_like(bank.values))
    original = bank.values.clone()
    query = torch.randn(2, 3, dtype=torch.float64)
    weights = _reference(query, bank.centers, bank.sigma, mode)
    update = weights.t() @ query / weights.sum(0).unsqueeze(1).clamp_min(1e-6)
    bank.write(query)
    torch.testing.assert_close(bank.values, original * 0.7 + 0.2 * update, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bad", [
    torch.ones(2), torch.ones(2, 3), torch.empty(0, 2),
    torch.ones(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.int64),
    torch.ones(1, 2, dtype=torch.complex64),
    torch.full((1, 2), float("nan")), torch.full((1, 2), float("inf")),
])
def test_bad_write_cannot_corrupt_memory(bad):
    bank = GaussianMemoryBank(2, cells=3)
    bank.values.fill_(0.5)
    original = bank.values.clone()
    with pytest.raises((ValueError, TypeError)):
        bank.write(bad)
    assert torch.equal(bank.values, original)


def test_overflowing_write_retains_previous_memory():
    bank = GaussianMemoryBank(2, cells=3, mode="cosine", lr=torch.finfo(torch.float32).max)
    original = bank.values.clone()
    with pytest.raises(FloatingPointError, match="memory update"):
        bank.write(torch.full((1, 2), 2.))
    assert torch.equal(bank.values, original)


@pytest.mark.parametrize("kwargs", [
    {"dim": 0}, {"dim": True}, {"cells": 0}, {"cells": 1.5},
    {"sigma": 0.}, {"sigma": -1.}, {"sigma": float("nan")},
    {"sigma": float("inf")}, {"decay": -0.1}, {"decay": 1.1},
    {"decay": float("nan")}, {"lr": -1.}, {"lr": float("inf")},
    {"mode": "unknown"},
])
def test_invalid_bank_configuration_fails_before_execution(kwargs):
    options = {"dim": 2, "cells": 3, **kwargs}
    with pytest.raises((TypeError, ValueError)):
        GaussianMemoryBank(**options)


def test_real_gmf_sequence_backward_stays_valid_across_memory_writes():
    torch.manual_seed(21)
    model = GMFModel(dim=8, layers=1, memory_cells=4, refine_steps=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 3, 8)
    result = model(inputs)["prediction"]
    result.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(inputs)["prediction"]).all()
