"""Compare opt-in batched GMF against exact retained single-sequence code."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import types

import pytest
import torch

from src.gmf.ablations import ablation_plan
from src.gmf.model import GMFModel
from src.gmf.train import build_model, run_train
from src.gmf.eval import run_eval


PARENT_PATH = (Path(__file__).resolve().parents[1] / "research" / "verification"
               / "independent_batch_20261010" / "parent_model.py")
PARENT_SHA256 = "60971d7e2d662fc2cea8734c01a54315e5e2b1b035771281252fddd9307d2d82"


def retained_model():
    source = PARENT_PATH.read_bytes()
    assert hashlib.sha256(source).hexdigest() == PARENT_SHA256
    module = types.ModuleType("src.gmf._retained_model")
    module.__package__ = "src.gmf"
    exec(compile(source, str(PARENT_PATH), "exec"), module.__dict__)
    return module.GMFModel


def models(ablation="full", dtype=torch.float64, training=False):
    torch.manual_seed(7041)
    reference = retained_model()(dim=4, layers=2, memory_cells=5,
                                 refine_steps=2, ablation=ablation).to(dtype=dtype)
    candidate = GMFModel(dim=4, layers=2, memory_cells=5, refine_steps=2,
                         ablation=ablation, execution_mode="batched-v1").to(dtype=dtype)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    reference.train(training)
    candidate.train(training)
    return reference, candidate


@pytest.mark.parametrize("ablation", tuple(ablation_plan()))
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_parent_outputs_gradients_state_and_dropout_rng(ablation, training, dtype):
    reference, candidate = models(ablation, dtype, training)
    generator = torch.Generator().manual_seed(19)
    left = torch.randn(3, 4, 4, generator=generator, dtype=dtype, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    torch.manual_seed(10591)
    expected = reference(left)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(10591)
    actual = candidate(right)
    assert torch.equal(expected_rng, torch.get_rng_state())
    tolerance = dict(rtol=4e-4, atol=3e-6) if dtype == torch.float32 else dict(rtol=2e-10, atol=1e-12)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], **tolerance)
    # Include every output so uncertainty/controller/memory gradients are checked.
    sum(value.square().sum() for value in expected.values()).backward()
    sum(value.square().sum() for value in actual.values()).backward()
    torch.testing.assert_close(right.grad, left.grad, **tolerance)
    for (name, p), (_, q) in zip(reference.named_parameters(), candidate.named_parameters(), strict=True):
        if p.grad is None or q.grad is None:
            assert p.grad is q.grad is None, name
        else:
            torch.testing.assert_close(q.grad, p.grad, msg=name, **tolerance)
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], tensor, **tolerance)
    for p, q in zip(reference.encoder.layers, candidate.encoder.layers, strict=True):
        torch.testing.assert_close(q.state, p.state, **tolerance)


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("batch", [1, 4])
def test_each_sequence_has_private_memory_and_successive_calls_reset(batch, write):
    reference, candidate = models()
    data = torch.arange(batch * 3 * 4, dtype=torch.float64).reshape(batch, 3, 4) / 11
    expected, actual = reference(data, write_memory=write), candidate(data, write_memory=write)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=1e-12, rtol=2e-10)
    changed = data.clone()
    # Compare unchanged rows after a genuinely different first sequence.
    changed[0] = -data[0] + 40
    other = candidate(changed, write_memory=write)
    for key in actual:
        if batch > 1:
            torch.testing.assert_close(other[key][1:], actual[key][1:], rtol=0, atol=0)
    again = candidate(data, write_memory=write)
    for key in actual:
        torch.testing.assert_close(again[key], actual[key], rtol=0, atol=0)
    if not write:
        assert torch.count_nonzero(candidate.memory.fast.values) == 0
        assert torch.count_nonzero(candidate.memory.slow.values) == 0


def test_batching_reduces_encoder_invocations_from_batch_times_time_to_time():
    reference, candidate = models()
    counts = [0, 0]
    def count(index):
        def hook(*_args):
            counts[index] += 1
        return hook
    a = reference.encoder.register_forward_hook(count(0))
    b = candidate.encoder.register_forward_hook(count(1))
    try:
        data = torch.randn(5, 7, 4, dtype=torch.float64)
        reference(data)
        candidate(data)
    finally:
        a.remove()
        b.remove()
    assert counts == [35, 7]


def test_default_mode_preserves_retained_outputs_exactly():
    reference, _ = models(training=True)
    candidate = GMFModel(dim=4, layers=2, memory_cells=5, refine_steps=2).double().train()
    candidate.load_state_dict(reference.state_dict())
    data = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 13
    torch.manual_seed(655)
    expected = reference(data)
    torch.manual_seed(655)
    actual = candidate(data)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", [torch.zeros(0, 3, 4), torch.zeros(2, 0, 4),
                                   torch.zeros(2, 3, 5), torch.ones(2, 3, 4, dtype=torch.int64),
                                   torch.full((2, 3, 4), float("nan"))])
def test_invalid_batched_input_does_not_clear_existing_state(invalid):
    _, candidate = models(dtype=torch.float32)
    candidate(torch.ones(2, 3, 4))
    before = copy.deepcopy(candidate.state_dict())
    with pytest.raises(ValueError):
        candidate(invalid)
    for name, value in before.items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)


def test_explicit_mode_factory_and_checkpoint_to_evaluation(tmp_path):
    result = run_train(benchmark="temporal_prediction", model_name="gmf", dim=4,
                       memory_cells=5, seq_len=3, n_pairs=1, batch_size=2, steps=2,
                       seed=127, execution_mode="batched-v1", out_dir=str(tmp_path))
    directory = Path(result["run_dir"])
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["configuration"]["execution_mode"] == "batched-v1"
    payload = torch.load(directory / "checkpoint.pt", weights_only=True)
    assert payload["extra"]["execution_mode"] == "batched-v1"
    evaluated = run_eval(benchmark="temporal_prediction", model_name="gmf", dim=4,
                         memory_cells=5, seq_len=3, n_pairs=1, batch_size=2, batches=1,
                         seed=128, execution_mode="batched-v1", out_dir=str(tmp_path),
                         ckpt=str(directory / "checkpoint.pt"))
    assert evaluated["execution_mode"] == evaluated["trained_execution_mode"] == "batched-v1"
    assert len(result["history"]) == 2


@pytest.mark.parametrize("name", ["mlp", "gru", "transformer", "gmf"])
def test_factory_rejects_unsupported_modes(name):
    with pytest.raises(ValueError):
        build_model(name, 4, 5, execution_mode="unknown")
    if name != "gmf":
        with pytest.raises(ValueError):
            build_model(name, 4, 5, execution_mode="batched-v1")
