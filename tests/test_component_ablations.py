from __future__ import annotations

import hashlib
import json
import os
import subprocess
from unittest.mock import patch

import pytest
import torch

from src.gmf.ablations import ablation_plan, build_ablation, run_component_ablations
from src.gmf.eval import build_model as build_eval_model
from src.gmf.eval import run_eval
from src.gmf.model import GMFModel
from src.gmf.train import build_model as build_train_model
from src.gmf.train import run_train


def _model(name="full"):
    torch.manual_seed(93)
    return build_ablation(
        name, dim=4, layers=1, memory_cells=5, refine_steps=2
    ).double().eval()


def _input():
    generator = torch.Generator().manual_seed(127)
    return torch.randn(2, 3, 4, generator=generator, dtype=torch.float64)


def _legacy_full_forward(model, seq):
    """Retained forward equations from main 9f08ce9, without control dispatch."""
    rows = []
    for sequence in seq:
        model.reset_state()
        latent = None
        for token in sequence:
            z = model.encoder(token.unsqueeze(0))
            memory = model.memory(z, write=True)
            gates = model.controller(z, memory)
            fused = (
                gates[:, :1] * z
                + gates[:, 1:2] * memory
                + gates[:, 2:3] * 0.5 * (z + memory)
            )
            latent = (
                fused
                if latent is None
                else model.latent_decay * latent + (1.0 - model.latent_decay) * fused
            )
            for _ in range(model.refine_steps):
                latent = model.final_norm(latent + 0.15 * model.post(latent))
        rows.append(
            {
                "latent": latent.squeeze(0),
                "memory": memory.squeeze(0),
                "gates": gates.squeeze(0),
                "prediction": model.predictor(latent).squeeze(0),
                "uncertainty": model.uncertainty(latent).squeeze(0),
            }
        )
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


@pytest.mark.parametrize("training", [False, True])
def test_full_preserves_retained_outputs_and_gradients_exactly(training):
    actual, reference = _model(), _model()
    actual.train(training)
    reference.train(training)
    torch.manual_seed(441)
    result = actual(_input())
    torch.manual_seed(441)
    expected = _legacy_full_forward(reference, _input())
    for key in result:
        torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)
    result["prediction"].square().sum().backward()
    expected["prediction"].square().sum().backward()
    for (name, left), (_, right) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        if left.grad is None or right.grad is None:
            assert left.grad is right.grad is None, name
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize("name", tuple(ablation_plan()))
def test_all_controls_preserve_checkpoint_keys_and_allocated_parameters(name):
    full, control = _model(), _model(name)
    control.load_state_dict(full.state_dict(), strict=True)
    assert list(control.state_dict()) == list(full.state_dict())
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in full.parameters()
    )
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)


def test_no_memory_bypasses_reads_writes_and_memory_gradients():
    model = _model("no_memory")
    with patch.object(model.memory, "forward", side_effect=AssertionError("memory ran")):
        result = model(_input())
    assert torch.count_nonzero(result["memory"]) == 0
    assert torch.count_nonzero(model.memory.fast.values) == 0
    assert torch.count_nonzero(model.memory.slow.values) == 0
    result["prediction"].square().sum().backward()
    assert all(p.grad is None for p in model.memory.parameters())
    assert model.encoder.layers[0].W.grad.abs().sum() > 0
    assert model.controller.net[0].weight.grad.abs().sum() > 0


def test_no_controller_uses_fixed_symmetric_fusion_and_keeps_memory_active():
    model = _model("no_controller")
    with patch.object(model.controller, "forward", side_effect=AssertionError("gates ran")):
        result = model(_input())
    torch.testing.assert_close(result["gates"], torch.full_like(result["gates"], 1 / 3))
    assert torch.count_nonzero(result["memory"]) > 0
    result["prediction"].square().sum().backward()
    assert all(p.grad is None for p in model.controller.parameters())
    assert model.memory.fast.centers.grad.abs().sum() > 0


def test_no_refine_bypasses_both_refinement_modules_and_gradients():
    model = _model("no_refine")
    with (
        patch.object(model.post, "forward", side_effect=AssertionError("post ran")),
        patch.object(model.final_norm, "forward", side_effect=AssertionError("norm ran")),
    ):
        result = model(_input())
    assert model.refine_steps == 2
    result["prediction"].square().sum().backward()
    assert all(p.grad is None for p in model.post.parameters())
    assert all(p.grad is None for p in model.final_norm.parameters())
    assert model.controller.net[0].weight.grad.abs().sum() > 0


def test_single_memory_uses_fast_read_and_never_executes_slow_bank():
    model = _model("single_memory")
    reads = []
    handle = model.memory.fast.register_forward_hook(
        lambda _module, _inputs, output: reads.append(output.detach().clone())
    )
    try:
        with (
            patch.object(model.memory.slow, "read", side_effect=AssertionError("slow read")),
            patch.object(model.memory.slow, "write", side_effect=AssertionError("slow write")),
        ):
            result = model(_input())
    finally:
        handle.remove()
    torch.testing.assert_close(result["memory"], torch.cat([reads[2], reads[5]]))
    assert torch.count_nonzero(model.memory.fast.values) > 0
    assert torch.count_nonzero(model.memory.slow.values) == 0
    result["prediction"].square().sum().backward()
    assert model.memory.fast.centers.grad.abs().sum() > 0
    assert model.memory.slow.centers.grad is None


@pytest.mark.parametrize("name", [name for name in ablation_plan() if name != "full"])
def test_each_control_has_a_discriminating_output(name):
    expected = _model()(_input())["prediction"]
    actual = _model(name)(_input())["prediction"]
    assert not torch.allclose(actual, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", tuple(ablation_plan()))
def test_train_and_eval_factories_construct_the_same_control(name):
    left = build_train_model(f"gmf_{name}", 4, 5)
    right = build_eval_model(f"gmf_{name}", 4, 5)
    assert left.ablation == right.ablation == name
    assert list(left.state_dict()) == list(right.state_dict())


def test_invalid_control_fails_without_advancing_initialization_rng():
    torch.manual_seed(6)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="unknown GMF ablation"):
        build_ablation("no_such_component", dim=4)
    assert torch.equal(before, torch.get_rng_state())


def test_plan_mutation_does_not_modify_later_models():
    plan = ablation_plan()
    plan["full"]["memory"] = False
    assert _model()._ablation_settings["memory"] is True
    assert GMFModel(dim=4).ablation == "full"


def test_control_can_train_save_reload_and_evaluate(tmp_path):
    trained = run_train(
        model_name="gmf_no_memory", dim=4, memory_cells=5, n_pairs=2,
        steps=1, batch_size=2, out_dir=str(tmp_path),
    )
    result = run_eval(
        model_name="gmf_no_memory", dim=4, memory_cells=5, n_pairs=2,
        batches=1, batch_size=2,
        ckpt=str(tmp_path / "associative_recall_gmf_no_memory_seed0" / "checkpoint.pt"),
        out_dir=str(tmp_path),
    )
    assert len(trained["history"]) == 1
    assert result["model"] == "gmf_no_memory"
    assert torch.isfinite(torch.tensor(result["agg"]["mse_mean"]))


def test_shell_entrypoint_executes_all_five_controls_with_retained_artifacts(tmp_path):
    output = tmp_path / "matrix"
    subprocess.run(
        [
            "bash", "experiments/run_ablation.sh", "--out-dir", str(output),
            "--dim", "4", "--memory-cells", "5", "--n-pairs", "1",
            "--seq-len", "3", "--batch-size", "2", "--steps", "2",
            "--eval-batches", "2",
        ],
        check=True, capture_output=True, text=True, env=os.environ.copy(),
    )
    manifest = json.loads((output / "component_ablations.json").read_text())
    assert manifest["status"] == "completed"
    assert [row["ablation"] for row in manifest["runs"]] == list(ablation_plan())
    assert len(manifest["source_sha256"]) == 7
    train_hashes = manifest["paired_streams"]["train"]["batch_sha256"]
    eval_hashes = manifest["paired_streams"]["evaluation"]["batch_sha256"]
    assert len(set(train_hashes)) == len(set(eval_hashes)) == 2
    assert set(train_hashes).isdisjoint(eval_hashes)
    for row in manifest["runs"]:
        checkpoint = output / row["checkpoint"]
        assert row["status"] == "completed"
        assert row["model_name"] == f"gmf_{row['ablation']}"
        assert row["evaluation"]["model"] == row["model_name"]
        assert row["pairing_verified"] is True
        history = json.loads((checkpoint.parent / "train_history.json").read_text())
        assert [item["batch_sha256"] for item in history] == train_hashes
        assert [item["batch_sha256"] for item in row["evaluation"]["metrics"]] == eval_hashes
        assert row["evaluation"]["checkpoint_sha256"] == row["checkpoint_sha256"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["checkpoint_sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["extra"]["model_name"] == row["model_name"]


def test_runner_retains_completed_work_and_records_the_failed_control(tmp_path):
    output = tmp_path / "interrupted"

    def fail_after_full(**kwargs):
        if kwargs["model_name"] == "gmf_no_memory":
            raise RuntimeError("injected development failure")
        return run_train(**kwargs)

    with patch("src.gmf.train.run_train", side_effect=fail_after_full):
        with pytest.raises(RuntimeError, match="injected development failure"):
            run_component_ablations(
                output, dim=4, memory_cells=5, n_pairs=1, seq_len=3,
                batch_size=2, steps=1, eval_batches=1,
            )
    manifest = json.loads((output / "component_ablations.json").read_text())
    assert manifest["status"] == "failed"
    assert [row["status"] for row in manifest["runs"]] == ["completed", "failed"]
    assert (output / manifest["runs"][0]["checkpoint"]).is_file()
    assert manifest["runs"][1]["ablation"] == "no_memory"
    assert "injected development failure" in manifest["runs"][1]["traceback"]
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        run_component_ablations(output, steps=1, eval_batches=1)
    assert before == {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}


@pytest.mark.parametrize("bad", [{"steps": 0}, {"eval_batches": -1}, {"seed": True}])
def test_runner_rejects_invalid_budget_before_creating_output(tmp_path, bad):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="configuration"):
        run_component_ablations(output, **bad)
    assert not output.exists()
