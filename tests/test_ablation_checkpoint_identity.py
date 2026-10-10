"""Checkpoint shape compatibility must not silently select another control."""
import json
from unittest.mock import patch

import pytest
import torch

from src.gmf.ablations import checkpoint_evaluation_identity
from src.gmf.eval import build_model, run_eval
from src.gmf.utils import load_checkpoint


def checkpoint(tmp_path, trained="gmf_no_memory"):
    model = build_model(trained, dim=4, memory_cells=5)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "extra": {"model_name": trained}}, path)
    return path


def test_accidental_ablation_change_fails_before_loading_or_evaluating(tmp_path):
    path = checkpoint(tmp_path)
    requested = build_model("gmf_full", dim=4, memory_cells=5)
    before = {key: value.clone() for key, value in requested.state_dict().items()}
    with pytest.raises(ValueError, match="ablation differs"):
        load_checkpoint(path, requested, expected_model_name="gmf_full")
    for key, value in requested.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with patch("src.gmf.eval.evaluate", side_effect=AssertionError("evaluation started")):
        with pytest.raises(ValueError, match="ablation differs"):
            run_eval(model_name="gmf_full", ckpt=path, dim=4, memory_cells=5, out_dir=tmp_path / "output")
    # Admission still blocks model-state loading and evaluation. The new
    # evaluation-evidence contract retains this rejected attempt explicitly.
    attempt = tmp_path / "output" / "eval_associative_recall_gmf_full_seed0.attempt"
    status = json.loads((attempt / "status.json").read_text())
    assert status["status"] == "failed" and status["completed_batches"] == 0
    assert not (tmp_path / "output" / "eval_associative_recall_gmf_full_seed0.json").exists()


@pytest.mark.parametrize("trained,requested", [("gmf", "gmf_full"), ("gmf_full", "GMF"), ("gmf_no_memory", "gmf_no_memory")])
def test_same_control_and_full_aliases_are_accepted(trained, requested):
    identity = checkpoint_evaluation_identity({"model_name": trained}, requested)
    assert not identity["ablation_intervention"]
    assert identity["trained_model"] == identity["evaluation_model"]


def test_explicit_intervention_is_executed_and_recorded(tmp_path):
    path = checkpoint(tmp_path)
    result = run_eval(model_name="gmf_full", ckpt=path, dim=4, memory_cells=5,
                      n_pairs=1, batch_size=2, batches=1,
                      out_dir=tmp_path / "evaluation", allow_ablation_override=True)
    contract = result["checkpoint_model_contract"]
    assert contract == {"schema": "gmf.checkpoint-evaluation.v1",
                        "trained_model": "gmf_no_memory", "evaluation_model": "gmf_full",
                        "ablation_intervention": True, "override_authorized": True}
    assert torch.isfinite(torch.tensor(result["agg"]["mse_mean"]))


@pytest.mark.parametrize("extra", [None, {}, {"model_name": "unknown"}, {"model_name": 3}])
def test_missing_or_unknown_identity_cannot_be_guessed(extra):
    with pytest.raises(ValueError):
        checkpoint_evaluation_identity(extra, "gmf")


def test_override_does_not_allow_a_different_model_family():
    with pytest.raises(ValueError, match="families differ"):
        checkpoint_evaluation_identity({"model_name": "mlp"}, "gmf_full", allow_ablation_override=True)
