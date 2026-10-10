"""Discriminating admission cases for the integrated component runner.

The fake attempts below isolate the evidence gate; the existing shell-entrypoint
test executes all five real models with two actual advancing batches per split.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.gmf import data, eval as evaluation
from src.gmf.ablations import (
    ablation_plan, checkpoint_evaluation_identity, run_component_ablations,
)
from src.gmf.utils import load_checkpoint


def fake_attempts(monkeypatch, mutate=None):
    """Create explicit synthetic receipts, never disguised as real model runs."""
    cfg = data.BenchmarkConfig(name="temporal_prediction", dim=4, seq_len=3,
                               n_pairs=1, seed=71)

    def metadata(split):
        _, result = data.make_data_stream(cfg, split)
        result["batches"] = 2
        return result

    def rows(split):
        return [{"batch_sha256": hashlib.sha256(f"fake-{split}-{i}".encode()).hexdigest()}
                for i in range(2)]

    def train(**kwargs):
        root = Path(kwargs["out_dir"]) / kwargs["model_name"]
        root.mkdir()
        checkpoint = root / "checkpoint.pt"
        checkpoint.write_bytes(b"explicit synthetic checkpoint fixture")
        result = {"run_dir": str(root), "data_stream": metadata("train"),
                  "history": rows("train")}
        if mutate:
            mutate("train", kwargs["model_name"], result)
        return result

    def evaluate(**kwargs):
        name = kwargs["model_name"]
        result = {"model": name, "data_stream": metadata("evaluation"),
                  "metrics": rows("evaluation"),
                  "checkpoint_sha256": hashlib.sha256(Path(kwargs["ckpt"]).read_bytes()).hexdigest(),
                  "checkpoint_model_contract": checkpoint_evaluation_identity(
                      {"model_name": name}, name)}
        if mutate:
            mutate("evaluation", name, result)
        return result

    monkeypatch.setattr("src.gmf.train.run_train", train)
    monkeypatch.setattr("src.gmf.eval.run_eval", evaluate)


def run_fake(tmp_path):
    return run_component_ablations(
        tmp_path / "matrix", benchmark="temporal_prediction", dim=4,
        memory_cells=5, seq_len=3, n_pairs=1, batch_size=2, steps=2,
        eval_batches=2, seed=71,
    )


def test_complete_fake_receipts_are_admitted_and_pairing_is_explicit(tmp_path, monkeypatch):
    fake_attempts(monkeypatch)
    run_fake(tmp_path)
    result = json.loads((tmp_path / "matrix/component_ablations.json").read_text())
    assert result["status"] == "completed"
    assert result["format"] == "gmf-component-ablations-v2"
    assert len(result["runs"]) == len(ablation_plan()) == 5
    assert all(row["pairing_verified"] for row in result["runs"])
    assert result["paired_streams"]["train"]["metadata"]["config"] == asdict(
        data.BenchmarkConfig(name="temporal_prediction", dim=4, seq_len=3,
                             n_pairs=1, seed=71))


@pytest.mark.parametrize("split", ["train", "evaluation"])
@pytest.mark.parametrize("defect", ["reorder", "different_batch", "missing_batch",
                                    "schema", "source", "split", "count", "config"])
def test_mismatched_second_control_stops_with_completed_first_evidence(
    tmp_path, monkeypatch, split, defect,
):
    def mutate(phase, name, result):
        if phase != split or name != "gmf_no_memory":
            return
        sequence = result["history" if phase == "train" else "metrics"]
        if defect == "reorder":
            sequence.reverse()
        elif defect == "different_batch":
            sequence[-1]["batch_sha256"] = "f" * 64
        elif defect == "missing_batch":
            sequence.pop()
        elif defect == "source":
            result["data_stream"]["data_source_sha256"] = "0" * 64
        elif defect == "schema":
            result["data_stream"]["schema"] = "gmf.synthetic.v1"
        elif defect == "split":
            result["data_stream"]["split"] = "protected"
        elif defect == "count":
            result["data_stream"]["batches"] = 1
        else:
            result["data_stream"]["config"]["noise"] = 9.0

    fake_attempts(monkeypatch, mutate)
    with pytest.raises(ValueError):
        run_fake(tmp_path)
    root = tmp_path / "matrix"
    result = json.loads((root / "component_ablations.json").read_text())
    assert result["status"] == "failed"
    assert [row["status"] for row in result["runs"]] == ["completed", "failed"]
    assert result["runs"][0]["pairing_verified"] is True
    assert (root / result["runs"][0]["checkpoint"]).is_file()
    assert (root / result["runs"][1]["checkpoint"]).is_file()
    assert "ValueError" in result["runs"][1]["traceback"]
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        run_fake(tmp_path)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("defect", ["checkpoint", "override", "same_domain", "bad_hash"])
def test_a_self_consistent_first_control_still_requires_declared_identity(
    tmp_path, monkeypatch, defect,
):
    def mutate(phase, name, result):
        if phase != "evaluation":
            return
        if defect == "checkpoint":
            result["checkpoint_sha256"] = "0" * 64
        elif defect == "override":
            result["checkpoint_model_contract"]["override_authorized"] = True
        elif defect == "same_domain":
            cfg = data.BenchmarkConfig(name="temporal_prediction", dim=4, seq_len=3,
                                       n_pairs=1, seed=71)
            result["data_stream"]["generator_seed"] = data.make_data_stream(cfg, "train")[1]["generator_seed"]
        else:
            result["metrics"][0]["batch_sha256"] = "not-a-content-digest"

    fake_attempts(monkeypatch, mutate)
    with pytest.raises(ValueError):
        run_fake(tmp_path)
    result = json.loads((tmp_path / "matrix/component_ablations.json").read_text())
    assert result["status"] == "failed"
    assert len(result["runs"]) == 1
    assert result["runs"][0]["status"] == "failed"


def test_cli_exposes_the_explicit_control_intervention(monkeypatch):
    with patch.object(evaluation, "run_eval") as run:
        monkeypatch.setattr("sys.argv", ["eval.py", "--model", "gmf_full",
                                        "--allow-ablation-override"])
        evaluation.main()
    assert run.call_args.kwargs["model_name"] == "gmf_full"
    assert run.call_args.kwargs["allow_ablation_override"] is True


@pytest.mark.parametrize("defect", ["coherent_wrong_seed", "coherent_wrong_torch"])
def test_first_control_must_match_the_declared_seed_recipe_and_runtime(
    tmp_path, monkeypatch, defect,
):
    def mutate(phase, name, result):
        if defect == "coherent_wrong_seed":
            result["data_stream"]["generator_seed"] += 1
        else:
            result["data_stream"]["torch_version"] = "incorrect-runtime"

    fake_attempts(monkeypatch, mutate)
    with pytest.raises(ValueError, match="paired-stream metadata"):
        run_fake(tmp_path)
    result = json.loads((tmp_path / "matrix/component_ablations.json").read_text())
    assert result["status"] == "failed"
    assert [row["status"] for row in result["runs"]] == ["failed"]
    assert "pairing_verified" not in result["runs"][0]


def test_checkpoint_bytes_and_control_guard_survive_a_path_replacement(tmp_path, monkeypatch):
    model = evaluation.build_model("gmf_no_memory", dim=4, memory_cells=5)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "extra": {"model_name": "gmf_no_memory"}}, path)
    original_bytes = path.read_bytes()
    original_state = deepcopy(model.state_dict())
    read_bytes = Path.read_bytes

    def replace_after_read(candidate):
        value = read_bytes(candidate)
        if candidate == path:
            path.write_bytes(b"invalid replacement must not be decoded")
        return value

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    loaded = evaluation.build_model("gmf_no_memory", dim=4, memory_cells=5)
    result = load_checkpoint(path, loaded, expected_model_name="gmf_no_memory")
    assert result["checkpoint_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, original_state[name])
