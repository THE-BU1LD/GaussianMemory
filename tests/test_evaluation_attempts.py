"""Constructed failures and alignment checks for development evaluation."""

import json
import pytest
import torch

from src.gmf.data import BenchmarkConfig
from src.gmf.eval import evaluate, run_eval


class FixtureModel(torch.nn.Module):
    def __init__(self, *, bad_shape=False, bad_value=None, fail_at=None, interrupt=False):
        super().__init__()
        self.bad_shape, self.bad_value = bad_shape, bad_value
        self.fail_at, self.interrupt, self.calls = fail_at, interrupt, 0

    def forward(self, seq):
        self.calls += 1
        if self.calls == self.fail_at:
            if self.interrupt:
                raise KeyboardInterrupt("constructed interruption")
            raise RuntimeError("constructed model failure")
        pred = seq[:, -1].clone()
        if self.bad_shape:
            pred = pred.unsqueeze(1)
        if self.bad_value is not None:
            pred[0, 0] = self.bad_value
        return {"prediction": pred}


def run(tmp_path, monkeypatch, model, **kwargs):
    monkeypatch.setattr("src.gmf.eval.build_model", lambda *args, **options: model)
    return run_eval(benchmark="copy_task", model_name="mlp", dim=4, seq_len=3,
                    batch_size=2, batches=3, seed=19, out_dir=tmp_path, **kwargs)


def attempt(tmp_path):
    return tmp_path / "eval_copy_task_mlp_seed19.attempt"


@pytest.mark.parametrize("interrupt", [False, True])
def test_failed_later_batch_retains_completed_evidence_and_prevents_reuse(tmp_path, monkeypatch, interrupt):
    model = FixtureModel(fail_at=2, interrupt=interrupt)
    with pytest.raises(KeyboardInterrupt if interrupt else RuntimeError):
        run(tmp_path, monkeypatch, model)
    root = attempt(tmp_path)
    events = [json.loads(line) for line in (root / "batches.jsonl").read_text().splitlines()]
    assert [(e["event"], e["batch"]) for e in events] == [
        ("started", 1), ("completed", 1), ("started", 2),
    ]
    assert events[1]["metrics"]["mse"] == 0.0
    assert events[0]["batch_sha256"] == events[1]["batch_sha256"]
    assert events[0]["batch_sha256"] != events[2]["batch_sha256"]
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == ("interrupted" if interrupt else "failed")
    assert status["completed_batches"] == 1 and status["attempted_batch"] == 2
    assert not (tmp_path / "eval_copy_task_mlp_seed19.json").exists()
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        run(tmp_path, monkeypatch, model)
    assert model.calls == 2
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_checkpoint_failure_is_retained_before_any_evaluation_batch(tmp_path, monkeypatch):
    model = FixtureModel()
    with pytest.raises(FileNotFoundError):
        run(tmp_path, monkeypatch, model, ckpt=str(tmp_path / "missing.pt"))
    root = attempt(tmp_path)
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["completed_batches"] == status["attempted_batch"] == model.calls == 0
    assert json.loads((root / "manifest.json").read_text())["configuration"]["batches"] == 3


def test_broadcast_prediction_is_rejected_before_cross_sample_metrics():
    with pytest.raises(ValueError, match="prediction.*shape"):
        evaluate(FixtureModel(bad_shape=True), "copy_task", BenchmarkConfig(name="copy_task", dim=4),
                 batch_size=2, batches=1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_prediction_is_rejected(value):
    with pytest.raises(ValueError, match="prediction.*finite"):
        evaluate(FixtureModel(bad_value=value), "copy_task", BenchmarkConfig(name="copy_task", dim=4),
                 batch_size=2, batches=1)


@pytest.mark.parametrize("uncertainty", [torch.zeros(2, 2), torch.tensor([[0.2], [float("nan")]])])
def test_uncertainty_must_be_one_finite_scalar_per_example(uncertainty):
    class Uncertain(FixtureModel):
        def forward(self, seq):
            return dict(super().forward(seq), uncertainty=uncertainty)
    with pytest.raises(ValueError, match="uncertainty"):
        evaluate(Uncertain(), "copy_task", BenchmarkConfig(name="copy_task", dim=4),
                 batch_size=2, batches=1)


def test_complete_attempt_publishes_existing_metrics_and_source_identity(tmp_path, monkeypatch):
    result = run(tmp_path, monkeypatch, FixtureModel())
    root = attempt(tmp_path)
    assert result["agg"]["mse_mean"] == result["agg"]["mse_std"] == 0.0
    assert result["agg"]["cosine_mean"] == pytest.approx(1.0, abs=1e-6)
    assert json.loads((root / "result.json").read_text()) == result
    assert json.loads((tmp_path / "eval_copy_task_mlp_seed19.json").read_text()) == result
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "completed" and status["completed_batches"] == 3
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema"] == "gmf.evaluation-attempt.v1"
    assert len(manifest["source_sha256"]["eval.py"]) == 64
    assert result["evaluation_evidence"]["contract"] == "gmf.evaluation-evidence.v1"


def test_concurrently_published_result_is_preserved(tmp_path, monkeypatch):
    class Concurrent(FixtureModel):
        def forward(self, seq):
            out = super().forward(seq)
            if self.calls == 1:
                (tmp_path / "eval_copy_task_mlp_seed19.json").write_text("other completed result\n")
            return out
    with pytest.raises(FileExistsError):
        run(tmp_path, monkeypatch, Concurrent())
    assert (tmp_path / "eval_copy_task_mlp_seed19.json").read_text() == "other completed result\n"
    assert json.loads((attempt(tmp_path) / "status.json").read_text())["status"] == "failed"


def test_dangling_result_symlink_blocks_model_work(tmp_path, monkeypatch):
    output = tmp_path / "eval_copy_task_mlp_seed19.json"
    output.symlink_to(tmp_path / "missing.json")
    model = FixtureModel()
    with pytest.raises(FileExistsError):
        run(tmp_path, monkeypatch, model)
    assert model.calls == 0 and output.is_symlink()
    assert not attempt(tmp_path).exists()
