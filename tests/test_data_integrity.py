import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from src.gmf.data import (
    DATA_SCHEMA, BenchmarkConfig, batch_fingerprint, benchmark_registry,
    make_data_stream, temporal_prediction,
)
from src.gmf.eval import evaluate, run_eval
from src.gmf.model import MeanPoolMLP
from src.gmf.train import run_train
from src.gmf.utils import save_json


@pytest.mark.parametrize("name", list(benchmark_registry))
def test_streams_advance_replay_and_separate_training_from_evaluation(name):
    cfg = BenchmarkConfig(name=name, dim=4, seq_len=5, n_pairs=2, seed=19)
    first, meta = make_data_stream(cfg, "train")
    replay, replay_meta = make_data_stream(cfg, "train")
    held_out, held_meta = make_data_stream(cfg, "evaluation")
    generator = benchmark_registry[name].generator
    first_batches, replay_batches, held_batches = [], [], []
    for _ in range(3):
        first_batches.append(batch_fingerprint(*generator(3, cfg, generator=first)))
        # Model initialization/dropout/global draws cannot move the data stream.
        torch.randn(100)
        replay_batches.append(batch_fingerprint(*generator(3, cfg, generator=replay)))
        held_batches.append(batch_fingerprint(*generator(3, cfg, generator=held_out)))
    assert len(set(first_batches)) == 3
    assert len(set(held_batches)) == 3
    assert first_batches == replay_batches
    assert set(first_batches).isdisjoint(held_batches)
    assert meta == replay_meta
    assert meta["generator_seed"] != held_meta["generator_seed"]
    assert meta["schema"] == DATA_SCHEMA


@pytest.mark.parametrize("seq_len", [1, 4])
def test_temporal_target_is_the_next_unobserved_transition(seq_len):
    cfg = BenchmarkConfig(name="temporal_prediction", dim=2, seq_len=seq_len)
    generated = torch.Generator().manual_seed(37)
    expected = torch.Generator().manual_seed(37)
    matrix = torch.eye(2) * 0.92 + 0.02 * torch.randn(2, 2, generator=expected)
    state = torch.randn(3, 2, generator=expected)
    states = [state]
    for _ in range(seq_len):
        state = torch.tanh(state @ matrix) + 0.02 * torch.randn(3, 2, generator=expected)
        states.append(state)
    seq, target = temporal_prediction(3, cfg, generator=generated)
    torch.testing.assert_close(seq, torch.stack(states[:-1], dim=1), rtol=0, atol=0)
    torch.testing.assert_close(target, states[-1], rtol=0, atol=0)
    assert not torch.equal(target, seq[:, -1])
    assert target.data_ptr() != seq[:, -1].data_ptr()


@pytest.mark.parametrize("split", ["test", "validation", "", None])
def test_unbound_split_names_are_rejected(split):
    with pytest.raises(ValueError, match="split"):
        make_data_stream(BenchmarkConfig(name="copy_task"), split)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_invalid_stream_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        make_data_stream(BenchmarkConfig(name="copy_task", seed=seed), "train")


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_is_rejected(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        temporal_prediction(batch_size, BenchmarkConfig(name="temporal_prediction"))


@pytest.mark.parametrize("batches", [0, -1, True, 1.5])
def test_empty_or_invalid_evaluation_cannot_emit_statistics(batches):
    with pytest.raises(ValueError, match="batches"):
        evaluate(MeanPoolMLP(4), "copy_task", BenchmarkConfig(name="copy_task", dim=4), batches=batches)


def test_training_baselines_share_draws_and_evaluation_is_distinct(tmp_path):
    common = dict(benchmark="temporal_prediction", dim=4, memory_cells=8,
                  seq_len=3, steps=3, batch_size=2, seed=3, out_dir=str(tmp_path))
    mlp = run_train(model_name="mlp", **common)
    gru = run_train(model_name="gru", **common)
    train_hashes = [row["batch_sha256"] for row in mlp["history"]]
    assert len(set(train_hashes)) == 3
    assert train_hashes == [row["batch_sha256"] for row in gru["history"]]
    assert all(row["uncertainty"] is None for row in mlp["history"])
    checkpoint = Path(mlp["run_dir"]) / "checkpoint.pt"
    result = run_eval(benchmark="temporal_prediction", model_name="mlp", ckpt=str(checkpoint),
                      dim=4, memory_cells=8, seq_len=3, batch_size=2, batches=3,
                      seed=3, out_dir=str(tmp_path))
    evaluation_hashes = [row["batch_sha256"] for row in result["metrics"]]
    assert len(set(evaluation_hashes)) == 3
    assert set(train_hashes).isdisjoint(evaluation_hashes)
    assert result["data_stream"]["split"] == "evaluation"
    assert result["data_stream"]["schema"] == DATA_SCHEMA
    assert result["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = json.loads((tmp_path / "eval_temporal_prediction_mlp_seed3.json").read_text(),
                       parse_constant=lambda value: pytest.fail(f"nonstandard JSON: {value}"))
    assert saved == result
    assert saved["agg"]["uncertainty_mean"] is None
    before = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        run_train(model_name="mlp", **common)
    with pytest.raises(FileExistsError):
        run_eval(benchmark="temporal_prediction", model_name="mlp", seed=3, out_dir=str(tmp_path))
    assert checkpoint.read_bytes() == before


def test_repeated_evaluation_replays_matched_batches_without_global_rng_dependence():
    cfg = BenchmarkConfig(name="associative_recall", dim=4, n_pairs=2, seed=8)
    model = MeanPoolMLP(4)
    first = evaluate(model, cfg.name, cfg, batch_size=2, batches=3)
    torch.manual_seed(987)
    torch.randn(500)
    second = evaluate(model, cfg.name, cfg, batch_size=2, batches=3)
    assert first == second
    assert len({row["batch_sha256"] for row in first[1]}) == 3


def test_evaluation_seeds_keep_separate_files_and_seed_model_initialization(tmp_path):
    args = dict(benchmark="copy_task", model_name="mlp", dim=4, seq_len=3,
                batch_size=2, batches=2)
    first = run_eval(**args, seed=1, out_dir=str(tmp_path))
    second = run_eval(**args, seed=2, out_dir=str(tmp_path))
    replay = run_eval(**args, seed=1, out_dir=str(tmp_path / "replay"))
    assert first == replay
    assert first["metrics"] != second["metrics"]
    assert first["checkpoint_sha256"] is None
    assert first["model_state"] == "untrained seeded initialization"
    assert (tmp_path / "eval_copy_task_mlp_seed1.json").is_file()
    assert (tmp_path / "eval_copy_task_mlp_seed2.json").is_file()


def test_cli_really_trains_and_evaluates_selected_model(tmp_path):
    root = Path(__file__).resolve().parents[1]
    common = ["--benchmark", "temporal_prediction", "--model", "mlp", "--dim", "4",
              "--seq_len", "3", "--batch_size", "2", "--out_dir", str(tmp_path)]
    train = subprocess.run([sys.executable, "train.py", *common, "--steps", "2"],
                           cwd=root, capture_output=True, text=True, timeout=60)
    assert train.returncode == 0, train.stdout + train.stderr
    checkpoint = tmp_path / "temporal_prediction_mlp_seed0" / "checkpoint.pt"
    evaluation = subprocess.run([sys.executable, "eval.py", *common, "--batches", "2",
                                 "--ckpt", str(checkpoint)],
                                cwd=root, capture_output=True, text=True, timeout=60)
    assert evaluation.returncode == 0, evaluation.stdout + evaluation.stderr
    result = json.loads((tmp_path / "eval_temporal_prediction_mlp_seed0.json").read_text())
    assert result["model"] == "mlp"
    assert result["model_state"] == "checkpoint"
    assert len({row["batch_sha256"] for row in result["metrics"]}) == 2


def test_json_refuses_nonfinite_values_and_exclusive_writes(tmp_path):
    path = tmp_path / "evidence.json"
    with pytest.raises(ValueError):
        save_json(path, {"metric": float("nan")})
    assert not path.exists()
    save_json(path, {"metric": None}, overwrite=False)
    with pytest.raises(FileExistsError):
        save_json(path, {"metric": 1}, overwrite=False)
    assert json.loads(path.read_text()) == {"metric": None}
