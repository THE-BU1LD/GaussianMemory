"""Bounded actual Torch steps must leave a source-bound terminal record."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.gmf import train


class FaultModel(torch.nn.Module):
    def __init__(self, error=None, fail_at=2):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.calls = 0
        self.error = error
        self.fail_at = fail_at

    def forward(self, sequence):
        self.calls += 1
        if self.error and self.calls == self.fail_at:
            raise self.error
        prediction = self.linear(sequence.mean(dim=1))
        if self.error == 'nonfinite':
            prediction = prediction * float('nan')
        return {'prediction': prediction}


def run(tmp_path, monkeypatch, model, steps=2):
    monkeypatch.setattr(train, 'build_model', lambda *args, **kwargs: model)
    return train.run_train(model_name='mlp', dim=8, n_pairs=2, seq_len=5,
                           batch_size=4, steps=steps, seed=83, out_dir=str(tmp_path))


@pytest.mark.parametrize('error', [RuntimeError('injected step failure'), KeyboardInterrupt()])
def test_partial_steps_survive_failure_and_interruption(tmp_path, monkeypatch, error):
    with pytest.raises(type(error)):
        run(tmp_path, monkeypatch, FaultModel(error))
    root = tmp_path / 'associative_recall_mlp_seed83'
    state = json.loads((root / 'run_status.json').read_text())
    history = json.loads((root / 'train_history.json').read_text())
    journal = [json.loads(line) for line in (root / 'train_history.jsonl').read_text().splitlines()]
    assert state['status'] == ('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
    assert state['completed_steps'] == 1
    assert state['attempted_step'] == 2
    assert len(state['last_batch_sha256']) == 64
    assert len(history) == 1 and journal == history
    assert not (root / 'checkpoint.pt').exists()


def test_nonfinite_objective_is_rejected_before_optimizer_step(tmp_path, monkeypatch):
    model = FaultModel('nonfinite', fail_at=99)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match='finite'):
        run(tmp_path, monkeypatch, model)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    root = tmp_path / 'associative_recall_mlp_seed83'
    state = json.loads((root / 'run_status.json').read_text())
    assert state['status'] == 'failed' and state['completed_steps'] == 0


def test_completed_run_retains_manifest_and_checkpoint_identity(tmp_path, monkeypatch):
    result = run(tmp_path, monkeypatch, FaultModel())
    root = Path(result['run_dir'])
    manifest = json.loads((root / 'run_manifest.json').read_text())
    state = json.loads((root / 'run_status.json').read_text())
    assert state['status'] == 'completed' and state['completed_steps'] == 2
    assert state['checkpoint_sha256'] == hashlib.sha256((root / 'checkpoint.pt').read_bytes()).hexdigest()
    assert manifest['configuration']['lr'] == 3e-4
    assert manifest['configuration']['steps'] == 2
    assert manifest['source_sha256']['train.py'] == hashlib.sha256(Path(train.__file__).read_bytes()).hexdigest()
    assert len((root / 'train_history.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('lr', [float('nan'), float('inf'), 0, -1])
def test_invalid_learning_rate_does_not_create_a_run(tmp_path, lr):
    with pytest.raises(ValueError):
        train.run_train(lr=lr, steps=1, dim=8, n_pairs=2, seq_len=5,
                        batch_size=4, out_dir=str(tmp_path))
    assert not list(tmp_path.iterdir())
