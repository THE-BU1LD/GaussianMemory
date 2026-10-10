
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch


@dataclass
class BenchmarkConfig:
    name: str
    dim: int = 64
    seq_len: int = 17
    n_pairs: int = 8
    distractors: int = 8
    noise: float = 0.05
    seed: int = 0


DATA_SCHEMA = "gmf.synthetic.v2"


def make_data_stream(cfg: BenchmarkConfig, split: str):
    """Create a reproducible CPU stream independent of model/global RNG state.

    Keep this generator for an entire train or evaluation call. Restarting it
    deliberately replays that split, so different models can use matched data.
    Split names domain-separate training draws from held-out evaluation draws.
    """
    if split not in {"train", "evaluation"}:
        raise ValueError("split must be 'train' or 'evaluation'")
    if type(cfg.seed) is not int or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if cfg.name not in benchmark_registry:
        raise ValueError(f"unknown benchmark: {cfg.name}")
    identity = json.dumps([DATA_SCHEMA, cfg.name, cfg.seed, split], separators=(",", ":"))
    generator_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % (2**63)
    metadata = {
        "schema": DATA_SCHEMA, "split": split, "generator_seed": generator_seed,
        "config": asdict(cfg), "generator_device": "cpu",
        "torch_version": str(torch.__version__),
        "data_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return torch.Generator().manual_seed(generator_seed), metadata


def batch_fingerprint(seq: torch.Tensor, target: torch.Tensor) -> str:
    """Bind a generated batch's actual values, shapes, and dtypes to its log row."""
    digest = hashlib.sha256()
    for tensor in (seq, target):
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True).encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _generator(batch_size, cfg, generator, offset):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    for name in ("dim", "seq_len", "n_pairs"):
        value = getattr(cfg, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    # Direct two-argument calls retain deterministic single-fixture semantics.
    # Training/evaluation always pass their persistent, split-specific stream.
    if generator is None:
        return torch.Generator().manual_seed(cfg.seed + offset + batch_size + cfg.seq_len)
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("generator must be a CPU torch.Generator")
    return generator


def _randn(shape, generator):
    return torch.randn(*shape, generator=generator)


def associative_recall(batch_size: int, cfg: BenchmarkConfig, *, generator=None):
    g = _generator(batch_size, cfg, generator, 0)
    dim = cfg.dim
    n = cfg.n_pairs

    keys = _randn((batch_size, n, dim), g)
    vals = _randn((batch_size, n, dim), g)
    q_idx = torch.randint(0, n, (batch_size,), generator=g)

    seq = []
    for i in range(n):
        seq.append(keys[:, i])
        seq.append(vals[:, i])

    query = torch.stack([keys[b, q_idx[b]] for b in range(batch_size)], dim=0)
    query = query + cfg.noise * _randn(query.shape, g)
    seq.append(query)
    seq = torch.stack(seq, dim=1)
    target = torch.stack([vals[b, q_idx[b]] for b in range(batch_size)], dim=0)
    return seq, target


def needle_in_haystack(batch_size: int, cfg: BenchmarkConfig, *, generator=None):
    g = _generator(batch_size, cfg, generator, 1000)
    dim = cfg.dim
    seq_len = cfg.seq_len
    if seq_len < 4:
        raise ValueError("needle_in_haystack requires seq_len >= 4")
    needle_pos = torch.randint(1, seq_len - 2, (batch_size,), generator=g)

    seq = _randn((batch_size, seq_len, dim), g) * 0.3
    needle = _randn((batch_size, dim), g)
    cue = needle + cfg.noise * _randn((batch_size, dim), g)

    for b in range(batch_size):
        seq[b, needle_pos[b]] = needle[b]
        seq[b, -1] = cue[b]
    target = needle
    return seq, target


def temporal_prediction(batch_size: int, cfg: BenchmarkConfig, *, generator=None):
    g = _generator(batch_size, cfg, generator, 2000)
    dim = cfg.dim
    seq_len = cfg.seq_len
    A = torch.eye(dim) * 0.92 + 0.02 * _randn((dim, dim), g)
    x = _randn((batch_size, dim), g)
    seq = [x]
    for _ in range(seq_len - 1):
        x = torch.tanh(x @ A) + 0.02 * _randn((batch_size, dim), g)
        seq.append(x)
    seq = torch.stack(seq, dim=1)
    # Predict one unobserved transition beyond the context. The old target was
    # seq[:, -1], allowing an identity readout to solve "prediction" perfectly.
    target = torch.tanh(x @ A) + 0.02 * _randn((batch_size, dim), g)
    return seq, target


def noisy_retrieval(batch_size: int, cfg: BenchmarkConfig, *, generator=None):
    g = _generator(batch_size, cfg, generator, 3000)
    dim = cfg.dim
    n = cfg.n_pairs
    keys = _randn((batch_size, n, dim), g)
    vals = _randn((batch_size, n, dim), g)
    q_idx = torch.randint(0, n, (batch_size,), generator=g)

    seq = []
    for i in range(n):
        key = keys[:, i] + 0.2 * _randn((batch_size, dim), g)
        val = vals[:, i] + 0.2 * _randn((batch_size, dim), g)
        seq.extend([key, val])
    query = torch.stack([keys[b, q_idx[b]] for b in range(batch_size)], dim=0)
    query = query + 0.15 * _randn(query.shape, g)
    seq.append(query)
    seq = torch.stack(seq, dim=1)
    target = torch.stack([vals[b, q_idx[b]] for b in range(batch_size)], dim=0)
    return seq, target


def copy_task(batch_size: int, cfg: BenchmarkConfig, *, generator=None):
    g = _generator(batch_size, cfg, generator, 4000)
    dim = cfg.dim
    seq_len = cfg.seq_len
    x = _randn((batch_size, seq_len, dim), g)
    target = x[:, -1].clone()
    return x, target


@dataclass
class Benchmark:
    name: str
    generator: Callable[..., Tuple[torch.Tensor, torch.Tensor]]


benchmark_registry: Dict[str, Benchmark] = {
    "associative_recall": Benchmark("associative_recall", associative_recall),
    "needle_in_haystack": Benchmark("needle_in_haystack", needle_in_haystack),
    "temporal_prediction": Benchmark("temporal_prediction", temporal_prediction),
    "noisy_retrieval": Benchmark("noisy_retrieval", noisy_retrieval),
    "copy_task": Benchmark("copy_task", copy_task),
}


def make_benchmark(name: str, **kwargs) -> BenchmarkConfig:
    if name not in benchmark_registry:
        raise KeyError(f"Unknown benchmark: {name}")
    return BenchmarkConfig(name=name, **kwargs)
