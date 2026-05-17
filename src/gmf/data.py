
from __future__ import annotations
from dataclasses import dataclass
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


def _randn(shape, generator):
    return torch.randn(*shape, generator=generator)


def associative_recall(batch_size: int, cfg: BenchmarkConfig):
    g = torch.Generator().manual_seed(cfg.seed + batch_size + cfg.seq_len)
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


def needle_in_haystack(batch_size: int, cfg: BenchmarkConfig):
    g = torch.Generator().manual_seed(cfg.seed + 1000 + batch_size + cfg.seq_len)
    dim = cfg.dim
    seq_len = cfg.seq_len
    needle_pos = torch.randint(1, seq_len - 2, (batch_size,), generator=g)

    seq = _randn((batch_size, seq_len, dim), g) * 0.3
    needle = _randn((batch_size, dim), g)
    cue = needle + cfg.noise * _randn((batch_size, dim), g)

    for b in range(batch_size):
        seq[b, needle_pos[b]] = needle[b]
        seq[b, -1] = cue[b]
    target = needle
    return seq, target


def temporal_prediction(batch_size: int, cfg: BenchmarkConfig):
    g = torch.Generator().manual_seed(cfg.seed + 2000 + batch_size + cfg.seq_len)
    dim = cfg.dim
    seq_len = cfg.seq_len
    A = torch.eye(dim) * 0.92 + 0.02 * _randn((dim, dim), g)
    x = _randn((batch_size, dim), g)
    seq = [x]
    for _ in range(seq_len - 1):
        x = torch.tanh(x @ A) + 0.02 * _randn((batch_size, dim), g)
        seq.append(x)
    seq = torch.stack(seq, dim=1)
    target = seq[:, -1]
    return seq, target


def noisy_retrieval(batch_size: int, cfg: BenchmarkConfig):
    g = torch.Generator().manual_seed(cfg.seed + 3000 + batch_size + cfg.seq_len)
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


def copy_task(batch_size: int, cfg: BenchmarkConfig):
    g = torch.Generator().manual_seed(cfg.seed + 4000 + batch_size + cfg.seq_len)
    dim = cfg.dim
    seq_len = cfg.seq_len
    x = _randn((batch_size, seq_len, dim), g)
    target = x[:, -1].clone()
    return x, target


@dataclass
class Benchmark:
    name: str
    generator: Callable[[int, BenchmarkConfig], Tuple[torch.Tensor, torch.Tensor]]


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
