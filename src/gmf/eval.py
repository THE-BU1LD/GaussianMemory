
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import benchmark_registry, BenchmarkConfig
from .losses import cosine_loss, mse_loss
from .model import GMFModel, MeanPoolMLP, GRUReasoner, TransformerReasoner
from .utils import load_checkpoint, save_json, ensure_dir


def build_model(name: str, dim: int, memory_cells: int):
    name = name.lower()
    if name == "gmf":
        return GMFModel(dim=dim, memory_cells=memory_cells)
    if name == "mlp":
        return MeanPoolMLP(dim=dim)
    if name == "gru":
        return GRUReasoner(dim=dim)
    if name == "transformer":
        return TransformerReasoner(dim=dim)
    raise KeyError(name)


@torch.no_grad()
def evaluate(model, bench_name: str, cfg: BenchmarkConfig, device="cpu", batch_size=64, batches=20):
    bench = benchmark_registry[bench_name]
    model.eval()
    metrics = []
    for i in range(batches):
        seq, target = bench.generator(batch_size, cfg)
        seq = seq.to(device)
        target = target.to(device)
        out = model(seq)
        pred = out["prediction"]
        cos = F.cosine_similarity(pred, target, dim=-1)
        mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        metrics.append({
            "cosine": float(cos.mean().item()),
            "mse": float(mse.mean().item()),
            "uncertainty": float(out["uncertainty"].mean().item()) if "uncertainty" in out else float("nan"),
        })
    agg = {
        "cosine_mean": float(np.mean([m["cosine"] for m in metrics])),
        "cosine_std": float(np.std([m["cosine"] for m in metrics])),
        "mse_mean": float(np.mean([m["mse"] for m in metrics])),
        "mse_std": float(np.std([m["mse"] for m in metrics])),
        "uncertainty_mean": float(np.mean([m["uncertainty"] for m in metrics])),
    }
    return agg, metrics


def run_eval(
    benchmark: str = "associative_recall",
    model_name: str = "gmf",
    ckpt: str | None = None,
    dim: int = 64,
    memory_cells: int = 128,
    seq_len: int = 17,
    n_pairs: int = 8,
    distractors: int = 8,
    noise: float = 0.05,
    batch_size: int = 64,
    batches: int = 20,
    seed: int = 0,
    device: str = "cpu",
    out_dir: str = "outputs",
):
    cfg = BenchmarkConfig(name=benchmark, dim=dim, seq_len=seq_len, n_pairs=n_pairs, distractors=distractors, noise=noise, seed=seed)
    model = build_model(model_name, dim=dim, memory_cells=memory_cells).to(device)
    if ckpt:
        load_checkpoint(ckpt, model, None)
    agg, metrics = evaluate(model, benchmark, cfg, device=device, batch_size=batch_size, batches=batches)
    out_root = ensure_dir(out_dir)
    result = {
        "benchmark": benchmark,
        "model": model_name,
        "agg": agg,
        "metrics": metrics,
    }
    save_json(Path(out_root) / f"eval_{benchmark}_{model_name}.json", result)
    print(json.dumps(result["agg"], indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="associative_recall")
    p.add_argument("--model", default="gmf")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--memory_cells", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=17)
    p.add_argument("--n_pairs", type=int, default=8)
    p.add_argument("--distractors", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()
    run_eval(**vars(args))

if __name__ == "__main__":
    main()
