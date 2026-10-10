
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import benchmark_registry, BenchmarkConfig, batch_fingerprint, make_data_stream
from .losses import cosine_loss, mse_loss
from .model import GMFModel, MeanPoolMLP, GRUReasoner, TransformerReasoner
from .utils import load_checkpoint, save_json, ensure_dir, set_seed


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
def evaluate(model, bench_name: str, cfg: BenchmarkConfig, device="cpu", batch_size=64, batches=20, *, return_metadata=False):
    if type(batches) is not int or batches < 1:
        raise ValueError("batches must be a positive integer")
    if bench_name != cfg.name:
        raise ValueError("bench_name must match cfg.name")
    bench = benchmark_registry[bench_name]
    data_generator, data_stream = make_data_stream(cfg, "evaluation")
    model.eval()
    metrics = []
    for i in range(batches):
        seq, target = bench.generator(batch_size, cfg, generator=data_generator)
        batch_sha256 = batch_fingerprint(seq, target)
        seq = seq.to(device)
        target = target.to(device)
        out = model(seq)
        pred = out["prediction"]
        cos = F.cosine_similarity(pred, target, dim=-1)
        mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        metrics.append({
            "cosine": float(cos.mean().item()),
            "mse": float(mse.mean().item()),
            "uncertainty": float(out["uncertainty"].mean().item()) if "uncertainty" in out else None,
            "batch_sha256": batch_sha256,
        })
    uncertainties = [m["uncertainty"] for m in metrics if m["uncertainty"] is not None]
    agg = {
        "cosine_mean": float(np.mean([m["cosine"] for m in metrics])),
        "cosine_std": float(np.std([m["cosine"] for m in metrics])),
        "mse_mean": float(np.mean([m["mse"] for m in metrics])),
        "mse_std": float(np.std([m["mse"] for m in metrics])),
        "uncertainty_mean": float(np.mean(uncertainties)) if len(uncertainties) == len(metrics) else None,
    }
    data_stream["batches"] = len(metrics)
    data_stream["standard_deviation_unit"] = "batch means within one stream; not across seeds"
    if return_metadata:
        return agg, metrics, data_stream
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
    out_root = ensure_dir(out_dir)
    output_path = Path(out_root) / f"eval_{benchmark}_{model_name}_seed{seed}.json"
    if output_path.exists():
        raise FileExistsError(f"existing evaluation evidence: {output_path}")
    set_seed(seed)
    cfg = BenchmarkConfig(name=benchmark, dim=dim, seq_len=seq_len, n_pairs=n_pairs, distractors=distractors, noise=noise, seed=seed)
    model = build_model(model_name, dim=dim, memory_cells=memory_cells).to(device)
    checkpoint_sha256 = None
    if ckpt:
        checkpoint_sha256 = load_checkpoint(ckpt, model, None)["checkpoint_sha256"]
    agg, metrics, data_stream = evaluate(model, benchmark, cfg, device=device, batch_size=batch_size, batches=batches, return_metadata=True)
    result = {
        "benchmark": benchmark,
        "model": model_name,
        "data_stream": data_stream,
        "checkpoint_sha256": checkpoint_sha256,
        "model_state": "checkpoint" if ckpt else "untrained seeded initialization",
        "batch_size": batch_size,
        "agg": agg,
        "metrics": metrics,
    }
    save_json(output_path, result, overwrite=False)
    print(json.dumps(result["agg"], indent=2, allow_nan=False))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="associative_recall")
    p.add_argument("--model", dest="model_name", default="gmf")
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
