
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import benchmark_registry, BenchmarkConfig, batch_fingerprint, make_data_stream
from .ablations import build_ablation, checkpoint_evaluation_identity
from .model import GMFModel, MeanPoolMLP, GRUReasoner, TransformerReasoner
from .utils import load_checkpoint, set_seed


def build_model(name: str, dim: int, memory_cells: int, *, execution_mode: str = "serial-v1"):
    name = name.lower()
    if execution_mode not in {"serial-v1", "batched-v1"}:
        raise ValueError("execution_mode must be serial-v1 or batched-v1")
    if execution_mode != "serial-v1" and not (name == "gmf" or name.startswith("gmf_")):
        raise ValueError("independent batch execution is only available for GMF")
    if name == "gmf":
        return GMFModel(dim=dim, memory_cells=memory_cells, execution_mode=execution_mode)
    if name.startswith("gmf_"):
        return build_ablation(name.removeprefix("gmf_"), dim=dim, memory_cells=memory_cells, execution_mode=execution_mode)
    if name == "mlp":
        return MeanPoolMLP(dim=dim)
    if name == "gru":
        return GRUReasoner(dim=dim)
    if name == "transformer":
        return TransformerReasoner(dim=dim)
    raise KeyError(name)


@torch.no_grad()
def evaluate(model, bench_name: str, cfg: BenchmarkConfig, device="cpu", batch_size=64, batches=20, *, return_metadata=False, record_event=None):
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
        if record_event is not None:
            record_event({"event": "started", "batch": i + 1, "batch_sha256": batch_sha256})
        seq = seq.to(device)
        target = target.to(device)
        if not target.is_floating_point() or target.ndim != 2 or target.shape[0] != batch_size or not torch.isfinite(target).all():
            raise ValueError("evaluation target must be a finite floating [batch, dimension] tensor")
        out = model(seq)
        pred = out["prediction"]
        if not isinstance(pred, torch.Tensor) or not pred.is_floating_point() or pred.shape != target.shape:
            raise ValueError("evaluation prediction must be a floating tensor with exactly the target shape")
        if not torch.isfinite(pred).all():
            raise ValueError("evaluation prediction must be finite")
        uncertainty = out.get("uncertainty")
        if uncertainty is not None and (
            not isinstance(uncertainty, torch.Tensor) or not uncertainty.is_floating_point()
            or uncertainty.shape not in ((batch_size,), (batch_size, 1))
            or not torch.isfinite(uncertainty).all()
        ):
            raise ValueError("evaluation uncertainty must contain one finite floating scalar per example")
        cos = F.cosine_similarity(pred, target, dim=-1)
        mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        rec = {
            "cosine": float(cos.mean().item()),
            "mse": float(mse.mean().item()),
            "uncertainty": float(uncertainty.mean().item()) if uncertainty is not None else None,
            "batch_sha256": batch_sha256,
        }
        if any(not math.isfinite(rec[name]) for name in ("cosine", "mse")) or (
            rec["uncertainty"] is not None and not math.isfinite(rec["uncertainty"])
        ):
            raise ValueError("evaluation metrics must remain finite")
        if record_event is not None:
            record_event({"event": "completed", "batch": i + 1,
                          "batch_sha256": batch_sha256, "metrics": rec})
        metrics.append(rec)
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


def _write_evaluation_json(path, payload):
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


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
    allow_ablation_override: bool = False,
    execution_mode: str = "serial-v1",
):
    out_root = Path(out_dir)
    output_path = Path(out_root) / f"eval_{benchmark}_{model_name}_seed{seed}.json"
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"existing evaluation evidence: {output_path}")
    out_root.mkdir(parents=True, exist_ok=True)
    attempt_dir = output_path.with_suffix(".attempt")
    attempt_dir.mkdir(exist_ok=False)
    progress = {"completed_batches": 0, "attempted_batch": 0, "last_batch_sha256": None}
    configuration = {
        "benchmark": benchmark, "model_name": model_name, "ckpt": str(ckpt) if ckpt else None,
        "dim": dim, "memory_cells": memory_cells, "seq_len": seq_len, "n_pairs": n_pairs,
        "distractors": distractors, "noise": noise, "batch_size": batch_size,
        "batches": batches, "seed": seed, "device": str(device),
        "allow_ablation_override": allow_ablation_override,
        "execution_mode": execution_mode,
    }
    try:
        _write_evaluation_json(attempt_dir / "manifest.json", {
            "schema": "gmf.evaluation-attempt.v1", "configuration": configuration,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("eval.py", "model.py", "data.py", "ablations.py", "utils.py")},
            "scope": "synthetic development evaluation; no scientific superiority claim",
        })
        set_seed(seed)
        cfg = BenchmarkConfig(name=benchmark, dim=dim, seq_len=seq_len, n_pairs=n_pairs, distractors=distractors, noise=noise, seed=seed)
        model = build_model(model_name, dim=dim, memory_cells=memory_cells, execution_mode=execution_mode).to(device)
        checkpoint_contract = None
        checkpoint_sha256 = None
        trained_execution_mode = None
        if ckpt:
            payload = load_checkpoint(
                ckpt, model, None, expected_model_name=model_name,
                allow_ablation_override=allow_ablation_override,
            )
            checkpoint_contract = checkpoint_evaluation_identity(
                payload.get("extra"), model_name,
                allow_ablation_override=allow_ablation_override,
            )
            checkpoint_sha256 = payload["checkpoint_sha256"]
            trained_execution_mode = payload["extra"].get("execution_mode", "serial-v1")
        _write_evaluation_json(attempt_dir / "model_identity.json", {
            "model_state": "checkpoint" if ckpt else "untrained seeded initialization",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_model_contract": checkpoint_contract,
            "execution_mode": execution_mode,
            "trained_execution_mode": trained_execution_mode,
        })
        with (attempt_dir / "batches.jsonl").open("x", encoding="utf-8") as journal:
            def record_event(event):
                journal.write(json.dumps(event, allow_nan=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                progress["attempted_batch"] = event["batch"]
                progress["last_batch_sha256"] = event["batch_sha256"]
                if event["event"] == "completed":
                    progress["completed_batches"] += 1

            agg, metrics, data_stream = evaluate(
                model, benchmark, cfg, device=device, batch_size=batch_size,
                batches=batches, return_metadata=True, record_event=record_event,
            )
        result = {
            "implementation_revision": "gmf.integrated-development.v1",
            "execution_mode": execution_mode,
            "trained_execution_mode": trained_execution_mode,
            "benchmark": benchmark, "model": model_name, "data_stream": data_stream,
            "checkpoint_sha256": checkpoint_sha256,
            "model_state": "checkpoint" if ckpt else "untrained seeded initialization",
            "batch_size": batch_size, "agg": agg, "metrics": metrics,
            "checkpoint_model_contract": checkpoint_contract,
            "evaluation_evidence": {"contract": "gmf.evaluation-evidence.v1",
                                    "attempt_directory": attempt_dir.name},
        }
        _write_evaluation_json(attempt_dir / "result.json", result)
        # Claim the final name atomically without replacing a concurrent result.
        os.link(attempt_dir / "result.json", output_path)
        _write_evaluation_json(attempt_dir / "status.json", {
            "status": "completed", **progress,
            "result_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        })
    except BaseException as error:
        _write_evaluation_json(attempt_dir / "status.json", {
            "status": "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            **progress, "error_type": type(error).__name__, "error": str(error),
        })
        raise
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
    p.add_argument("--execution-mode", choices=("serial-v1", "batched-v1"), default="serial-v1")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--allow-ablation-override", action="store_true",
                   help="Record an intentional within-GMF change from the trained control")
    args = p.parse_args()
    run_eval(**vars(args))

if __name__ == "__main__":
    main()
