
from __future__ import annotations
import argparse
import json
import hashlib
import math
import os
import platform
from pathlib import Path

import torch

from .data import benchmark_registry, BenchmarkConfig, batch_fingerprint, make_data_stream
from .ablations import build_ablation
from .losses import cosine_loss, mse_loss, vicreg_regularizer
from .model import GMFModel, MeanPoolMLP, GRUReasoner, TransformerReasoner
from .utils import set_seed, save_checkpoint, save_json, ensure_dir


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


def run_train(
    benchmark: str = "associative_recall",
    model_name: str = "gmf",
    dim: int = 64,
    memory_cells: int = 128,
    seq_len: int = 17,
    n_pairs: int = 8,
    distractors: int = 8,
    noise: float = 0.05,
    batch_size: int = 32,
    steps: int = 200,
    lr: float = 3e-4,
    seed: int = 0,
    device: str = "cpu",
    out_dir: str = "outputs",
    execution_mode: str = "serial-v1",
):
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    configuration = dict(benchmark=benchmark, model_name=model_name, dim=dim,
                         memory_cells=memory_cells, seq_len=seq_len, n_pairs=n_pairs,
                         distractors=distractors, noise=noise, batch_size=batch_size,
                         steps=steps, lr=lr, seed=seed, device=device, execution_mode=execution_mode)
    set_seed(seed)
    out_root = ensure_dir(out_dir)
    run_dir = Path(out_root) / f"{benchmark}_{model_name}_seed{seed}"
    run_dir.mkdir(exist_ok=False)
    history = []
    step, batch_sha256 = 0, None
    manifest = {"schema": "gmf.training-attempt.v1", "configuration": configuration,
                "implementation_revision": "gmf.integrated-development.v1",
                "python": platform.python_version(), "torch": str(torch.__version__),
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("train.py", "data.py", "model.py", "ablations.py", "losses.py", "utils.py")},
                "scope": "synthetic development execution; no model superiority claim"}
    try:
        save_json(run_dir / "run_manifest.json", manifest, overwrite=False)
        cfg = BenchmarkConfig(name=benchmark, dim=dim, seq_len=seq_len, n_pairs=n_pairs, distractors=distractors, noise=noise, seed=seed)
        data_generator, data_stream = make_data_stream(cfg, "train")
        model = build_model(model_name, dim=dim, memory_cells=memory_cells, execution_mode=execution_mode).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        bench = benchmark_registry[benchmark]
        with (run_dir / "train_history.jsonl").open("x", encoding="utf-8") as journal:
            for step in range(1, steps + 1):
                seq, target = bench.generator(batch_size, cfg, generator=data_generator)
                batch_sha256 = batch_fingerprint(seq, target)
                seq = seq.to(device)
                target = target.to(device)
                model.train()
                opt.zero_grad(set_to_none=True)
                out = model(seq)
                pred = out["prediction"]
                loss_main = cosine_loss(pred, target) + 0.25 * mse_loss(pred, target)
                loss_reg = vicreg_regularizer(pred)
                loss = loss_main + 0.05 * loss_reg
                if not bool(torch.isfinite(loss).all()):
                    raise ValueError("training objective must be finite before backward")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                if any(not bool(torch.isfinite(value).all()) for value in model.parameters()):
                    raise ValueError("model parameters must remain finite after optimization")
                with torch.no_grad():
                    cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
                    mse = mse_loss(pred, target).item()
                    unc = float(out["uncertainty"].mean().item()) if "uncertainty" in out else None
                rec = {"step": step, "loss": float(loss.item()), "cosine": cos, "mse": mse, "uncertainty": unc, "batch_sha256": batch_sha256}
                encoded = json.dumps(rec, allow_nan=False)
                journal.write(encoded + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                history.append(rec)
                if step % max(1, steps // 5) == 0 or step == 1:
                    print(encoded)
        save_json(run_dir / "train_history.json", history, overwrite=False)
        data_stream["batches"] = len(history)
        save_checkpoint(run_dir / "checkpoint.pt", model, opt, steps, extra={"benchmark": benchmark, "model_name": model_name, "cfg": cfg.__dict__, "memory_cells": memory_cells, "data_stream": data_stream, "execution_mode": execution_mode})
        save_json(run_dir / "data_stream.json", data_stream, overwrite=False)
        save_json(run_dir / "run_status.json", {"status": "completed", "completed_steps": len(history),
                  "attempted_step": step, "last_batch_sha256": batch_sha256,
                  "checkpoint_sha256": hashlib.sha256((run_dir / "checkpoint.pt").read_bytes()).hexdigest()}, overwrite=False)
    except BaseException as error:
        if not (run_dir / "train_history.json").exists():
            save_json(run_dir / "train_history.json", history, overwrite=False)
        save_json(run_dir / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                  "completed_steps": len(history), "attempted_step": step, "last_batch_sha256": batch_sha256,
                  "error_type": type(error).__name__, "error": str(error), "checkpoint_sha256": None}, overwrite=False)
        raise
    return {"run_dir": str(run_dir), "history": history, "data_stream": data_stream}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="associative_recall")
    p.add_argument("--model", dest="model_name", default="gmf")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--memory_cells", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=17)
    p.add_argument("--n_pairs", type=int, default=8)
    p.add_argument("--distractors", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--execution-mode", choices=("serial-v1", "batched-v1"), default="serial-v1")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()
    run_train(**vars(args))

if __name__ == "__main__":
    main()
