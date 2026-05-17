
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from .data import benchmark_registry, BenchmarkConfig
from .losses import cosine_loss, mse_loss, vicreg_regularizer
from .model import GMFModel, MeanPoolMLP, GRUReasoner, TransformerReasoner
from .utils import set_seed, save_checkpoint, save_json, ensure_dir


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
):
    set_seed(seed)
    out_root = ensure_dir(out_dir)
    run_dir = ensure_dir(Path(out_root) / f"{benchmark}_{model_name}_seed{seed}")
    cfg = BenchmarkConfig(name=benchmark, dim=dim, seq_len=seq_len, n_pairs=n_pairs, distractors=distractors, noise=noise, seed=seed)

    model = build_model(model_name, dim=dim, memory_cells=memory_cells).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    bench = benchmark_registry[benchmark]
    history = []

    for step in range(1, steps + 1):
        seq, target = bench.generator(batch_size, cfg)
        seq = seq.to(device)
        target = target.to(device)

        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(seq) if model_name == "gmf" else model(seq)
        pred = out["prediction"] if "prediction" in out else out["prediction"]
        loss_main = cosine_loss(pred, target) + 0.25 * mse_loss(pred, target)
        loss_reg = vicreg_regularizer(pred)
        loss = loss_main + 0.05 * loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
            mse = mse_loss(pred, target).item()
            unc = float(out["uncertainty"].mean().item()) if "uncertainty" in out else float("nan")
        rec = {"step": step, "loss": float(loss.item()), "cosine": cos, "mse": mse, "uncertainty": unc}
        history.append(rec)
        if step % max(1, steps // 5) == 0 or step == 1:
            print(json.dumps(rec))

    save_json(run_dir / "train_history.json", history)
    save_checkpoint(run_dir / "checkpoint.pt", model, opt, steps, extra={"benchmark": benchmark, "model_name": model_name, "cfg": cfg.__dict__})
    return {"run_dir": str(run_dir), "history": history}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="associative_recall")
    p.add_argument("--model", default="gmf")
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
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()
    run_train(**vars(args))

if __name__ == "__main__":
    main()
