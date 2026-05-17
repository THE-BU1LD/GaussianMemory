
from __future__ import annotations
import itertools
import json
from pathlib import Path

from .train import run_train
from .eval import run_eval
from .utils import save_json


def run_suite(out_dir="outputs"):
    tasks = ["associative_recall", "needle_in_haystack", "temporal_prediction", "noisy_retrieval"]
    models = ["gmf", "gru", "transformer", "mlp"]
    summary = []
    for task, model in itertools.product(tasks, models):
        train_res = run_train(benchmark=task, model_name=model, steps=60, batch_size=24, dim=48, memory_cells=96, out_dir=out_dir)
        ckpt = Path(train_res["run_dir"]) / "checkpoint.pt"
        eval_res = run_eval(benchmark=task, model_name=model, ckpt=str(ckpt), batches=8, batch_size=48, dim=48, memory_cells=96, out_dir=out_dir)
        row = {
            "benchmark": task,
            "model": model,
            **eval_res["agg"],
        }
        summary.append(row)
        print(json.dumps(row))
    save_json(Path(out_dir) / "suite_summary.json", summary)
    return summary
