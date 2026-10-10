
from __future__ import annotations
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import traceback


ABLATIONS = {
    "full": {"memory": True, "controller": True, "refine": True, "dual_memory": True},
    "no_memory": {"memory": False, "controller": True, "refine": True, "dual_memory": False},
    "no_controller": {"memory": True, "controller": False, "refine": True, "dual_memory": True},
    "no_refine": {"memory": True, "controller": True, "refine": False, "dual_memory": True},
    "single_memory": {"memory": True, "controller": True, "refine": True, "dual_memory": False},
}


def ablation_plan():
    return deepcopy(ABLATIONS)


def checkpoint_evaluation_identity(extra, requested_name, *, allow_ablation_override=False):
    """Bind evaluation to the trained component setting unless explicitly intervening.

    GMF component settings share every parameter name and shape, so strict
    state-dictionary loading alone cannot detect a changed scientific control.
    """
    if type(allow_ablation_override) is not bool:
        raise ValueError("allow_ablation_override must be a boolean")

    def canonical(name):
        if not isinstance(name, str):
            raise ValueError("checkpoint and evaluation require recorded model names")
        name = name.lower()
        if name == "gmf":
            name = "gmf_full"
        if name not in {"mlp", "gru", "transformer", *(f"gmf_{key}" for key in ABLATIONS)}:
            raise ValueError(f"unknown checkpoint/evaluation model: {name!r}")
        return name

    if not isinstance(extra, dict):
        raise ValueError("checkpoint lacks model identity metadata")
    trained = canonical(extra.get("model_name"))
    requested = canonical(requested_name)
    changed = trained != requested
    if changed and not (trained.startswith("gmf_") and requested.startswith("gmf_")):
        raise ValueError("checkpoint and evaluation model families differ")
    if changed and not allow_ablation_override:
        raise ValueError("checkpoint ablation differs; an intentional intervention requires allow_ablation_override=True")
    return {
        "schema": "gmf.checkpoint-evaluation.v1", "trained_model": trained,
        "evaluation_model": requested, "ablation_intervention": changed,
        "override_authorized": allow_ablation_override,
    }


def build_ablation(name: str = "full", **model_kwargs):
    """Instantiate one declared component intervention on the existing GMF.

    All modules remain allocated with unchanged checkpoint keys. Disabled paths
    do not execute; equal allocated parameters do not imply compute matching.
    """
    from .model import GMFModel

    return GMFModel(ablation=name, **model_kwargs)


def run_component_ablations(
    out_dir: str | Path,
    *,
    benchmark: str = "associative_recall",
    dim: int = 64,
    memory_cells: int = 128,
    seq_len: int = 17,
    n_pairs: int = 8,
    distractors: int = 8,
    noise: float = 0.05,
    batch_size: int = 32,
    steps: int = 200,
    eval_batches: int = 20,
    lr: float = 3e-4,
    seed: int = 0,
    device: str = "cpu",
):
    """Run the five declared controls and retain a manifested development record.

    This is an execution utility, not a frozen scientific protocol or a positive
    claim rule. Each setting starts with the same model-initialization seed.
    """
    import torch

    from .data import benchmark_registry
    from .eval import run_eval
    from .train import run_train

    positive_counts = (dim, memory_cells, seq_len, n_pairs, batch_size, steps, eval_batches)
    if (
        any(not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in positive_counts)
        or not isinstance(distractors, int) or isinstance(distractors, bool) or distractors < 0
        or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        or benchmark not in benchmark_registry
        or not math.isfinite(lr) or lr <= 0
        or not math.isfinite(noise) or noise < 0
    ):
        raise ValueError("invalid component-ablation execution configuration")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    config = dict(
        benchmark=benchmark, dim=dim, memory_cells=memory_cells, seq_len=seq_len,
        n_pairs=n_pairs, distractors=distractors, noise=noise, batch_size=batch_size,
        seed=seed, device=device,
    )
    source_root = Path(__file__).resolve().parent
    manifest = {
        "format": "gmf-component-ablations-v1",
        "status": "running",
        "scope": "development_only",
        "configuration": {**config, "steps": steps, "eval_batches": eval_batches, "lr": lr},
        "settings": ablation_plan(),
        "source_sha256": {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in ("ablations.py", "model.py", "data.py", "train.py", "eval.py", "losses.py", "utils.py")
        },
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "claim_boundary": (
            "Component execution only. No advancement rule, statistical confirmation, "
            "compute matching, calibrated uncertainty, or protected-data authorization. "
            "Audit and freeze the integrated data-generation protocol before a scientific run."
        ),
        "runs": [],
    }
    manifest_path = root / "component_ablations.json"

    def save_manifest():
        temporary = root / "component_ablations.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    save_manifest()
    for name in manifest["settings"]:
        model_name = f"gmf_{name}"
        row = {"ablation": name, "model_name": model_name, "status": "running"}
        manifest["runs"].append(row)
        save_manifest()
        try:
            trained = run_train(
                **config, model_name=model_name, steps=steps, lr=lr, out_dir=str(root)
            )
            checkpoint = Path(trained["run_dir"]) / "checkpoint.pt"
            row["checkpoint"] = str(checkpoint.relative_to(root))
            row["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            row["status"] = "trained"
            save_manifest()
            evaluation = run_eval(
                **config, model_name=model_name, ckpt=str(checkpoint),
                batches=eval_batches, out_dir=str(root),
            )
            row["evaluation"] = evaluation
            row["status"] = "completed"
            save_manifest()
        except BaseException as error:
            row["status"] = "failed"
            row["error_type"] = type(error).__name__
            row["error"] = str(error)
            row["traceback"] = traceback.format_exc()
            manifest["status"] = "failed"
            save_manifest()
            raise
    manifest["status"] = "completed"
    save_manifest()
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Run all five declared GMF component controls.")
    parser.add_argument("--out-dir", required=True, help="New directory; existing paths are refused.")
    parser.add_argument("--benchmark", default="associative_recall")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--memory-cells", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=17)
    parser.add_argument("--n-pairs", type=int, default=8)
    parser.add_argument("--distractors", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    run_component_ablations(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
