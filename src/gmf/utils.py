
from __future__ import annotations
import json
import hashlib
import io
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: str | Path, payload: Any, *, overwrite: bool = True) -> None:
    encoded = json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
    with Path(path).open("w" if overwrite else "x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, extra: Dict[str, Any] | None = None) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
                    *, expected_model_name: str | None = None, allow_ablation_override: bool = False):
    # Hash and decode one snapshot while retaining pre-mutation control admission.
    source = Path(path).read_bytes()
    payload = torch.load(io.BytesIO(source), map_location="cpu", weights_only=True)
    if expected_model_name is not None:
        from .ablations import checkpoint_evaluation_identity

        checkpoint_evaluation_identity(
            payload.get("extra"), expected_model_name,
            allow_ablation_override=allow_ablation_override,
        )
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    payload["checkpoint_sha256"] = hashlib.sha256(source).hexdigest()
    return payload


def metric_summary(metrics: list[dict]) -> dict:
    out = {}
    if not metrics:
        return out
    keys = metrics[0].keys()
    for k in keys:
        vals = [float(m[k]) for m in metrics if k in m]
        if vals:
            out[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return out
