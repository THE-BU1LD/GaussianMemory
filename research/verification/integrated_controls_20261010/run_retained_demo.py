"""Execute the single bounded development demonstration and retain its receipt.

Run this file from an immutable repository checkout in a fresh Python process.
The output directory must not already exist. This does not select a model or
authorize a scientific experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback


def main() -> None:
    evidence = Path(__file__).resolve().parent
    root = evidence.parents[2]
    output = evidence / "demo"
    receipt_path = evidence / "demo_execution.json"
    if output.exists() or receipt_path.exists():
        raise FileExistsError("A retained demo attempt already exists; do not overwrite it")
    sources = sorted((root / "src").rglob("*.py"))

    def source_hashes():
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        }

    command = [
        sys.executable, "-m", "src.gmf.ablations", "--out-dir", str(output),
        "--benchmark", "temporal_prediction", "--dim", "4",
        "--memory-cells", "5", "--seq-len", "3", "--n-pairs", "1",
        "--batch-size", "2", "--steps", "2", "--eval-batches", "2",
        "--seed", "20261010", "--device", "cpu",
    ]
    limits = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    environment = {**os.environ, **limits, "PYTHONPATH": str(root)}
    receipt = {
        "schema": "gmf.integrated-development-demo.v1",
        "scope": "generated development integration fixture; no efficacy comparison",
        "command": command,
        "cwd": str(root),
        "environment_overrides": {**limits, "PYTHONPATH": str(root)},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256_before": source_hashes(),
        "status": "running",
        "control_count": 5,
        "training_steps_per_control": 2,
        "total_optimizer_steps": 10,
        "protected_outcomes": 0,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    try:
        with (evidence / "demo_stdout.log").open("xb") as stdout, (evidence / "demo_stderr.log").open("xb") as stderr:
            result = subprocess.run(command, cwd=root, env=environment, stdout=stdout, stderr=stderr, timeout=60)
        receipt["exit_code"] = result.returncode
        if result.returncode != 0:
            raise RuntimeError(f"Demo process exited with {result.returncode}")
        manifest = json.loads((output / "component_ablations.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["implementation_revision"] == "gmf.integrated-development.v1"
        assert len(manifest["runs"]) == 5
        assert all(row["status"] == "completed" and row["pairing_verified"] for row in manifest["runs"])
        train_hashes = manifest["paired_streams"]["train"]["batch_sha256"]
        evaluation_hashes = manifest["paired_streams"]["evaluation"]["batch_sha256"]
        assert len(set(train_hashes)) == len(set(evaluation_hashes)) == 2
        assert not set(train_hashes) & set(evaluation_hashes)
        for row in manifest["runs"]:
            checkpoint = output / row["checkpoint"]
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["checkpoint_sha256"]
            assert row["evaluation"]["checkpoint_sha256"] == row["checkpoint_sha256"]
            assert row["training_batch_sha256"] == train_hashes
            assert [item["batch_sha256"] for item in row["evaluation"]["metrics"]] == evaluation_hashes
            terminal = json.loads((checkpoint.parent / "run_status.json").read_text())
            assert terminal["status"] == "completed"
        receipt["source_sha256_after"] = source_hashes()
        assert receipt["source_sha256_before"] == receipt["source_sha256_after"]
        receipt["verified"] = {
            "all_five_controls_completed": True,
            "actual_checkpoint_hashes_match_evaluation": True,
            "ordered_inputs_and_targets_paired_across_controls": True,
            "two_distinct_batches_per_split": True,
            "generated_training_and_evaluation_batch_hashes_disjoint": True,
            "all_training_terminal_records_completed": True,
            "source_files_unchanged_during_execution": True,
        }
        receipt["status"] = "completed"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)
        receipt["traceback"] = traceback.format_exc()
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["child_cpu_user_seconds"] = usage.ru_utime
        receipt["child_cpu_system_seconds"] = usage.ru_stime
        receipt["child_peak_rss_kib_linux"] = usage.ru_maxrss
        receipt["artifact_sha256"] = {
            str(path.relative_to(evidence)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*")) if path.is_file()
        } if output.exists() else {}
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({key: receipt[key] for key in ("status", "wall_seconds", "child_peak_rss_kib_linux")}, indent=2))


if __name__ == "__main__":
    main()
