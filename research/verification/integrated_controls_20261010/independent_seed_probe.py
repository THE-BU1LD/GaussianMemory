"""Reproduce the pre-repair coherent seed-metadata admission; no real training."""
import importlib.util
import json
import pathlib
import tempfile
import time

import pytest

start = time.monotonic()
spec = importlib.util.spec_from_file_location(
    "pairing_cases", pathlib.Path("tests/test_integrated_pairing.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as directory:
    with pytest.MonkeyPatch.context() as patch:
        def mutate(phase, name, result):
            result["data_stream"]["generator_seed"] += 1

        module.fake_attempts(patch, mutate)
        module.run_fake(pathlib.Path(directory))
        result = json.loads(
            (pathlib.Path(directory) / "matrix/component_ablations.json").read_text()
        )
        cfg = module.data.BenchmarkConfig(
            name="temporal_prediction", dim=4, seq_len=3, n_pairs=1, seed=71
        )
        print(json.dumps({
            "status": result["status"],
            "pairing_verified": [row["pairing_verified"] for row in result["runs"]],
            "seeds": {
                split: {
                    "expected_v2_seed": module.data.make_data_stream(cfg, split)[1]["generator_seed"],
                    "admitted_seed": result["paired_streams"][split]["metadata"]["generator_seed"],
                }
                for split in ("train", "evaluation")
            },
            "training_steps_executed": 0,
            "fixture": "Explicit fake receipts from tests/test_integrated_pairing.py",
            "seconds_excluding_imports": round(time.monotonic() - start, 3),
        }, sort_keys=True))
