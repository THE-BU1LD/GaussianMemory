# Checkpoint control identity before evaluation

## Defect and repair

All declared GMF controls allocate the same modules and therefore have compatible
state-dict keys. Previously, a checkpoint trained with `gmf_no_memory` loaded into
`gmf_full` and reached evaluation without an intervention record. The retained
constructed witness uses a mocked evaluator, so it contains no benchmark result.

`run_eval` now supplies its requested control to `load_checkpoint`, which verifies
`extra.model_name` before mutating model or optimizer state. Known `gmf`/`GMF`
aliases canonicalize to `gmf_full`. Missing/unknown identity fails closed. A
deliberate within-GMF intervention requires `allow_ablation_override=True` and
records both trained and evaluated controls under `gmf.checkpoint-evaluation.v1`.
This override never authorizes a different model family.

The public Python evaluation API provides the explicit intervention switch. PR #4
separately repairs the pre-existing CLI argument mapping and data streams; both
that change and PR #5 numerics still require integration before scientific runs.

## Evidence

Ten new cases check pre-load/pre-evaluation failure with unchanged model tensors,
identity aliases, missing metadata, family mismatch, and a real two-example,
one-batch generated evaluation with explicit intervention provenance. The entire
local suite passes 41 tests. This establishes checkpoint-control admission, not
component efficacy, fair active-parameter/compute matching, or research completion.

## Verification identity and reproduction

Exact source parent: `89bec9805a311946f05a3038d89a1a0ff46d2c43`. Revision: `gmf.checkpoint-evaluation.v1`.
The source/test SHA-256 map and test environment are retained in
`research/verification/checkpoint_identity_20261010/receipt.json`. Failed constructed witnesses and any JUnit files
remain in the same directory. These are engineering artifacts, not scientific
results or a research-completion checkpoint.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -o addopts='' -q
```
