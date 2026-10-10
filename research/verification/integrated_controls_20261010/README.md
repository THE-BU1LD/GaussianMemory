# Integrated development controls: retained evidence

This directory records one bounded engineering continuation, version
`gmf.integrated-development.v1`. It is not a confirmatory experiment or a model
selection result. Read the prospective contract in
`docs/development/integrated-controls-contract-20261010.md` first.

## Inputs and implementation

The selected inputs are controls/checkpoint PR #6 at `5094bbb699eebd0cf85364bea759959fb35d6d33`,
synthetic-stream PR #4 at `0dcd2bd666e0575a5454cc121bb4a5a7bf3b3066`, and
Gaussian-kernel PR #5 at `35c5bd05252460297e7cb69b31dfffdb69eb23aa`.
`parent_research_state.json` preserves the exact pre-continuation canonical state.
The source commits retain their full histories. Merge receipts preserve the
three overlapping train/eval/checkpoint conflicts and the clean kernel import.
`imported_behavior_identity.json` checks 25 selected parent behaviors by exact
bytes or AST; the full suite also exercises their composition.

## Discriminating failures and repair

`pre_pairing_runner.py`, `baseline_tests.log` and `baseline_tests.xml` preserve a
reconstructed integration with the earlier PR #6 runner. All 20 injected bad
receipt cases failed the expected-rejection assertion: the earlier runner
accepted them. These are explicit mocked receipts, not historical scientific
results or actual trained checkpoints. The retained post-gate reordered-receipt
example has the same explicit fixture scope in `pairing_witnesses.json`.

The first combined candidate passed 144 tests. Independent review then supplied
`independent_seed_probe.py` and its exact log: consistently offsetting the seed
in every mocked control could evade agreement with the first control. Its
pre-repair source is retained as `pre_review_pairing.py`. The source hash was
reconstructed from that saved file, as stated in `independent_review.json`; it
was not emitted by the original witness. `review_repair_1.json` records the
observation, cause, repair and required evidence before the repair was tested.

The repaired runner derives the required seed independently from the declared
configuration and split, and checks the actual Torch version before establishing
the first reference. Two new regression cases cover common incorrect metadata.
The final complete suite passed **146 tests**, with warnings treated as errors,
in 39.04 pytest seconds / 45.734 command seconds. The reviewer read the final
repair and reported no remaining blocker. All three pytest attempts are indexed
in `verification_attempts.json`, with exact commands, outcomes and source hashes.

## One retained real execution

`run_retained_demo.py` is the executed wrapper. `demo_execution.json` retains its
exact command, environment overrides, driver and before/after source hashes,
exit status, timing, Linux child-process peak RSS and every output file hash.
The directory `demo/` contains the complete output of the five actual controls:
ten optimizer steps in total, their five checkpoints, training journals,
terminal records, stream metadata and evaluation receipts. The checkpoint bytes
and ordered input/target hashes were checked against the completed matrix.

All five controls completed with two different advancing batches in each split.
The two training batch hashes and two evaluation batch hashes were disjoint in
this fixture. Runtime was 12.253 seconds and child peak RSS was 317,272 KiB
(Linux high-water measurement including imports). These numbers describe this
tiny shared-environment execution, not a performance benchmark or full-run cost.
No ranking or hyperparameter decision was made from its metric values.

To reproduce the same fixture in a fresh checkout, use a new output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m src.gmf.ablations --out-dir outputs/integrated-control-reproduction \
  --benchmark temporal_prediction --dim 4 --memory-cells 5 --seq-len 3 \
  --n-pairs 1 --batch-size 2 --steps 2 --eval-batches 2 --seed 20261010
```

The retained wrapper deliberately refuses to overwrite this completed attempt.
Reproduction is a new run and must receive its own budget and fresh destination.

## Closure and limits

This continuation consumed three of six pytest commands, 86.301 of 600 test
command seconds, one of three review repair cycles, and one of two permitted
demonstration matrices. Optional testing stopped after the final suite,
independent review and completed retained matrix. No GPU, paid provider,
external dataset, protected outcome or scientific campaign was used.

`receipt.json` binds the final verification, implementation and evidence files.
These records establish engineering behavior under a fresh-process, unchanged
checkout precondition. They do not establish independent experimental units,
matched active parameter counts or compute, calibrated uncertainty, mechanism
efficacy, secure execution attestation, or scientific completion. The temporal
target and Gaussian normalization changes remain explicitly versioned; original
target-leaking measurements cannot be treated as comparable results.
