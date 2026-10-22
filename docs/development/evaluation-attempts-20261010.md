# Evaluation attempt accounting: development contract

This bounded engineering continuation starts from exact integrated PR #7 head
`bdb3a82c600c316c424af586813aa9ef7240066d`. The previous closed integration
budget and all previous receipts, failures and research findings are preserved.
The new user instruction authorizes this separate implementation correction;
it does not authorize a protected scientific campaign.

## Failure and behavior

`run_eval` currently saves only after every batch completes. If a later model
call fails or is interrupted, the completed batch measurements disappear and
the same evaluation can be repeated in the same output location without any
record of the failed attempt. `evaluate` also admits broadcast-compatible
prediction shapes, permitting cross-example comparisons in place of aligned
predictions. These violate complete evidence and one-prediction-per-target
evaluation requirements.

The correction must reserve an exclusive per-evaluation attempt directory
before model/checkpoint work; save its configuration and execution-source
identities; flush a start event with each generated batch identity and a complete
event with each completed metric row; retain explicit completed, failed or
interrupted status; and reject reusing either an existing result or prior
attempt. Completed aggregate JSON is published atomically without replacement
at the established result filename. No partial cohort receives a completed
aggregate. JSON evidence remains readable after a catchable failure.

Predictions must be finite floating tensors with exactly the target shape.
Targets and scalar-per-example uncertainty must be finite. Metric values must
be finite before admission. Valid metrics retain their original equations,
batch-mean aggregation, normalization, RNG streams and standard-deviation unit.
Model architecture, losses, task definitions and checkpoints remain unchanged.
The prospective evidence contract is `gmf.evaluation-evidence.v1`.

## Verification budget and scientific boundary

At most two full-suite and four focused test commands, no more than 600 command
seconds, CPU and one numerical thread. Tests use constructed tensors and
injected failure models, plus the unchanged bounded training fixtures already
in the suite. No optional scientific matrix, external dataset, paid compute,
protected outcome or parameter search. Baseline failure, corrected checks,
source hashes and independently reviewed limits must be retained.

Batch fingerprints identify inputs; these journals retain batch statistics,
not full per-example predictions or an independent-unit statistical analysis.
The study remains `UNVALIDATED_INFRASTRUCTURE`. Semantic validity, model
superiority, calibrated uncertainty, and publication readiness are untested.
An uncatchable process failure may leave no terminal receipt; consumers must
require completed status instead of treating a nonterminal attempt as success.

## Verified outcome

All **158 combined offline tests passed in 38.55 seconds**, with warnings
treated as errors. This includes all 146 inherited cases and 12 new cases.
Critical Ruff and whitespace checks passed. Independent source-only review
found no scoped blocker; it did not rerun tests. Baseline failures, the
intermediate test-only import failure, exact source hashes, logs and both full
test runs are retained in `research/verification/evaluation_attempts_20261010/`.

The old checkpoint-admission test now explicitly expects a failed-attempt
receipt; it still verifies rejection before evaluation and unchanged model
weights. Final publication requires filesystem hard-link support and fails
without replacement when unavailable. Catchable-failure durability assumes a
writable filesystem; disk errors and arbitrary process death remain outside
that guarantee. No model, task, loss, seed stream or research result changed.
