# Independent sequence batching: development contract

Parent: `bdb3a82c600c316c424af586813aa9ef7240066d` (draft #7).
Implementation version: `gmf.independent-batch.v1`.

The current GMF forward loop invokes the complete encoder, retrieval, controller
and refinement stack separately for every sequence in a batch. The proposed
opt-in execution batches those operations across independent sequences while
keeping a distinct fast and slow memory state for each sequence. This is an
engineering-equivalent optimization of the existing recurrence, not a different
memory mechanism, scientific hypothesis, or additional scientific campaign.

Default `execution_mode="serial-v1"` preserves the retained implementation.
Explicit `execution_mode="batched-v1"` uses the independent states. The only
stochastic operation in the maintained GMF forward is predictor dropout. Predictor
calls remain one sequence at a time in the original order, after deterministic
batched recurrence, so dropout draw shapes, call order and subsequent RNG state
are preserved for successful calls on the checked runtime. Matrix batching can
change floating-point rounding; bitwise output/gradient identity is not claimed.
Parameter names, initialization, state-dictionary keys and checkpoint shapes stay
the same. Successfully completed calls publish the last sequence's field and bank
states, matching the serial API's observable state convention.

All five existing ablations, write-disabled execution, singleton and ordinary
batches must agree with an independent retained serial oracle on outputs, input
and parameter gradients, persistent state and subsequent RNG state. Per-sequence
memory must not mix across batch rows. The explicit factory and CLI options must
reach actual training and evaluation and record the selected mode in their
manifest/checkpoint/evaluation metadata; legacy checkpoints are labeled serial.

Bounds: at most eight local test commands, 600 seconds of total pytest time,
three independently reviewed repair cycles, two local generated train/eval
fixtures (at most two optimizer steps each), and one repeated timing profile
with fixed shapes and at most 20 repetitions per execution mode. Use one CPU
thread, no protected datasets, provider calls, paid compute or scientific trials.
Stop optional checking once equivalence, integration and a scoped profile pass.
Retain all failed checks and exact source identities. Preserve every existing
canonical-state field and append this continuation and verification before finish.

The existing UNVALIDATED_INFRASTRUCTURE disposition, illustrative paper figures,
component-control limitations, protected boundaries and outstanding scientific
protocol/evidence requirements remain authoritative.

## Executed verification and use

Full local suite: **182 passed in 39.36 seconds**, comprising 146 inherited tests
and 36 new cases. The exact parent model is retained under the verification
directory and SHA-256-checked before the comparison oracle loads it. A separate
reviewer passed 12 additional float64 configurations, including dimensions
1/3/7, zero/one/three field layers, zero refinement, noncontiguous inputs and
explicit checks that selected-row gradients are zero in other rows.

One fixed generated profile used B=16, T=17, D=16, 32 memory cells, two layers,
three refinement steps, float32, evaluation mode, no autograd and one CPU thread.
Seven timings per mode gave medians of 0.7753 seconds serial and 0.04705 seconds
batched (16.48 times for this fixture). Maximum prediction difference was
1.19e-7. Hardware, workload and system load affect this ratio. Independent bank
storage increases temporary memory approximately with batch size; no universal
throughput or memory-reduction claim is made.

An initial oracle-loading harness produced 31 setup failures and five passes
because its synthetic module omitted `__package__`; warnings-as-errors rejected
the relative-import fallback. That log remains retained. Setting the module
package fixed the harness without changing either model implementation. Existing
tiny training fixtures repeat inside the bounded pytest commands; they are not
new scientific trials.

Explicit API: `GMFModel(..., execution_mode="batched-v1")`. The train and evaluation
factories accept the same keyword for GMF and its five controls. The ordinary
root CLIs expose `--execution-mode batched-v1`; both default to `serial-v1`.
Training configuration/checkpoint metadata records the mode, and evaluation
records its selected mode and the trained mode (legacy absent metadata is serial).
For example, a deliberately bounded development fixture is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python train.py \
  --benchmark temporal_prediction --model gmf --dim 4 --memory_cells 5 \
  --seq_len 3 --n_pairs 1 --batch_size 2 --steps 2 --seed 127 \
  --execution-mode batched-v1 --out_dir outputs/new-batch-fixture
```

The independent review, source hashes, profile samples, failures and local test
records are in `research/verification/independent_batch_20261010/receipt.json`.
The default five-control matrix remains serial unless callers explicitly use
the train/evaluation APIs with the new mode. No study result or paper was rerun.
