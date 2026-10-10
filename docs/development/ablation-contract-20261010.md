# Executable component ablations: development contract

## Defect and scope

On source `9f08ce93dd270216983f16767fa2e2ad47f10252`,
`src/gmf/ablations.py` returns a dictionary of five settings. None of its
flags reaches `GMFModel`; the model always uses dual memory, the learned
controller, and the configured refinement steps. The existing
`experiments/run_ablation.sh` compares four model families and does not run
component ablations. An inert plan cannot identify a component's contribution.

This development change implements the already declared five settings. It does
not change the default architecture, infer empirical benefits, or launch a
scientific sweep. The following choices are specified before testing outcomes.

## Exact forward contract

All settings instantiate the same modules in the same order. They retain the
same parameter names, initialization draws, and state-dictionary shapes. A
checkpoint loaded into a different setting is a deliberate intervention on the
same weights; callers must record the selected setting alongside that checkpoint.
The default `ablation="full"` retains the original computation exactly.

| Setting | Memory read and write | Fusion gates | Refinement |
| --- | --- | --- | --- |
| `full` | Existing 0.65 fast + 0.35 slow read; both writes | Existing learned three-way controller | Existing configured number of steps |
| `no_memory` | Zero read of the encoder output's shape; neither bank writes | Existing learned controller receives the zero read | Unchanged |
| `no_controller` | Unchanged dual memory | Fixed `(1/3, 1/3, 1/3)`, giving exactly `0.5*z + 0.5*memory` | Unchanged |
| `no_refine` | Unchanged dual memory | Unchanged learned controller | Zero applications of both `post` and `final_norm` |
| `single_memory` | The existing fast bank alone, with weight 1 and its unchanged cells, decay, bandwidth, and write rate; slow bank bypassed | Unchanged learned controller | Unchanged |

Field-state resets, bank resets at sequence boundaries, latent decay, predictor,
uncertainty output, output shapes, and existing state detachment rules are
unchanged. Disabled modules remain allocated for checkpoint compatibility, but
receive no gradients or writes through the disabled path. The setting is a
constructor choice; there is no mutation of global ablation defaults.

The no-memory controller remains free to learn how to scale the current latent.
The no-controller setting replaces adaptive gating with a fixed symmetric blend.
The single-memory setting removes the slow bank; it is **not** a matched active
memory-capacity or compute control. Equal allocated parameter counts do not mean
equal effective parameters, compute, or trainable paths. The unchanged sigmoid
uncertainty head is not calibrated by this change.

## Required verification

- The default and explicit full setting must give identical outputs and parameter
  gradients on identical weights and inputs.
- All settings must accept the same existing state dictionary strictly.
- No-memory must produce zero readouts, leave both banks zero, and bypass memory
  read/write calls while the encoder and controller remain active.
- No-controller must produce the prescribed fixed gates, bypass controller calls,
  and leave controller gradients absent while memory still executes.
- No-refinement must bypass both refinement modules and their gradients, while
  preserving the configured step count for an explicit full model.
- Single-memory must match the existing fast-bank read, update fast values, and
  bypass slow reads/writes and gradients.
- Each non-full setting must produce a discriminating change on a fixed small
  input, without claiming that the change improves a scientific endpoint.

## Research boundary

`RESEARCH_TRUTH.md` remains authoritative: there is no validated baseline
advantage or mechanistic contribution, and retained manuscript plots are
illustrative. Existing open data-stream and Gaussian-kernel repairs are separate
changes. A future experiment must freeze their integrated source, data streams,
matching controls, tuning budgets, seeds, and primary endpoint before evaluation.
This contract records engineering decisions and generated test fixtures only.

## Executable runner and verification receipt

`experiments/run_ablation.sh` now executes the five component settings through
`run_component_ablations`, using the actual training, checkpoint, and evaluation
functions. For example, the following is a deliberately tiny generated smoke
check, not a scientific sweep:

```bash
bash experiments/run_ablation.sh --out-dir outputs/component-smoke \
  --dim 4 --memory-cells 5 --seq-len 3 --n-pairs 1 \
  --batch-size 2 --steps 1 --eval-batches 1
```

The output directory must not exist. The runner records all settings before
execution, configuration, seven source-file hashes, environment, checkpoint
hashes, evaluations, completion status, and a traceback if a control fails.
Completed artifacts remain in place after an interrupted or failed matrix. This
is a serial development runner, not a resume mechanism or statistical protocol.
`gmf_full`, `gmf_no_memory`, `gmf_no_controller`, `gmf_no_refine`, and
`gmf_single_memory` are also accepted by the training and evaluation factories.

On 2026-10-10, Python 3.12 with isolated CPU Torch 2.14.1 ran all **31 tests** in
13.19 seconds, including 28 new cases and three existing tests. The actual shell
entrypoint trained, saved, and evaluated all five controls for one generated
step each. Another fixture deliberately failed after the first completed
control and verified its retained checkpoint and failure record. The train/eval
and output/gradient equivalence fixtures use bounded generated inputs only.
No validation of a research hypothesis or accuracy advantage was attempted.

The base revision still has the data-generation defects tracked in open PR #4
(repeated batches and temporal target leakage). Open PR #5 repairs retrieval
numerics. These are separate changes, and this runner does not convert the base
generator into a valid scientific evaluation protocol. Integrate and audit those
repairs before freezing a meaningful development or confirmatory experiment.
