# Run the integrated GMF development controls

This development version combines the reviewed synthetic-v2 stream, Gaussian
retrieval, component-ablation and checkpoint-identity changes. Its implementation
identifier is `gmf.integrated-development.v1`; the component manifest format is
`gmf-component-ablations-v2`.

## A bounded executable example

Use a new output directory and a fresh Python process in an unchanged checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m src.gmf.ablations --out-dir outputs/temporal-paired-development \
  --benchmark temporal_prediction --dim 4 --memory-cells 5 --seq-len 3 \
  --n-pairs 1 --batch-size 2 --steps 2 --eval-batches 2 --seed 20261010
```

This trains and evaluates `full`, `no_memory`, `no_controller`, `no_refine`, and
`single_memory` in that order, two training steps per control. It is an integration
fixture, not a scientific comparison or a model-selection procedure. The temporal
target is one transition beyond the observed sequence under `gmf.synthetic.v2`.
Historical target-leaking measurements cannot be compared as the same task.

Each control starts from the same initialization seed and consumes advancing
training and evaluation streams independent of model RNG activity. Before a
control can be marked complete, the runner verifies all of the following:

- Synthetic-v2 schema, split, complete batch count and resolved data configuration.
  The generator seed must independently match the declared schema/name/seed/split
  recipe, and the recorded Torch version must match the running interpreter.
- Source identity and complete ordered input/target fingerprints match the first
  control for the corresponding split.
- Training and evaluation generator domains differ.
- Evaluation identifies the same serialized checkpoint retained after training.
- Evaluation uses the trained component setting without an intervention override.

A mismatched control stops the matrix and retains its attempted record, error,
checkpoint if created, and every earlier completed control. A fresh output
directory is mandatory; existing attempts cannot be overwritten by this runner.

## Retained records

The root `component_ablations.json` records the configuration, environment, seven
source hashes, declared component switches, reference streams, and every attempted
control. Each successful training directory contains an fsynced step journal,
complete history, data-stream identity, checkpoint, source/configuration manifest,
and terminal status. Seed-qualified evaluation JSON contains the evaluated
checkpoint hash, trained/evaluated control identities, stream identity and batch
metrics. Absent baseline uncertainty is JSON null.

These are local provenance records, not cryptographic authentication of a trusted
producer. Input/target fingerprints identify the generated batches; they do not
replace raw prediction/data retention in a future scientific evidence package.
The fresh-process and immutable-checkout execution precondition still applies.

## Deliberate checkpoint interventions

Ordinary evaluation refuses a checkpoint trained under a different GMF control
before loading weights or starting evaluation. The corrected public CLI also
exposes the existing explicit intervention option:

```bash
python eval.py --benchmark temporal_prediction --model gmf_no_memory \
  --dim 4 --memory_cells 5 --seq_len 3 --n_pairs 1 --batch_size 2 --batches 1 \
  --seed 20261010 --out_dir outputs/explicit-intervention \
  --ckpt outputs/temporal-paired-development/temporal_prediction_gmf_full_seed20261010/checkpoint.pt \
  --allow-ablation-override
```

This records both the trained and evaluated controls. It is a deliberate new
intervention and must be declared in any study using it. The option does not
permit loading a different model family. The five-control matrix itself never
uses this override.

## Scientific limits and reproduction evidence

The five settings retain equal allocated modules and checkpoint shapes, but
their active parameters and compute differ. Pairing does not establish parameter
or compute matching, independent experimental units, calibrated uncertainty,
mechanistic benefit or superiority over simpler methods. Within-stream batch
standard deviations are not confidence intervals across independent runs.

The prospective integration contract, exact input commits, preserved parent state,
baseline/candidate test logs and retained demonstration are indexed under
`research/verification/integrated_controls_20261010/`. Existing paper figures and
scientific conclusions are unchanged. `RESEARCH_STATE.json` and
`RESEARCH_TRUTH.md` retain the outstanding scientific requirements.
