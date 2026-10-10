# Synthetic stream and entrypoint correction — 2026-10-10

## Defects and corrected behavior

Every original generator constructed a newly seeded `torch.Generator` for each
call, using only benchmark seed, batch size, sequence length, and a task offset.
Consequently, every training step with fixed arguments received the same batch.
Evaluation repeated one batch too; at equal train/evaluation batch size it could
repeat the training batch exactly. Repeated-batch variation is not an estimate
of variation across new evaluation samples.

The runners now instantiate one persistent CPU generator per call. Its seed is
derived from a stable SHA-256 identity containing data schema, benchmark name,
seed, and split (`train` or `evaluation`). Model/global RNG consumption does not
advance this generator. Within each call, successive batches consume new draws;
restarting the same split intentionally replays it. Different model families
with the same benchmark/configuration/seed/batch budget therefore receive
matched data. Split domain separation prevents accidental reuse of the same
random stream; it is not a mathematical guarantee that random values never
coincide. Direct two-argument generator calls remain deterministic single
fixtures; iterative callers must use the persistent generator:

```python
from src.gmf.data import BenchmarkConfig, benchmark_registry, make_data_stream

cfg = BenchmarkConfig(name="associative_recall", dim=16, n_pairs=4, seed=17)
generator, identity = make_data_stream(cfg, "train")
for step in range(3):
    inputs, target = benchmark_registry[cfg.name].generator(4, cfg, generator=generator)
```

The original temporal target was the final state already present in its input.
Under the explicit `gmf.synthetic.v2` contract, the input contains `seq_len`
states and the target is one subsequent nonlinear transition, including its
noise draw. The copy task retains its stated identity-copy semantics. Other
task equations and noise constants are unchanged. Future temporal scores must
identify v2 and cannot be interpreted as comparable measurements of the old
target definition.

Both advertised root entrypoints originally parsed `--model` into `model` and
then forwarded that unsupported keyword to a function expecting `model_name`.
They now forward the correct name, and regression tests actually train and
evaluate a tiny selected baseline through both scripts rather than stopping
at `--help`.

## Records and result retention

Training history and evaluation rows now include SHA-256 fingerprints of their
actual generated input/target tensors, shapes, and dtypes. Stream metadata
records the data schema, split, derived generator seed, resolved benchmark
configuration, number of batches, PyTorch version, and data source SHA-256.
Training persists that metadata beside the checkpoint and inside its metadata.
Evaluation identifies the exact serialized checkpoint bytes that were loaded;
untrained evaluations are seeded and explicitly labeled as untrained.

Baseline models have no uncertainty head, so their absent uncertainty is JSON
`null`. Nonfinite values cannot silently produce nonstandard JSON numbers.
Reported standard deviations describe batch means within one stream; they are
not across-seed uncertainty or confidence intervals.

Training refuses an existing named run directory. Evaluation filenames include
the seed, and exclusive creation refuses an existing result even if a file
appears after preflight. Use a fresh output directory to rerun a configuration;
do not delete evidence to obtain a clean run. Existing checked-in artifacts,
manuscript figures, and `RESEARCH_TRUTH.md` are unchanged.

## Bounded development execution

For an engineering smoke run only, using a new output directory:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python train.py --benchmark temporal_prediction --model mlp --dim 4 \
  --seq_len 3 --batch_size 2 --steps 2 --out_dir /tmp/gmf-development-new
python eval.py --benchmark temporal_prediction --model mlp --dim 4 \
  --seq_len 3 --batch_size 2 --batches 2 --out_dir /tmp/gmf-development-new \
  --ckpt /tmp/gmf-development-new/temporal_prediction_mlp_seed0/checkpoint.pt
```

The regression suite executes this bounded path using a temporary directory.
It also checks all five data generators for advancing batches, replay, split
separation and independence from global RNG; reconstructs the temporal target
from an independent recurrence; compares actual training draws across MLP/GRU;
checks evaluation/checkpoint identity and result retention; and rejects invalid
or empty evaluation requests. The existing GMF shape and train/eval tests remain
in the suite.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
```

Source compilation and patch whitespace checks passed locally. Local PyTorch
is unavailable in this development environment, so tensor/CLI test execution is
delegated to the existing exact-source hosted CI job; the PR records its actual
result after completion. CI runs the tests and existing evidence-boundary
sentinel with one numeric thread; no benchmark suite or research campaign is
launched. There is no performance finding, trained scientific artifact, or
claim of publication readiness from this development correction.

A future publication run still needs a frozen multi-seed protocol, matched
compute/parameter controls, complete source/environment identity, retained
per-seed raw outputs, and figures generated from those outputs. The limits and
unestablished claims in `RESEARCH_TRUTH.md` continue to apply.
