# Gaussian Memory Fields — Research Truth

This file defines the evidence boundary for the current repository state. It distinguishes implemented research infrastructure from empirical findings that have actually been regenerated from raw experiment outputs.

## Current verdict

**The repository contains a complete-looking experimental scaffold and manuscript, but the checked-in manuscript should not yet be treated as an empirical demonstration that Gaussian Memory Fields improve memory/reasoning performance.**

The paper itself states that it is structured to ingest JSON outputs after a complete run. The currently checked-in manuscript also contains hand-authored illustrative/reference plot coordinates. Those figures are useful as layout/diagnostic placeholders, but they are not evidence of executed benchmark performance unless replaced by generated artifacts traceable to raw runs.

## Verified repository observations

- The repository implements training, evaluation, benchmark, ablation, plotting, configuration, and test infrastructure.
- The paper describes controlled synthetic benchmarks for associative recall, needle retrieval, noisy retrieval, copy tasks, and temporal prediction.
- `experiments/` currently contains shell runners rather than a checked-in canonical raw-results package.
- The paper includes a hard-coded training/reference diagnostic curve whose caption says it is an expected reference diagnostic.
- The paper includes a hard-coded Gaussian memory heatmap explicitly captioned as illustrative.
- The paper includes a hard-coded cosine-score histogram captioned as a distribution for the benchmark suite. Until that plot is generated from raw evaluation outputs, it must be treated as illustrative rather than empirical.

## Claims currently safe

- GMF is an implemented persistent-memory architecture using Gaussian retrieval, neural-field dynamics, controller-gated fusion, recursive refinement, and uncertainty estimation.
- The repository provides benchmark generators, baseline model paths, ablation scripts, training/evaluation code, tests, and a manuscript scaffold.
- The current benchmarks are synthetic and controlled.

## Claims not currently established by checked-in evidence

Do not state these as validated findings until a complete raw-results package exists:

- GMF improves associative recall over GRU/Transformer/MLP baselines.
- Gaussian memory improves robustness to noise or longer contexts.
- Dual-scale memory outperforms a single bank.
- Controller gating or recursive refinement causally improves benchmark performance.
- The plotted reference curves/histograms are measurements from executed benchmark runs.
- GMF is superior to Transformers or other memory architectures in general.

## Required publication evidence

1. Freeze benchmark configs, model/control definitions, budgets, and a predeclared seed list.
2. Execute all paper-facing models and ablations under matched splits and budgets.
3. Retain per-seed raw outputs, resolved configs, checkpoint identity, commit SHA, and environment metadata.
4. Generate every empirical table and figure directly from raw run artifacts.
5. Remove or unmistakably mark illustrative figures that are not generated from experiment data.
6. Report mean/variation across multiple seeds for headline comparisons.
7. Include parameter-count or compute-matched comparisons where mechanism claims depend on architecture size.
8. Preserve baseline wins, null effects, failures, and unstable seeds.

## Integrity rule

A compiled PDF is not evidence that an experiment ran. Manuscript layout placeholders and illustrative plots must never be presented as measured results. If completed experiments are null or negative, the paper should report that outcome directly.
