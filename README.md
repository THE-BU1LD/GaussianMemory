# Gaussian Memory Fields

End-to-end research code for studying persistent latent memory with Gaussian retrieval and neural-field dynamics.

## Scientific status

This repository currently provides an experimental scaffold, not a validated claim that Gaussian Memory Fields outperform the included baselines. The manuscript contains illustrative/reference diagnostic plots that must be replaced by generated artifacts before they are treated as empirical results.

Read `RESEARCH_TRUTH.md` before quoting paper-facing findings.

## Core idea

Gaussian memory retrieval + neural field dynamics + future prediction + ablations + benchmark suite.

## Quickstart

```bash
pip install -r requirements.txt
python train.py --benchmark associative_recall --model gmf
python eval.py --benchmark associative_recall --model gmf --ckpt outputs/associative_recall_gmf_seed0/checkpoint.pt
```

## Suite

```bash
bash experiments/run_all.sh
```

For publication-facing evidence, freeze a multi-seed protocol and generate every empirical table/figure directly from retained raw outputs rather than from illustrative manuscript coordinates.
