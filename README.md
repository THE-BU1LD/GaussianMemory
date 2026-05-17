
# Gaussian Memory Fields

End-to-end research repo for persistent latent memory reasoning.

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
