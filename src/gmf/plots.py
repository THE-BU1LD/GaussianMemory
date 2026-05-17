
from __future__ import annotations
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


def plot_training(history_path: str | Path, out_path: str | Path):
    history = json.loads(Path(history_path).read_text())
    steps = [r["step"] for r in history]
    losses = [r["loss"] for r in history]
    cos = [r["cosine"] for r in history]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(steps, losses, label="loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(steps, cos, color="orange", label="cosine")
    ax2.set_ylabel("cosine")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_heatmap(weights: np.ndarray, out_path: str | Path, title: str = "Memory Heatmap"):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(weights, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_distribution(values: np.ndarray, out_path: str | Path, title: str = "Distribution"):
    fig = plt.figure(figsize=(6, 4))
    plt.hist(values, bins=40)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bar(labels, values, out_path: str | Path, title: str):
    fig = plt.figure(figsize=(7, 4))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
