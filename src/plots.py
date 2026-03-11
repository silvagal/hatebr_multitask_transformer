from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_comparison_bar(output_path: str, metrics: Dict[str, Dict[str, float]]) -> None:
    labels = list(metrics.keys())
    f1_bin = [metrics[k]["f1_bin_pos"] for k in labels]
    f1_level = [metrics[k]["f1_level_macro"] for k in labels]
    f1_target = [metrics[k]["f1_target_micro"] for k in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, f1_bin, width, label="F1 bin pos")
    ax.bar(x, f1_level, width, label="F1 level macro")
    ax.bar(x + width, f1_target, width, label="F1 target micro")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Single-task vs Multitask")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix(output_path: str, confusion: List[List[int]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(confusion, cmap="Blues")
    ax.set_title("Confusion Matrix - Level")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(cax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pr_curve(output_path: str, curves: Dict[str, Dict[str, List[float]]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in curves.items():
        ax.plot(data["recall"], data["precision"], label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Binary Offensive)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_inconsistencies(output_path: str, data: Dict[str, Dict[str, float]]) -> None:
    labels = list(data.keys())
    target_vals = [data[k]["inconsistency_rate_target"] for k in labels]
    level_vals = [data[k]["inconsistency_rate_level"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, target_vals, width, label="Target")
    ax.bar(x + width / 2, level_vals, width, label="Level")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Rate")
    ax.set_title("Inconsistency Rates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_target_f1_bars(output_path: str, labels: List[str], f1_scores: List[float]) -> None:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, f1_scores, color="#2c7fb8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("F1")
    ax.set_title("Target Class F1 (Multitask)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
