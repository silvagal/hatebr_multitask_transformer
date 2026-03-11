from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model import MultiHeadModel, SingleTaskModel


def _collect_single_task_predictions(
    model: SingleTaskModel,
    dataloader: torch.utils.data.DataLoader,
    task: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if task == "target":
                labels = batch["labels_target"].cpu().numpy()
            else:
                label_name = "labels_bin" if task == "bin" else "labels_level"
                labels = batch[label_name].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels)

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if task == "target":
        probs = 1 / (1 + np.exp(-logits))
    else:
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
    return logits, labels, probs


def _collect_multitask_predictions(
    model: MultiHeadModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    all_bin_logits = []
    all_level_logits = []
    all_target_logits = []
    all_bin_labels = []
    all_level_labels = []
    all_target_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_bin_logits.append(outputs.bin_logits.detach().cpu().numpy())
            all_level_logits.append(outputs.level_logits.detach().cpu().numpy())
            all_target_logits.append(outputs.target_logits.detach().cpu().numpy())
            all_bin_labels.append(batch["labels_bin"].cpu().numpy())
            all_level_labels.append(batch["labels_level"].cpu().numpy())
            all_target_labels.append(batch["labels_target"].cpu().numpy())

    bin_logits = np.concatenate(all_bin_logits, axis=0)
    level_logits = np.concatenate(all_level_logits, axis=0)
    target_logits = np.concatenate(all_target_logits, axis=0)
    bin_labels = np.concatenate(all_bin_labels, axis=0)
    level_labels = np.concatenate(all_level_labels, axis=0)
    target_labels = np.concatenate(all_target_labels, axis=0)

    return {
        "bin_logits": bin_logits,
        "level_logits": level_logits,
        "target_logits": target_logits,
        "bin_labels": bin_labels,
        "level_labels": level_labels,
        "target_labels": target_labels,
    }


def _bin_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict:
    preds = np.argmax(probs, axis=1)
    positive_probs = probs[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision_pos": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_pos": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(labels, positive_probs))
    except ValueError:
        metrics["auc_roc"] = float("nan")
    return metrics


def _level_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict:
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    try:
        metrics["spearman"] = float(spearmanr(labels, preds).correlation)
    except Exception:
        metrics["spearman"] = float("nan")
    return metrics


def _select_target_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> float:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    pos_rate_labels = float(labels.mean()) if labels.size else 0.0
    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    best_delta = float("inf")

    for candidate in thresholds:
        preds = (probs >= candidate).astype(int)
        score = f1_score(labels, preds, average="micro", zero_division=0)
        pos_rate_preds = float(preds.mean()) if preds.size else 0.0
        delta = abs(pos_rate_preds - pos_rate_labels)
        if score > best_f1 or (np.isclose(score, best_f1) and delta < best_delta):
            best_f1 = score
            best_threshold = float(candidate)
            best_delta = delta

    return best_threshold


def _target_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    auto_tune_if_empty: bool = False,
) -> Dict:
    threshold_used = float(threshold)
    preds = (probs >= threshold_used).astype(int)
    tuned = False
    if auto_tune_if_empty and labels.sum() > 0 and preds.sum() == 0:
        threshold_used = _select_target_threshold(labels, probs)
        preds = (probs >= threshold_used).astype(int)
        tuned = True
    metrics = {
        "subset_accuracy": float(accuracy_score(labels, preds)),
        "hamming_loss": float(hamming_loss(labels, preds)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(labels, preds, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(labels, preds, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "pos_rate_labels": float(labels.mean()) if labels.size else 0.0,
        "pos_rate_preds": float(preds.mean()) if preds.size else 0.0,
        "threshold": threshold_used,
        "threshold_tuned": tuned,
    }
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    per_class_precision = precision_score(labels, preds, average=None, zero_division=0)
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
    support = labels.sum(axis=0)
    metrics["per_class_f1"] = per_class_f1.tolist()
    metrics["per_class_precision"] = per_class_precision.tolist()
    metrics["per_class_recall"] = per_class_recall.tolist()
    metrics["per_class_support"] = support.tolist()
    return metrics


def _consistency_metrics(
    bin_preds: np.ndarray,
    level_preds: np.ndarray,
    target_preds: np.ndarray,
    bin_labels: np.ndarray,
    level_labels: np.ndarray,
    target_labels: np.ndarray,
) -> Dict:
    pred_bin_zero = bin_preds == 0
    inconsistency_target = np.logical_and(pred_bin_zero, target_preds.sum(axis=1) > 0)
    inconsistency_level = np.logical_and(pred_bin_zero, level_preds > 0)
    target_exact = (target_preds == target_labels).all(axis=1)
    all_correct = (bin_preds == bin_labels) & (level_preds == level_labels) & target_exact

    return {
        "inconsistency_rate_target": float(np.mean(inconsistency_target)),
        "inconsistency_rate_level": float(np.mean(inconsistency_level)),
        "all_correct_rate": float(np.mean(all_correct)),
    }


def evaluate_single_task(
    model: SingleTaskModel,
    dataloader: torch.utils.data.DataLoader,
    task: str,
    device: torch.device,
    target_threshold: float,
    auto_tune_target: bool = False,
) -> Dict:
    logits, labels, probs = _collect_single_task_predictions(model, dataloader, task, device)
    if task == "bin":
        metrics = _bin_metrics(labels, probs)
        primary_metric = metrics["f1_pos"]
        return {
            "task": "bin",
            "metrics": metrics,
            "primary_metric": primary_metric,
            "raw": {"labels": labels, "probs": probs[:, 1]},
        }
    if task == "level":
        preds = np.argmax(probs, axis=1)
        metrics = _level_metrics(labels, preds)
        primary_metric = metrics["f1_macro"]
        return {
            "task": "level",
            "metrics": metrics,
            "primary_metric": primary_metric,
            "raw": {"labels": labels, "preds": preds},
        }
    if task == "target":
        metrics = _target_metrics(labels, probs, target_threshold, auto_tune_if_empty=auto_tune_target)
        primary_metric = metrics["micro_f1"]
        threshold_used = metrics.get("threshold", target_threshold)
        preds = (probs >= threshold_used).astype(int)
        return {
            "task": "target",
            "metrics": metrics,
            "primary_metric": primary_metric,
            "raw": {"labels": labels, "preds": preds},
        }
    raise ValueError(f"Unknown task: {task}")


def evaluate_multitask(
    model: MultiHeadModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    target_threshold: float,
    auto_tune_target: bool = False,
) -> Dict:
    outputs = _collect_multitask_predictions(model, dataloader, device)
    bin_logits = outputs["bin_logits"]
    level_logits = outputs["level_logits"]
    target_logits = outputs["target_logits"]

    bin_labels = outputs["bin_labels"]
    level_labels = outputs["level_labels"]
    target_labels = outputs["target_labels"]

    bin_probs = np.exp(bin_logits - bin_logits.max(axis=1, keepdims=True))
    bin_probs = bin_probs / bin_probs.sum(axis=1, keepdims=True)
    level_probs = np.exp(level_logits - level_logits.max(axis=1, keepdims=True))
    level_probs = level_probs / level_probs.sum(axis=1, keepdims=True)
    target_probs = 1 / (1 + np.exp(-target_logits))

    bin_metrics = _bin_metrics(bin_labels, bin_probs)
    level_preds = np.argmax(level_probs, axis=1)
    level_metrics = _level_metrics(level_labels, level_preds)
    target_metrics = _target_metrics(
        target_labels,
        target_probs,
        target_threshold,
        auto_tune_if_empty=auto_tune_target,
    )

    bin_preds = np.argmax(bin_probs, axis=1)
    threshold_used = target_metrics.get("threshold", target_threshold)
    target_preds = (target_probs >= threshold_used).astype(int)
    consistency = _consistency_metrics(
        bin_preds=bin_preds,
        level_preds=level_preds,
        target_preds=target_preds,
        bin_labels=bin_labels,
        level_labels=level_labels,
        target_labels=target_labels,
    )

    primary_metric = (bin_metrics["f1_pos"] + level_metrics["f1_macro"] + target_metrics["micro_f1"]) / 3
    return {
        "task": "multitask",
        "metrics": {
            "bin": bin_metrics,
            "level": level_metrics,
            "target": target_metrics,
            "consistency": consistency,
        },
        "primary_metric": primary_metric,
        "raw": {
            "bin_probs": bin_probs[:, 1],
            "bin_labels": bin_labels,
            "level_preds": level_preds,
            "level_labels": level_labels,
        },
    }


def compute_pr_curve(labels: np.ndarray, probs: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    return precision.tolist(), recall.tolist(), thresholds.tolist()
