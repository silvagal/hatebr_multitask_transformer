from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.eval import evaluate_multitask, evaluate_single_task
from src.model import MultiHeadModel, SingleTaskModel, build_losses, get_single_task_num_labels

try:
    from torch.amp import autocast as amp_autocast
except ImportError:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import autocast as amp_autocast
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback if tqdm is missing
    def tqdm(iterable, **kwargs):
        return iterable


def _autocast_context(device: torch.device, scaler: Optional[GradScaler]):
    if not scaler:
        return nullcontext()
    try:
        return amp_autocast(device_type=device.type, enabled=True)
    except TypeError:
        return amp_autocast(enabled=True)


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    patience: int
    use_fp16: bool
    w_bin: float
    w_level: float
    w_target: float
    model_name: str


def _build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _build_scheduler(optimizer: AdamW, num_training_steps: int, warmup_ratio: float):
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def _train_epoch_single(
    model: SingleTaskModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    task: str,
    max_grad_norm: float,
    target_threshold: float,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    step_count = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for batch in progress:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if task == "target":
            labels = batch["labels_target"].to(device).float()
        else:
            label_name = "labels_bin" if task == "bin" else "labels_level"
            labels = batch[label_name].to(device)

        with _autocast_context(device, scaler):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        step_count += 1
        if task == "target":
            probs = torch.sigmoid(logits)
            preds = (probs >= target_threshold).int()
            correct = (preds == labels.int()).all(dim=1).sum().item()
            total_samples += labels.size(0)
        else:
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total_samples += labels.size(0)
        total_correct += correct
        avg_loss = total_loss / max(step_count, 1)
        train_acc = total_correct / max(total_samples, 1)
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{avg_loss:.4f}", train_acc=f"{train_acc:.4f}")
    avg_loss = total_loss / max(step_count, 1)
    train_acc = total_correct / max(total_samples, 1)
    return {"loss": avg_loss, "train_acc": train_acc}


def _train_epoch_multi(
    model: MultiHeadModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler,
    losses: Tuple[nn.Module, nn.Module, nn.Module],
    device: torch.device,
    scaler: Optional[GradScaler],
    max_grad_norm: float,
    weights: Tuple[float, float, float],
    target_threshold: float,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    loss_bin, loss_level, loss_target = losses
    w_bin, w_level, w_target = weights
    total_samples = 0
    bin_correct = 0
    level_correct = 0
    target_correct = 0
    step_count = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for batch in progress:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_bin = batch["labels_bin"].to(device)
        labels_level = batch["labels_level"].to(device)
        labels_target = batch["labels_target"].to(device).float()

        with _autocast_context(device, scaler):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = (
                w_bin * loss_bin(outputs.bin_logits, labels_bin)
                + w_level * loss_level(outputs.level_logits, labels_level)
                + w_target * loss_target(outputs.target_logits, labels_target)
            )

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        step_count += 1
        bin_preds = torch.argmax(outputs.bin_logits, dim=1)
        level_preds = torch.argmax(outputs.level_logits, dim=1)
        target_probs = torch.sigmoid(outputs.target_logits)
        target_preds = (target_probs >= target_threshold).int()
        bin_correct += (bin_preds == labels_bin).sum().item()
        level_correct += (level_preds == labels_level).sum().item()
        target_correct += (target_preds == labels_target.int()).all(dim=1).sum().item()
        total_samples += labels_bin.size(0)
        avg_loss = total_loss / max(step_count, 1)
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                loss=f"{avg_loss:.4f}",
                bin_acc=f"{bin_correct / max(total_samples, 1):.4f}",
                level_acc=f"{level_correct / max(total_samples, 1):.4f}",
                target_acc=f"{target_correct / max(total_samples, 1):.4f}",
            )
    avg_loss = total_loss / max(step_count, 1)
    return {
        "loss": avg_loss,
        "bin_acc": bin_correct / max(total_samples, 1),
        "level_acc": level_correct / max(total_samples, 1),
        "target_acc": target_correct / max(total_samples, 1),
    }


def train_single_task(
    dataloaders,
    task: str,
    config: TrainConfig,
    device: torch.device,
    checkpoint_path: str,
    target_threshold: float,
) -> Dict:
    num_labels = get_single_task_num_labels(task)
    model = SingleTaskModel(model_name=config.model_name, num_labels=num_labels).to(device)
    loss_bin, loss_level, loss_target = build_losses()
    loss_fn = loss_target if task == "target" else (loss_bin if task == "bin" else loss_level)

    optimizer = _build_optimizer(model, config.lr, config.weight_decay)
    total_steps = len(dataloaders.train) * config.epochs
    scheduler = _build_scheduler(optimizer, total_steps, config.warmup_ratio)
    scaler = GradScaler() if config.use_fp16 and device.type == "cuda" else None

    best_metric = -float("inf")
    best_state = None
    patience_count = 0
    metric_name = "f1_pos" if task == "bin" else ("f1_macro" if task == "level" else "micro_f1")
    best_threshold = target_threshold

    for epoch in range(1, config.epochs + 1):
        train_stats = _train_epoch_single(
            model,
            dataloaders.train,
            optimizer,
            scheduler,
            loss_fn,
            device,
            scaler,
            task,
            config.max_grad_norm,
            target_threshold,
            epoch,
            config.epochs,
        )
        metrics = evaluate_single_task(
            model,
            dataloaders.validation,
            task,
            device,
            target_threshold=target_threshold,
            auto_tune_target=task == "target",
        )
        if task == "target":
            val_acc = metrics["metrics"]["subset_accuracy"]
            current_threshold = metrics["metrics"].get("threshold", target_threshold)
            if metrics["metrics"].get("threshold_tuned"):
                print(
                    "Adjusted target threshold on validation "
                    f"from {target_threshold:.4f} to {metrics['metrics']['threshold']:.4f} "
                    "to avoid zero positive predictions."
                )
            if metrics["metrics"]["micro_f1"] == 0.0:
                if metrics["metrics"]["pos_rate_labels"] > 0.0 and metrics["metrics"]["pos_rate_preds"] == 0.0:
                    print(
                        "Warning: target micro_f1 is 0.0 on validation. "
                        f"pos_rate_labels={metrics['metrics']['pos_rate_labels']:.4f}, "
                        f"pos_rate_preds={metrics['metrics']['pos_rate_preds']:.4f}. "
                        "This usually indicates no positive targets predicted "
                        "or extremely imbalanced labels at the chosen threshold."
                    )
        else:
            val_acc = metrics["metrics"]["accuracy"]
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"loss {train_stats['loss']:.4f} | "
            f"train_acc {train_stats['train_acc']:.4f} | "
            f"val_acc {val_acc:.4f}"
        )
        metric = metrics["primary_metric"]
        if metric > best_metric:
            best_metric = metric
            best_state = {"model_state": model.state_dict(), "metric": metric, "epoch": epoch}
            patience_count = 0
            if task == "target":
                best_threshold = current_threshold
        else:
            patience_count += 1
            if patience_count >= config.patience:
                best_epoch = best_state["epoch"] if best_state else 0
                print(
                    f"Early stopping at epoch {epoch} after {patience_count} epochs without "
                    f"improvement on {metric_name}. Best {metric_name}={best_metric:.4f} "
                    f"at epoch {best_epoch}."
                )
                break

    if best_state is None:
        best_state = {"model_state": model.state_dict(), "metric": best_metric, "epoch": 0}

    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state["model_state"])
    eval_threshold = best_threshold if task == "target" else target_threshold
    test_metrics = evaluate_single_task(
        model,
        dataloaders.test,
        task,
        device,
        target_threshold=eval_threshold,
        auto_tune_target=False,
    )
    if task == "target" and abs(best_threshold - target_threshold) > 1e-6:
        test_metrics["metrics"]["threshold_tuned"] = True
    return {"model": model, "metrics": test_metrics}


def train_multitask(
    dataloaders,
    config: TrainConfig,
    device: torch.device,
    checkpoint_path: str,
    target_threshold: float,
) -> Dict:
    model = MultiHeadModel(model_name=config.model_name).to(device)
    losses = build_losses()

    optimizer = _build_optimizer(model, config.lr, config.weight_decay)
    total_steps = len(dataloaders.train) * config.epochs
    scheduler = _build_scheduler(optimizer, total_steps, config.warmup_ratio)
    scaler = GradScaler() if config.use_fp16 and device.type == "cuda" else None

    best_metric = -float("inf")
    best_state = None
    patience_count = 0
    metric_name = "primary_metric"
    best_threshold = target_threshold
    weights = (config.w_bin, config.w_level, config.w_target)

    for epoch in range(1, config.epochs + 1):
        train_stats = _train_epoch_multi(
            model,
            dataloaders.train,
            optimizer,
            scheduler,
            losses,
            device,
            scaler,
            config.max_grad_norm,
            weights,
            target_threshold,
            epoch,
            config.epochs,
        )
        metrics = evaluate_multitask(
            model,
            dataloaders.validation,
            device,
            target_threshold=target_threshold,
            auto_tune_target=True,
        )
        current_threshold = metrics["metrics"]["target"].get("threshold", target_threshold)
        val_bin_acc = metrics["metrics"]["bin"]["accuracy"]
        val_level_acc = metrics["metrics"]["level"]["accuracy"]
        val_target_acc = metrics["metrics"]["target"]["subset_accuracy"]
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"loss {train_stats['loss']:.4f} | "
            f"train_bin_acc {train_stats['bin_acc']:.4f} | "
            f"train_level_acc {train_stats['level_acc']:.4f} | "
            f"train_target_acc {train_stats['target_acc']:.4f} | "
            f"val_bin_acc {val_bin_acc:.4f} | "
            f"val_level_acc {val_level_acc:.4f} | "
            f"val_target_acc {val_target_acc:.4f}"
        )
        metric = metrics["primary_metric"]
        if metric > best_metric:
            best_metric = metric
            best_state = {"model_state": model.state_dict(), "metric": metric, "epoch": epoch}
            patience_count = 0
            best_threshold = current_threshold
        else:
            patience_count += 1
            if patience_count >= config.patience:
                best_epoch = best_state["epoch"] if best_state else 0
                print(
                    f"Early stopping at epoch {epoch} after {patience_count} epochs without "
                    f"improvement on {metric_name}. Best {metric_name}={best_metric:.4f} "
                    f"at epoch {best_epoch}."
                )
                break

    if best_state is None:
        best_state = {"model_state": model.state_dict(), "metric": best_metric, "epoch": 0}

    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state["model_state"])
    test_metrics = evaluate_multitask(
        model,
        dataloaders.test,
        device,
        target_threshold=best_threshold,
        auto_tune_target=False,
    )
    if abs(best_threshold - target_threshold) > 1e-6:
        test_metrics["metrics"]["target"]["threshold_tuned"] = True
    return {"model": model, "metrics": test_metrics}
