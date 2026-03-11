import argparse
import errno
import multiprocessing.util as mp_util
import os
import re
import shutil
import sys
import tempfile
from typing import Dict, List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
TMP_ROOT = os.path.join(RESULTS_ROOT, "tmp")
os.makedirs(TMP_ROOT, exist_ok=True)
os.environ["TMPDIR"] = TMP_ROOT
os.environ["TEMP"] = TMP_ROOT
os.environ["TMP"] = TMP_ROOT

CACHE_ROOT = os.path.join(PROJECT_ROOT, ".hf_cache")
os.makedirs(CACHE_ROOT, exist_ok=True)
os.environ["HF_HOME"] = CACHE_ROOT
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_ROOT, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_ROOT, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_ROOT, "transformers")
tempfile.tempdir = TMP_ROOT


def _safe_remove_temp_dir(*args, **kwargs) -> None:
    def is_pathlike(value: object) -> bool:
        return isinstance(value, (str, bytes, os.PathLike))

    tempdir = args[0] if args else None
    onerror = kwargs.get("onerror")
    if len(args) > 1 and onerror is None:
        onerror = args[1]

    if not is_pathlike(tempdir) and is_pathlike(onerror):
        tempdir, onerror = onerror, tempdir

    if not is_pathlike(tempdir):
        return

    if not callable(onerror):
        onerror = None

    try:
        shutil.rmtree(tempdir, onerror=onerror)
    except OSError as exc:
        if exc.errno not in (errno.EBUSY, errno.ENOENT):
            raise


mp_util._remove_temp_dir = _safe_remove_temp_dir

import numpy as np
import torch

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import build_dataloaders, load_hatebr_dataset, tokenize_dataset
from src.eval import compute_pr_curve
from src.plots import (
    plot_comparison_bar,
    plot_confusion_matrix,
    plot_inconsistencies,
    plot_pr_curve,
    plot_target_f1_bars,
)
from src.train import TrainConfig, train_multitask, train_single_task
from src.utils import (
    build_output_paths,
    get_device,
    print_header,
    print_kv_table,
    print_metrics_table,
    save_json,
    save_text,
    seed_worker,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HateBR multitask experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--mask_urls_users", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w_bin", type=float, default=1.0)
    parser.add_argument("--w_level", type=float, default=1.0)
    parser.add_argument("--w_target", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--target_threshold", type=float, default=0.5)
    return parser.parse_args()


def _device_banner(device: torch.device) -> None:
    line = "=" * 50
    if device.type == "cuda":
        status = "DEVICE: CUDA (GPU)"
    else:
        status = "DEVICE: CPU"
    print(f"\n{line}\n{status}\n{line}")


def _write_final_results(seed: int, metrics: Dict, output_path: str) -> None:
    def fmt(value: float) -> str:
        try:
            if np.isnan(value):
                return "nan"
        except TypeError:
            pass
        return f"{value:.4f}"

    lines = [f"Seed: {seed}", "", "Resumo de métricas:"]
    if metrics.get("task") == "multitask":
        bin_metrics = metrics["metrics"]["bin"]
        level_metrics = metrics["metrics"]["level"]
        target_metrics = metrics["metrics"]["target"]
        consistency = metrics["metrics"]["consistency"]
        lines.extend(
            [
                f"bin_accuracy: {fmt(bin_metrics['accuracy'])}",
                f"bin_balanced_accuracy: {fmt(bin_metrics['balanced_accuracy'])}",
                f"bin_precision_pos: {fmt(bin_metrics['precision_pos'])}",
                f"bin_recall_pos: {fmt(bin_metrics['recall_pos'])}",
                f"bin_precision_macro: {fmt(bin_metrics['precision_macro'])}",
                f"bin_recall_macro: {fmt(bin_metrics['recall_macro'])}",
                f"bin_f1_pos: {fmt(bin_metrics['f1_pos'])}",
                f"bin_f1_macro: {fmt(bin_metrics['f1_macro'])}",
                f"bin_f1_weighted: {fmt(bin_metrics['f1_weighted'])}",
                f"bin_mcc: {fmt(bin_metrics['mcc'])}",
                f"bin_auc_roc: {fmt(bin_metrics['auc_roc'])}",
                f"level_accuracy: {fmt(level_metrics['accuracy'])}",
                f"level_balanced_accuracy: {fmt(level_metrics['balanced_accuracy'])}",
                f"level_precision_macro: {fmt(level_metrics['precision_macro'])}",
                f"level_recall_macro: {fmt(level_metrics['recall_macro'])}",
                f"level_f1_macro: {fmt(level_metrics['f1_macro'])}",
                f"level_f1_weighted: {fmt(level_metrics['f1_weighted'])}",
                f"level_spearman: {fmt(level_metrics['spearman'])}",
                f"target_subset_accuracy: {fmt(target_metrics['subset_accuracy'])}",
                f"target_hamming_loss: {fmt(target_metrics['hamming_loss'])}",
                f"target_precision_micro: {fmt(target_metrics['precision_micro'])}",
                f"target_recall_micro: {fmt(target_metrics['recall_micro'])}",
                f"target_precision_macro: {fmt(target_metrics['precision_macro'])}",
                f"target_recall_macro: {fmt(target_metrics['recall_macro'])}",
                f"target_micro_f1: {fmt(target_metrics['micro_f1'])}",
                f"target_macro_f1: {fmt(target_metrics['macro_f1'])}",
                f"target_pos_rate_labels: {fmt(target_metrics['pos_rate_labels'])}",
                f"target_pos_rate_preds: {fmt(target_metrics['pos_rate_preds'])}",
                f"target_threshold: {fmt(target_metrics.get('threshold', float('nan')))}",
                f"target_threshold_tuned: {target_metrics.get('threshold_tuned', False)}",
                f"inconsistency_target: {fmt(consistency['inconsistency_rate_target'])}",
                f"inconsistency_level: {fmt(consistency['inconsistency_rate_level'])}",
                f"all_correct_rate: {fmt(consistency['all_correct_rate'])}",
            ]
        )
    else:
        task = metrics["task"]
        if task == "bin":
            lines.extend(
                [
                    f"accuracy: {fmt(metrics['metrics']['accuracy'])}",
                    f"balanced_accuracy: {fmt(metrics['metrics']['balanced_accuracy'])}",
                    f"precision_pos: {fmt(metrics['metrics']['precision_pos'])}",
                    f"recall_pos: {fmt(metrics['metrics']['recall_pos'])}",
                    f"precision_macro: {fmt(metrics['metrics']['precision_macro'])}",
                    f"recall_macro: {fmt(metrics['metrics']['recall_macro'])}",
                    f"f1_pos: {fmt(metrics['metrics']['f1_pos'])}",
                    f"f1_macro: {fmt(metrics['metrics']['f1_macro'])}",
                    f"f1_weighted: {fmt(metrics['metrics']['f1_weighted'])}",
                    f"mcc: {fmt(metrics['metrics']['mcc'])}",
                    f"auc_roc: {fmt(metrics['metrics']['auc_roc'])}",
                ]
            )
        elif task == "level":
            lines.extend(
                [
                    f"accuracy: {fmt(metrics['metrics']['accuracy'])}",
                    f"balanced_accuracy: {fmt(metrics['metrics']['balanced_accuracy'])}",
                    f"precision_macro: {fmt(metrics['metrics']['precision_macro'])}",
                    f"recall_macro: {fmt(metrics['metrics']['recall_macro'])}",
                    f"f1_macro: {fmt(metrics['metrics']['f1_macro'])}",
                    f"f1_weighted: {fmt(metrics['metrics']['f1_weighted'])}",
                    f"spearman: {fmt(metrics['metrics']['spearman'])}",
                ]
            )
        elif task == "target":
            lines.extend(
                [
                    f"subset_accuracy: {fmt(metrics['metrics']['subset_accuracy'])}",
                    f"hamming_loss: {fmt(metrics['metrics']['hamming_loss'])}",
                    f"precision_micro: {fmt(metrics['metrics']['precision_micro'])}",
                    f"recall_micro: {fmt(metrics['metrics']['recall_micro'])}",
                    f"precision_macro: {fmt(metrics['metrics']['precision_macro'])}",
                    f"recall_macro: {fmt(metrics['metrics']['recall_macro'])}",
                f"micro_f1: {fmt(metrics['metrics']['micro_f1'])}",
                f"macro_f1: {fmt(metrics['metrics']['macro_f1'])}",
                f"pos_rate_labels: {fmt(metrics['metrics']['pos_rate_labels'])}",
                f"pos_rate_preds: {fmt(metrics['metrics']['pos_rate_preds'])}",
                f"threshold: {fmt(metrics['metrics'].get('threshold', float('nan')))}",
                f"threshold_tuned: {metrics['metrics'].get('threshold_tuned', False)}",
            ]
        )
    save_text(output_path, "\n".join(lines))


def _parse_seeds(raw_seeds: str | None, base_seed: int) -> List[int]:
    if raw_seeds:
        parts = [part for part in re.split(r"[,\s]+", raw_seeds.strip()) if part]
        seeds = [int(part) for part in parts]
    else:
        seeds = [base_seed, base_seed + 1]
    if len(seeds) != 2:
        raise ValueError("Provide exactly two distinct seeds (e.g. --seeds 42,43).")
    if seeds[0] == seeds[1]:
        raise ValueError("Seeds must be different.")
    return seeds


def _single_task_report(metrics: Dict) -> Dict[str, float]:
    task = metrics["task"]
    values = metrics["metrics"]
    if task == "bin":
        return {
            "accuracy": values["accuracy"],
            "balanced_accuracy": values["balanced_accuracy"],
            "precision_pos": values["precision_pos"],
            "recall_pos": values["recall_pos"],
            "precision_macro": values["precision_macro"],
            "recall_macro": values["recall_macro"],
            "f1_pos": values["f1_pos"],
            "f1_macro": values["f1_macro"],
            "f1_weighted": values["f1_weighted"],
            "mcc": values["mcc"],
            "auc_roc": values["auc_roc"],
        }
    if task == "level":
        return {
            "accuracy": values["accuracy"],
            "balanced_accuracy": values["balanced_accuracy"],
            "precision_macro": values["precision_macro"],
            "recall_macro": values["recall_macro"],
            "f1_macro": values["f1_macro"],
            "f1_weighted": values["f1_weighted"],
            "spearman": values["spearman"],
        }
    if task == "target":
        return {
            "subset_accuracy": values["subset_accuracy"],
            "hamming_loss": values["hamming_loss"],
            "precision_micro": values["precision_micro"],
            "recall_micro": values["recall_micro"],
            "precision_macro": values["precision_macro"],
            "recall_macro": values["recall_macro"],
            "micro_f1": values["micro_f1"],
            "macro_f1": values["macro_f1"],
            "pos_rate_labels": values["pos_rate_labels"],
            "pos_rate_preds": values["pos_rate_preds"],
        }
    raise ValueError(f"Unknown task: {task}")


def _multitask_report(metrics: Dict) -> Dict[str, float]:
    bin_metrics = metrics["metrics"]["bin"]
    level_metrics = metrics["metrics"]["level"]
    target_metrics = metrics["metrics"]["target"]
    consistency = metrics["metrics"]["consistency"]
    report = {
        "bin_accuracy": bin_metrics["accuracy"],
        "bin_balanced_accuracy": bin_metrics["balanced_accuracy"],
        "bin_precision_pos": bin_metrics["precision_pos"],
        "bin_recall_pos": bin_metrics["recall_pos"],
        "bin_precision_macro": bin_metrics["precision_macro"],
        "bin_recall_macro": bin_metrics["recall_macro"],
        "bin_f1_pos": bin_metrics["f1_pos"],
        "bin_f1_macro": bin_metrics["f1_macro"],
        "bin_f1_weighted": bin_metrics["f1_weighted"],
        "bin_mcc": bin_metrics["mcc"],
        "bin_auc_roc": bin_metrics["auc_roc"],
        "level_accuracy": level_metrics["accuracy"],
        "level_balanced_accuracy": level_metrics["balanced_accuracy"],
        "level_precision_macro": level_metrics["precision_macro"],
        "level_recall_macro": level_metrics["recall_macro"],
        "level_f1_macro": level_metrics["f1_macro"],
        "level_f1_weighted": level_metrics["f1_weighted"],
        "level_spearman": level_metrics["spearman"],
        "target_subset_accuracy": target_metrics["subset_accuracy"],
        "target_hamming_loss": target_metrics["hamming_loss"],
        "target_precision_micro": target_metrics["precision_micro"],
        "target_recall_micro": target_metrics["recall_micro"],
        "target_precision_macro": target_metrics["precision_macro"],
        "target_recall_macro": target_metrics["recall_macro"],
        "target_micro_f1": target_metrics["micro_f1"],
        "target_macro_f1": target_metrics["macro_f1"],
        "target_pos_rate_labels": target_metrics["pos_rate_labels"],
        "target_pos_rate_preds": target_metrics["pos_rate_preds"],
        "inconsistency_target": consistency["inconsistency_rate_target"],
        "inconsistency_level": consistency["inconsistency_rate_level"],
        "all_correct_rate": consistency["all_correct_rate"],
        "primary_metric": metrics["primary_metric"],
    }
    return report


def _mean_std(values: List[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return mean, std


def _format_mean_std(mean: float, std: float) -> str:
    if np.isnan(mean) or np.isnan(std):
        return "nan"
    return f"{mean:.4f} ± {std:.4f}"


def _write_summary_report(
    output_path: str,
    seeds: List[int],
    results_by_seed: Dict[int, Dict],
    label_names: List[str],
) -> None:
    lines = [
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "Resumo agregado (média ± desvio padrão):",
    ]

    report_by_seed: Dict[int, Dict[str, Dict[str, float]]] = {}
    for seed, results in results_by_seed.items():
        report_by_seed[seed] = {
            "singletask_offensive": _single_task_report(results["singletask_offensive"]["metrics"]),
            "singletask_level": _single_task_report(results["singletask_level"]["metrics"]),
            "singletask_target": _single_task_report(results["singletask_target"]["metrics"]),
            "multitask": _multitask_report(results["multitask"]["metrics"]),
        }

    for experiment_name in (
        "singletask_offensive",
        "singletask_level",
        "singletask_target",
        "multitask",
    ):
        lines.append(f"\n[{experiment_name}]")
        keys = list(report_by_seed[seeds[0]][experiment_name].keys())
        for key in keys:
            values = [report_by_seed[seed][experiment_name][key] for seed in seeds]
            mean, std = _mean_std(values)
            lines.append(f"{key}: {_format_mean_std(mean, std)}")

    lines.append("\nPer-seed metrics:")
    for seed in seeds:
        lines.append(f"\n[seed {seed}]")
        for experiment_name, metrics in report_by_seed[seed].items():
            lines.append(f"{experiment_name}:")
            for key, value in metrics.items():
                lines.append(f"{key}: {value:.4f}")

    for experiment_name, task_key in (
        ("singletask_target", "singletask_target"),
        ("multitask", "multitask"),
    ):
        per_class_values = []
        for seed in seeds:
            if task_key == "singletask_target":
                metrics = results_by_seed[seed][task_key]["metrics"]["metrics"]["per_class_f1"]
            else:
                metrics = results_by_seed[seed][task_key]["metrics"]["metrics"]["target"]["per_class_f1"]
            per_class_values.append(metrics)
        per_class_array = np.asarray(per_class_values, dtype=float)
        means = per_class_array.mean(axis=0)
        stds = per_class_array.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(means)
        lines.append(f"\n[{experiment_name} per_class_f1]")
        for label, mean, std in zip(label_names, means, stds):
            lines.append(f"{label}: {_format_mean_std(float(mean), float(std))}")

    save_text(output_path, "\n".join(lines))


def _single_task_runner(
    name: str,
    task: str,
    dataloaders,
    config: TrainConfig,
    device: torch.device,
    output_root: str,
    seed: int,
    target_threshold: float,
) -> Dict:
    paths = build_output_paths(output_root, name)
    print_header(f"Treino {name}")
    result = train_single_task(
        dataloaders,
        task,
        config,
        device,
        checkpoint_path=os.path.join(paths.checkpoints, "best.pt"),
        target_threshold=target_threshold,
    )
    metrics = result["metrics"]
    save_json(paths.metrics_path, metrics)
    _write_final_results(seed, metrics, paths.final_results_path)

    rows = [
        {
            "metric": "primary",
            "value": f"{metrics['primary_metric']:.4f}",
            "output": paths.root,
        }
    ]
    print_metrics_table(rows, headers=["metric", "value", "output"])
    return {"paths": paths, "metrics": metrics}


def _multitask_runner(
    dataloaders,
    config: TrainConfig,
    device: torch.device,
    output_root: str,
    seed: int,
    target_threshold: float,
) -> Dict:
    paths = build_output_paths(output_root, "multitask")
    print_header("Treino multitask")
    result = train_multitask(
        dataloaders,
        config,
        device,
        checkpoint_path=os.path.join(paths.checkpoints, "best.pt"),
        target_threshold=target_threshold,
    )
    metrics = result["metrics"]
    save_json(paths.metrics_path, metrics)
    _write_final_results(seed, metrics, paths.final_results_path)
    rows = [
        {
            "metric": "primary",
            "value": f"{metrics['primary_metric']:.4f}",
            "output": paths.root,
        }
    ]
    print_metrics_table(rows, headers=["metric", "value", "output"])
    return {"paths": paths, "metrics": metrics}


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds, args.seed)
    device = get_device()
    _device_banner(device)

    output_root = os.path.join(RESULTS_ROOT, f"experiment_{seeds[0]}_{seeds[1]}")
    os.makedirs(output_root, exist_ok=True)

    print_header("Dataset")
    dataset_bundle = load_hatebr_dataset(args.mask_urls_users, args.model_name)
    results_by_seed: Dict[int, Dict] = {}
    for seed in seeds:
        set_seed(seed)
        seed_root = os.path.join(output_root, f"seed_{seed}")
        os.makedirs(seed_root, exist_ok=True)

        tokenized = tokenize_dataset(
            dataset_bundle.dataset,
            dataset_bundle.tokenizer,
            args.max_length,
            seed=seed,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloaders = build_dataloaders(
            tokenized,
            args.batch_size,
            args.num_workers,
            seed_worker_fn=seed_worker,
            generator=generator,
        )
        print_header("Data Partition")
        print_kv_table(
            {
                "seed": str(seed),
                "train": str(len(tokenized["train"])),
                "validation": str(len(tokenized["validation"])),
                "test": str(len(tokenized["test"])),
            },
            title="Split sizes",
        )

        config = TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            patience=args.patience,
            use_fp16=args.use_fp16,
            w_bin=args.w_bin,
            w_level=args.w_level,
            w_target=args.w_target,
            model_name=args.model_name,
        )

        results = {}
        results["singletask_offensive"] = _single_task_runner(
            "singletask_offensive",
            "bin",
            dataloaders,
            config,
            device,
            seed_root,
            seed,
            args.target_threshold,
        )
        results["singletask_level"] = _single_task_runner(
            "singletask_level",
            "level",
            dataloaders,
            config,
            device,
            seed_root,
            seed,
            args.target_threshold,
        )
        results["singletask_target"] = _single_task_runner(
            "singletask_target",
            "target",
            dataloaders,
            config,
            device,
            seed_root,
            seed,
            args.target_threshold,
        )
        results["multitask"] = _multitask_runner(
            dataloaders,
            config,
            device,
            seed_root,
            seed,
            args.target_threshold,
        )

        plot_comparison = {
            "singletask_offensive": {
                "f1_bin_pos": results["singletask_offensive"]["metrics"]["metrics"]["f1_pos"],
                "f1_level_macro": results["singletask_level"]["metrics"]["metrics"]["f1_macro"],
                "f1_target_micro": results["singletask_target"]["metrics"]["metrics"]["micro_f1"],
            },
            "multitask": {
                "f1_bin_pos": results["multitask"]["metrics"]["metrics"]["bin"]["f1_pos"],
                "f1_level_macro": results["multitask"]["metrics"]["metrics"]["level"]["f1_macro"],
                "f1_target_micro": results["multitask"]["metrics"]["metrics"]["target"]["micro_f1"],
            },
        }

        plot_paths = {
            "comparison": os.path.join(seed_root, "multitask", "plots", "comparison_bar.png"),
            "confusion": os.path.join(seed_root, "multitask", "plots", "confusion_level.png"),
            "pr_curve": os.path.join(seed_root, "multitask", "plots", "pr_curve_bin.png"),
            "inconsistency": os.path.join(seed_root, "multitask", "plots", "inconsistency_bar.png"),
            "target_f1": os.path.join(seed_root, "multitask", "plots", "target_f1_bar.png"),
        }

        plot_comparison_bar(plot_paths["comparison"], plot_comparison)
        confusion = results["multitask"]["metrics"]["metrics"]["level"]["confusion_matrix"]
        plot_confusion_matrix(plot_paths["confusion"], confusion)

        pr_single = results["singletask_offensive"]["metrics"]["raw"]
        pr_multi = results["multitask"]["metrics"]["raw"]
        precision_s, recall_s, _ = compute_pr_curve(pr_single["labels"], pr_single["probs"])
        precision_m, recall_m, _ = compute_pr_curve(pr_multi["bin_labels"], pr_multi["bin_probs"])
        plot_pr_curve(
            plot_paths["pr_curve"],
            {
                "singletask_offensive": {"precision": precision_s, "recall": recall_s},
                "multitask": {"precision": precision_m, "recall": recall_m},
            },
        )

        level_preds_single = results["singletask_level"]["metrics"]["raw"]["preds"]
        target_preds_single = results["singletask_target"]["metrics"]["raw"]["preds"]
        bin_pred_from_level = (level_preds_single > 0).astype(int)
        bin_pred_from_target = (target_preds_single.sum(axis=1) > 0).astype(int)
        inconsistency_single = {
            "inconsistency_rate_target": float(
                np.mean((bin_pred_from_target == 0) & (target_preds_single.sum(axis=1) > 0))
            ),
            "inconsistency_rate_level": float(np.mean((bin_pred_from_level == 0) & (level_preds_single > 0))),
        }
        inconsistency_multi = results["multitask"]["metrics"]["metrics"]["consistency"]

        plot_inconsistencies(
            plot_paths["inconsistency"],
            {
                "single-task": inconsistency_single,
                "multitask": inconsistency_multi,
            },
        )

        target_f1 = results["multitask"]["metrics"]["metrics"]["target"]["per_class_f1"]
        plot_target_f1_bars(
            plot_paths["target_f1"],
            dataset_bundle.label_names["target"],
            target_f1,
        )

        print_header("Resultados")
        print_kv_table(
            {
                "seed": str(seed),
                "comparison_plot": plot_paths["comparison"],
                "confusion_plot": plot_paths["confusion"],
                "pr_curve_plot": plot_paths["pr_curve"],
                "inconsistency_plot": plot_paths["inconsistency"],
                "target_f1_plot": plot_paths["target_f1"],
            },
            title="Plots",
        )

        results_by_seed[seed] = results

    final_results_path = os.path.join(output_root, "final_results.txt")
    _write_summary_report(
        final_results_path,
        seeds,
        results_by_seed,
        dataset_bundle.label_names["target"],
    )


if __name__ == "__main__":
    main()
