import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import torch


@dataclass
class OutputPaths:
    root: str
    checkpoints: str
    plots: str
    metrics_path: str
    final_results_path: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_header(title: str) -> None:
    line = "=" * 50
    print(f"\n{line}\n{title}\n{line}")


def print_subheader(title: str) -> None:
    line = "-" * 50
    print(f"\n{title}\n{line}")


def print_kv_table(data: Dict[str, str], title: str = "") -> None:
    if title:
        print_subheader(title)
    max_key = max((len(k) for k in data), default=0)
    for key, value in data.items():
        print(f"{key.ljust(max_key)} : {value}")


def print_metrics_table(rows: List[Dict[str, str]], headers: Iterable[str]) -> None:
    headers = list(headers)
    col_widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers))


def build_output_paths(base_dir: str, run_name: str) -> OutputPaths:
    root = os.path.join(base_dir, run_name)
    checkpoints = os.path.join(root, "checkpoints")
    plots = os.path.join(root, "plots")
    ensure_dir(checkpoints)
    ensure_dir(plots)
    metrics_path = os.path.join(root, "metrics.json")
    final_results_path = os.path.join(root, "final_results.txt")
    return OutputPaths(
        root=root,
        checkpoints=checkpoints,
        plots=plots,
        metrics_path=metrics_path,
        final_results_path=final_results_path,
    )


def save_json(path: str, data: Dict) -> None:
    def sanitize(value: object) -> object:
        if isinstance(value, dict):
            return {key: sanitize(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(val) for val in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        return value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize(data), f, ensure_ascii=False, indent=2)


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
