import json
import os
import re
import shutil
import tarfile
import unicodedata
import warnings
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    DownloadConfig,
    Sequence,
    Value,
    __version__ as datasets_version,
    config as datasets_config,
    load_dataset,
    load_from_disk,
)
from transformers import AutoTokenizer

RE_USER = re.compile(r"@\w+")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_SPACES = re.compile(r"\s+")

REQUIRED_COLUMNS = [
    "instagram_comments",
    "offensive_language",
    "offensiveness_levels",
    "xenophobia",
    "racism",
    "homophobia",
    "sexism",
    "religious_intolerance",
    "partyism",
    "apology_for_the_dictatorship",
    "antisemitism",
    "fatphobia",
]

TARGET_COLUMNS = [
    "xenophobia",
    "racism",
    "homophobia",
    "sexism",
    "religious_intolerance",
    "partyism",
    "apology_for_the_dictatorship",
    "antisemitism",
    "fatphobia",
]

_DATA_FILE_PARSERS: Dict[str, Tuple[str, Dict[str, object]]] = {
    ".parquet": ("parquet", {}),
    ".arrow": ("arrow", {}),
    ".jsonl": ("json", {"lines": True}),
    ".json": ("json", {}),
    ".csv": ("csv", {}),
    ".tsv": ("csv", {"delimiter": "\t"}),
    ".txt": ("text", {}),
}
_DATA_FILE_PRIORITY = [".parquet", ".arrow", ".jsonl", ".json", ".csv", ".tsv", ".txt"]
_ARCHIVE_EXTENSIONS = (".zip", ".tar.gz", ".tgz")
_MIN_TRAIN_ROWS = 100
_SPLIT_PATTERNS: Dict[str, re.Pattern] = {
    "train": re.compile(r"(^|[\\/._-])train([\\/._-]|$)", re.IGNORECASE),
    "test": re.compile(r"(^|[\\/._-])test([\\/._-]|$)", re.IGNORECASE),
    "validation": re.compile(r"(^|[\\/._-])(validation|valid|dev)([\\/._-]|$)", re.IGNORECASE),
}
_SKIP_REPO_BASENAMES = {
    "readme",
    "readme.md",
    "license",
    "citation.cff",
    "dataset_info.json",
    "dataset_infos.json",
    "dataset_infos.jsonl",
    "dataset_dict.json",
    "dataset_dict.jsonl",
    "indexes.json",
    "indexes.jsonl",
    "state.json",
    "state.jsonl",
    "splits.json",
    "splits.jsonl",
    "splits.csv",
    "builder_configs.json",
    "metadata.json",
}
_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "instagram_comments": (
        "instagram_comments",
        "instagram_comment",
        "comment",
        "comments",
        "comment_text",
        "text",
        "texto",
        "sentence",
        "post",
        "message",
        "mensagem",
        "comentario",
        "comentarios",
        "comentario_instagram",
        "comentarios_instagram",
    ),
    "offensive_language": (
        "offensive_language",
        "offensive",
        "offensivo",
        "linguagem_ofensiva",
        "ofensivo_binario",
        "offensive_binary",
        "is_offensive",
        "is_offensive_language",
    ),
    "offensiveness_levels": (
        "offensiveness_levels",
        "offensiveness_level",
        "offense_level",
        "offensive_level",
        "level",
        "nivel",
        "nivel_ofensividade",
        "nivel_ofensivo",
        "severity",
        "severity_level",
        "classe_ofensividade",
    ),
    "xenophobia": ("xenophobia", "xenofobia"),
    "racism": ("racism", "racismo"),
    "homophobia": ("homophobia", "homofobia"),
    "sexism": ("sexism", "sexismo"),
    "religious_intolerance": (
        "religious_intolerance",
        "intolerancia_religiosa",
        "intolerancia_a_religiao",
        "intolerancia_religiao",
    ),
    "partyism": ("partyism", "partidarismo", "partidarismo_politico"),
    "apology_for_the_dictatorship": (
        "apology_for_the_dictatorship",
        "apologia_ditadura",
        "apologia_a_ditadura",
        "apologia_da_ditadura",
        "apologia_ao_regime",
        "apologia_regime",
    ),
    "antisemitism": ("antisemitism", "antissemitismo", "antisemitismo"),
    "fatphobia": ("fatphobia", "gordofobia"),
}
_TEXT_HINTS = ("text", "comment", "coment", "sentence", "post", "message", "tweet", "body")
_BIN_HINTS = ("offen", "ofens", "bin", "binary", "class", "label", "hate")
_LEVEL_HINTS = ("level", "nivel", "grau", "severity", "classe")
_TARGET_HINTS = ("target", "label", "labels", "category", "categories", "class", "classes")
_IGNORE_HINTS = ("id", "idx", "index", "fold", "split")
_ENV_TEXT_COLUMN = "HATEBR_TEXT_COLUMN"
_ENV_BIN_COLUMN = "HATEBR_BIN_COLUMN"
_ENV_LEVEL_COLUMN = "HATEBR_LEVEL_COLUMN"
_ENV_TARGET_COLUMN = "HATEBR_TARGET_COLUMN"
_ENV_TARGET_COLUMNS = "HATEBR_TARGET_COLUMNS"


@dataclass
class DataLoaders:
    train: torch.utils.data.DataLoader
    validation: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader


@dataclass
class DatasetBundle:
    dataset: DatasetDict
    tokenizer: AutoTokenizer
    label_names: Dict[str, List[str]]


def _normalize_text(text: str, mask_urls_users: bool) -> str:
    text = text.strip()
    if mask_urls_users:
        text = RE_USER.sub("<USER>", text)
        text = RE_URL.sub("<URL>", text)
    text = RE_SPACES.sub(" ", text)
    return text


def _validate_columns(dataset: DatasetDict) -> None:
    available = set(dataset["train"].column_names)
    missing = [col for col in REQUIRED_COLUMNS if col not in available]
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Available columns: {sorted(available)}."
        )


def _build_labels(example: Dict) -> Dict:
    y_bin = _coerce_binary(example["offensive_language"])
    y_level = _coerce_level(example["offensiveness_levels"])
    target = [float(_coerce_binary(example[col])) for col in TARGET_COLUMNS]
    return {
        "labels_bin": y_bin,
        "labels_level": y_level,
        "labels_target": target,
    }


def _parse_version(version: str) -> Tuple[int, int, int]:
    parts = version.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = int(parts[2].split("+")[0]) if len(parts) > 2 and parts[2].split("+")[0].isdigit() else 0
    return major, minor, patch


def _ensure_supported_datasets_version() -> None:
    major, _, _ = _parse_version(datasets_version)
    if major >= 3:
        raise RuntimeError(
            f"Unsupported datasets version {datasets_version} detected. "
            "The HateBR dataset requires dataset script support. "
            "Install a compatible version with: pip install 'datasets<3.0'."
        )


def _dataset_cache_path(dataset_name: str) -> str:
    sanitized = dataset_name.replace("/", "___")
    return os.path.join(datasets_config.HF_DATASETS_CACHE, sanitized)


def _clear_dataset_cache(dataset_name: str) -> None:
    cache_path = _dataset_cache_path(dataset_name)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)


def _dataset_module_cache_path(dataset_name: str) -> str:
    sanitized = dataset_name.replace("/", "___")
    return os.path.join(datasets_config.HF_MODULES_CACHE, sanitized)


def _clear_dataset_module_cache(dataset_name: str) -> None:
    candidates = {
        _dataset_module_cache_path(dataset_name),
        os.path.join(datasets_config.HF_MODULES_CACHE, "datasets_modules", dataset_name),
        os.path.join(datasets_config.HF_MODULES_CACHE, "datasets_modules", dataset_name.replace("/", "___")),
        os.path.join(datasets_config.HF_MODULES_CACHE, "datasets_modules", *dataset_name.split("/")),
    }
    for module_cache in candidates:
        if os.path.exists(module_cache):
            shutil.rmtree(module_cache, ignore_errors=True)


def _build_isolated_cache_dir() -> str:
    env_root = os.getenv("HF_DATASETS_CACHE") or os.getenv("HF_HOME")
    if env_root:
        root = os.path.abspath(env_root)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root = os.path.join(project_root, ".hf_cache")
    os.makedirs(root, exist_ok=True)
    return root


def _load_dataset_with_cache(
    dataset_name: str,
    cache_dir: str | None,
    force: bool,
    name: str | None = None,
) -> DatasetDict:
    download_config = DownloadConfig(
        cache_dir=cache_dir,
        force_download=force,
        resume_download=False,
    )
    load_kwargs: Dict[str, object] = {
        "cache_dir": cache_dir,
        "download_mode": "force_redownload" if force else "reuse_dataset_if_exists",
        "download_config": download_config,
    }
    if name:
        load_kwargs["name"] = name
    return load_dataset(
        dataset_name,
        **load_kwargs,
    )


def _detect_data_file_type(filename: str) -> Tuple[str, str, Dict[str, object]] | None:
    lower = filename.lower()
    if lower.endswith(".gz"):
        lower = lower[:-3]
    if lower.endswith(".arrow"):
        return ".arrow", "arrow", {}
    for ext, (builder, kwargs) in _DATA_FILE_PARSERS.items():
        if lower.endswith(ext):
            return ext, builder, kwargs
    return None


def _infer_split_name(filename: str) -> str | None:
    for split_name, pattern in _SPLIT_PATTERNS.items():
        if pattern.search(filename):
            return split_name
    return None


def _build_repo_data_candidates(
    repo_files: List[str],
) -> List[Tuple[str, Dict[str, List[str]], Dict[str, object]]]:
    candidates_by_ext: Dict[str, Dict[str, object]] = {}
    for filename in repo_files:
        if os.path.basename(filename).lower() in _SKIP_REPO_BASENAMES:
            continue
        info = _detect_data_file_type(filename)
        if not info:
            continue
        ext, builder, kwargs = info
        entry = candidates_by_ext.setdefault(
            ext,
            {"builder": builder, "kwargs": kwargs, "files": []},
        )
        entry["files"].append(filename)

    if not candidates_by_ext:
        return []

    ordered_candidates: List[Tuple[str, Dict[str, List[str]], Dict[str, object]]] = []
    for ext in _DATA_FILE_PRIORITY:
        entry = candidates_by_ext.get(ext)
        if not entry:
            continue
        data_files: Dict[str, List[str]] = {}
        for filename in entry["files"]:
            split = _infer_split_name(filename) or "train"
            data_files.setdefault(split, []).append(filename)
        ordered_candidates.append((entry["builder"], data_files, entry["kwargs"]))
    if not ordered_candidates:
        for entry in candidates_by_ext.values():
            data_files = {}
            for filename in entry["files"]:
                split = _infer_split_name(filename) or "train"
                data_files.setdefault(split, []).append(filename)
            ordered_candidates.append((entry["builder"], data_files, entry["kwargs"]))
    return ordered_candidates


def _build_local_data_candidates(
    root_dir: str,
) -> List[Tuple[str, Dict[str, List[str]], Dict[str, object]]]:
    candidates_by_ext: Dict[str, Dict[str, object]] = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower() in _SKIP_REPO_BASENAMES:
                continue
            if any(filename.lower().endswith(ext) for ext in _ARCHIVE_EXTENSIONS):
                continue
            info = _detect_data_file_type(filename)
            if not info:
                continue
            ext, builder, kwargs = info
            entry = candidates_by_ext.setdefault(
                ext,
                {"builder": builder, "kwargs": kwargs, "files": []},
            )
            entry["files"].append(os.path.join(dirpath, filename))

    if not candidates_by_ext:
        return []

    ordered_candidates: List[Tuple[str, Dict[str, List[str]], Dict[str, object]]] = []
    for ext in _DATA_FILE_PRIORITY:
        entry = candidates_by_ext.get(ext)
        if not entry:
            continue
        data_files: Dict[str, List[str]] = {}
        for filepath in entry["files"]:
            relative = os.path.relpath(filepath, root_dir)
            split = _infer_split_name(relative) or "train"
            data_files.setdefault(split, []).append(filepath)
        ordered_candidates.append((entry["builder"], data_files, entry["kwargs"]))
    if not ordered_candidates:
        for entry in candidates_by_ext.values():
            data_files = {}
            for filepath in entry["files"]:
                relative = os.path.relpath(filepath, root_dir)
                split = _infer_split_name(relative) or "train"
                data_files.setdefault(split, []).append(filepath)
            ordered_candidates.append((entry["builder"], data_files, entry["kwargs"]))
    return ordered_candidates


def _collect_archives(root_dir: str) -> List[str]:
    archives = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            lower = filename.lower()
            if any(lower.endswith(ext) for ext in _ARCHIVE_EXTENSIONS):
                archives.append(os.path.join(dirpath, filename))
    return archives


def _load_from_disk_dir(path: str) -> DatasetDict | None:
    try:
        loaded = load_from_disk(path)
    except Exception:
        return None
    if isinstance(loaded, Dataset):
        return DatasetDict({"train": loaded})
    if isinstance(loaded, DatasetDict):
        return loaded
    return None


def _load_from_disk_snapshot(snapshot_dir: str) -> DatasetDict | None:
    dataset_dict_paths = []
    for dirpath, _, filenames in os.walk(snapshot_dir):
        if "dataset_dict.json" in filenames:
            dataset_dict_paths.append(dirpath)

    best_dataset = None
    best_len = -1
    for path in dataset_dict_paths:
        candidate = _load_from_disk_dir(path)
        if candidate is None or "train" not in candidate:
            continue
        train_len = len(candidate["train"])
        if train_len > best_len:
            best_len = train_len
            best_dataset = candidate
    if best_dataset is not None:
        return best_dataset

    split_dirs: Dict[str, Dataset] = {}
    for split_name in ("train", "validation", "test"):
        for dirpath, _, filenames in os.walk(snapshot_dir):
            if os.path.basename(dirpath) != split_name:
                continue
            if "state.json" not in filenames:
                continue
            candidate = _load_from_disk_dir(dirpath)
            if candidate is None or "train" not in candidate:
                continue
            split_dirs[split_name] = candidate["train"]
            break

    if split_dirs:
        return DatasetDict(split_dirs)

    if os.path.isfile(os.path.join(snapshot_dir, "state.json")):
        candidate = _load_from_disk_dir(snapshot_dir)
        if candidate is not None:
            return candidate

    return None


def _find_dataset_script(snapshot_dir: str, dataset_name: str) -> str | None:
    expected = f"{dataset_name.split('/')[-1]}.py"
    for dirpath, _, filenames in os.walk(snapshot_dir):
        if expected in filenames:
            return os.path.join(dirpath, expected)
    for dirpath, _, filenames in os.walk(snapshot_dir):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            if filename in {"__init__.py", "setup.py"}:
                continue
            return os.path.join(dirpath, filename)
    return None


def _load_dataset_from_script(
    snapshot_dir: str,
    dataset_name: str,
    cache_dir: str | None,
    force: bool,
) -> DatasetDict | None:
    script_path = _find_dataset_script(snapshot_dir, dataset_name)
    if not script_path:
        return None
    try:
        return load_dataset(
            script_path,
            data_dir=snapshot_dir,
            cache_dir=cache_dir,
            download_mode="force_redownload" if force else "reuse_dataset_if_exists",
            trust_remote_code=True,
        )
    except Exception:
        return None


def _extract_archives(archives: List[str], dest_dir: str) -> str | None:
    if not archives:
        return None
    os.makedirs(dest_dir, exist_ok=True)
    for archive_path in archives:
        lower = archive_path.lower()
        try:
            if lower.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as archive:
                    archive.extractall(dest_dir)
            elif lower.endswith(".tar.gz") or lower.endswith(".tgz"):
                with tarfile.open(archive_path, "r:gz") as archive:
                    archive.extractall(dest_dir)
        except (tarfile.TarError, zipfile.BadZipFile):
            continue
    return dest_dir


def _normalize_column_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def _build_target_alias_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for target in TARGET_COLUMNS:
        aliases = _COLUMN_ALIASES.get(target, ()) + (target,)
        for alias in aliases:
            lookup[_normalize_column_name(alias)] = target
    return lookup


_TARGET_ALIAS_LOOKUP = _build_target_alias_lookup()


def _column_name_score(name: str, hints: Tuple[str, ...]) -> int:
    normalized = _normalize_column_name(name)
    return sum(1 for hint in hints if hint in normalized)


def _is_probably_id_column(name: str) -> bool:
    normalized = _normalize_column_name(name)
    return any(hint in normalized for hint in _IGNORE_HINTS)


def _is_string_feature(feature: object) -> bool:
    return isinstance(feature, Value) and feature.dtype in {"string", "large_string"}


def _sample_train_batch(train, sample_size: int = 64) -> Dict[str, List[object]]:
    try:
        size = min(sample_size, len(train))
    except TypeError:
        size = sample_size
    if size <= 0:
        return {col: [] for col in train.column_names}
    batch = train[:size]
    return {col: batch.get(col, []) for col in train.column_names}


def _avg_text_length(values: List[object]) -> float:
    lengths = [len(value) for value in values if isinstance(value, str)]
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def _stringify_text_candidate(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if any(isinstance(item, str) for item in value):
            return " ".join(str(item) for item in value if item is not None)
        return ""
    if isinstance(value, dict):
        for key in ("text", "comment", "comments", "sentence", "message", "post"):
            if key in value and isinstance(value[key], str):
                return value[key]
        return ""
    return ""


def _avg_text_candidate_length(values: List[object]) -> float:
    lengths = [len(_stringify_text_candidate(value)) for value in values]
    lengths = [length for length in lengths if length > 0]
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def _infer_text_column(train) -> str | None:
    samples = _sample_train_batch(train)
    candidates: List[Tuple[int, float, str]] = []
    for col, values in samples.items():
        if _is_probably_id_column(col):
            continue
        feature = train.features.get(col) if hasattr(train, "features") else None
        has_string_feature = _is_string_feature(feature)
        if not has_string_feature and not any(isinstance(value, str) for value in values):
            continue
        hint_score = _column_name_score(col, _TEXT_HINTS)
        avg_len = _avg_text_length(values)
        candidates.append((hint_score, avg_len, col))
    if not candidates:
        for col, values in samples.items():
            if _is_probably_id_column(col):
                continue
            hint_score = _column_name_score(col, _TEXT_HINTS)
            avg_len = _avg_text_candidate_length(values)
            if avg_len <= 0:
                continue
            candidates.append((hint_score, avg_len, col))
        if not candidates:
            return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def _parse_int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.isdigit():
            return int(normalized)
    return None


def _parse_binary_value(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return int(value)
        return None
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        normalized = normalized.strip().lower()
        if normalized in {"0", "false", "no", "nao", "n"}:
            return 0
        if normalized in {"1", "true", "yes", "sim", "y"}:
            return 1
        if normalized in {"offensive", "ofensivo"}:
            return 1
        if normalized in {"non_offensive", "non-offensive", "nao_ofensivo"}:
            return 0
    return None


def _infer_binary_column(train) -> str | None:
    samples = _sample_train_batch(train)
    candidates: List[Tuple[int, int, str]] = []
    for col, values in samples.items():
        if _is_probably_id_column(col):
            continue
        feature = train.features.get(col) if hasattr(train, "features") else None
        if isinstance(feature, ClassLabel) and feature.num_classes == 2:
            hint_score = _column_name_score(col, _BIN_HINTS)
            candidates.append((hint_score, 2, col))
            continue
        parsed = [_parse_binary_value(value) for value in values if value is not None]
        parsed = [value for value in parsed if value is not None]
        if not parsed:
            continue
        unique = set(parsed)
        if unique.issubset({0, 1}):
            hint_score = _column_name_score(col, _BIN_HINTS)
            candidates.append((hint_score, len(unique), col))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def _parse_level_value(value: object) -> int | None:
    try:
        return _coerce_level(value)
    except ValueError:
        return None


def _infer_level_column(train) -> Tuple[str | None, int]:
    samples = _sample_train_batch(train)
    candidates: List[Tuple[int, int, int, str]] = []
    for col, values in samples.items():
        if _is_probably_id_column(col):
            continue
        feature = train.features.get(col) if hasattr(train, "features") else None
        if isinstance(feature, ClassLabel) and feature.num_classes in {3, 4}:
            hint_score = _column_name_score(col, _LEVEL_HINTS)
            candidates.append((hint_score, 4, 0, col))
            continue
        parsed = [_parse_level_value(value) for value in values if value is not None]
        parsed = [value for value in parsed if value is not None]
        if not parsed:
            continue
        unique = set(parsed)
        if not unique:
            continue
        max_val = max(unique)
        min_val = min(unique)
        if max_val <= 3 and min_val >= 0:
            hint_score = _column_name_score(col, _LEVEL_HINTS)
            candidates.append((hint_score, len(unique), 0, col))
        elif max_val <= 4 and min_val >= 1:
            hint_score = _column_name_score(col, _LEVEL_HINTS)
            candidates.append((hint_score, len(unique), 1, col))
    if not candidates:
        return None, 0
    candidates.sort(reverse=True)
    _, _, offset, col = candidates[0]
    return col, offset


def _get_label_names_from_feature(feature: object) -> List[str] | None:
    if isinstance(feature, ClassLabel):
        return list(feature.names)
    if isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel):
        return list(feature.feature.names)
    return None


def _infer_target_source(train) -> Tuple[str | None, List[str] | None]:
    samples = _sample_train_batch(train)
    candidates: List[Tuple[int, int, str, List[str] | None]] = []
    for col, values in samples.items():
        if _is_probably_id_column(col):
            continue
        if col in REQUIRED_COLUMNS or col in TARGET_COLUMNS:
            continue
        feature = train.features.get(col) if hasattr(train, "features") else None
        label_names = _get_label_names_from_feature(feature)
        kind_score = 0
        if isinstance(feature, Sequence) or any(isinstance(value, (list, tuple)) for value in values):
            kind_score = 3
        elif any(isinstance(value, dict) for value in values):
            kind_score = 3
        elif isinstance(feature, ClassLabel):
            kind_score = 2
        elif any(isinstance(value, str) and re.search(r"[;,|]", value) for value in values):
            kind_score = 1
        if kind_score == 0:
            continue
        hint_score = _column_name_score(col, _TARGET_HINTS)
        candidates.append((kind_score, hint_score, col, label_names))
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    _, _, col, label_names = candidates[0]
    return col, label_names


def _resolve_target_name(name: str) -> str | None:
    normalized = _normalize_column_name(name)
    return _TARGET_ALIAS_LOOKUP.get(normalized)


def _extract_targets_from_value(
    value: object,
    label_names: List[str] | None,
) -> Dict[str, int]:
    targets = {col: 0 for col in TARGET_COLUMNS}
    if value is None:
        return targets
    if isinstance(value, dict):
        for key, raw in value.items():
            if not isinstance(key, str):
                continue
            target = _resolve_target_name(key)
            if target:
                targets[target] = _coerce_binary(raw)
        return targets
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (int, float, bool)) for item in value):
            parsed = [_parse_int_value(item) for item in value]
            if len(parsed) == len(TARGET_COLUMNS) and all(item is not None for item in parsed):
                for idx, target in enumerate(TARGET_COLUMNS):
                    targets[target] = int(bool(parsed[idx]))
                return targets
            if label_names and all(item is not None for item in parsed):
                for idx in parsed:
                    if idx is not None and 0 <= idx < len(label_names):
                        label = label_names[idx]
                        target = _resolve_target_name(label)
                        if target:
                            targets[target] = 1
                return targets
        for item in value:
            if isinstance(item, str):
                target = _resolve_target_name(item)
                if target:
                    targets[target] = 1
            elif label_names:
                idx = _parse_int_value(item)
                if idx is not None and 0 <= idx < len(label_names):
                    target = _resolve_target_name(label_names[idx])
                    if target:
                        targets[target] = 1
        return targets
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[;,|]", value) if part.strip()]
        for part in parts:
            target = _resolve_target_name(part)
            if target:
                targets[target] = 1
        if targets != {col: 0 for col in TARGET_COLUMNS}:
            return targets
    if label_names:
        idx = _parse_int_value(value)
        if idx is not None and 0 <= idx < len(label_names):
            target = _resolve_target_name(label_names[idx])
            if target:
                targets[target] = 1
    return targets


def _build_column_rename_map(columns: List[str]) -> Dict[str, str]:
    normalized_to_original: Dict[str, List[str]] = {}
    for col in columns:
        normalized_to_original.setdefault(_normalize_column_name(col), []).append(col)

    rename_map: Dict[str, str] = {}
    used = set()
    for required in REQUIRED_COLUMNS:
        if required in columns:
            continue
        required_norm = _normalize_column_name(required)
        dynamic_aliases = (
            required_norm,
            f"label_{required_norm}",
            f"labels_{required_norm}",
            f"target_{required_norm}",
            f"targets_{required_norm}",
        )
        aliases = _COLUMN_ALIASES.get(required, ()) + dynamic_aliases
        for alias in aliases:
            normalized = _normalize_column_name(alias)
            originals = normalized_to_original.get(normalized)
            if not originals or len(originals) != 1:
                continue
            original = originals[0]
            if original in used:
                continue
            rename_map[original] = required
            used.add(original)
            break
    return rename_map


def _coerce_binary(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value) if value in (0, 1) else int(value > 0)
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        normalized = normalized.strip().lower()
        if normalized in {"0", "false", "no", "nao", "n"}:
            return 0
        if normalized in {"1", "true", "yes", "sim", "y"}:
            return 1
        if normalized in {"offensive", "ofensivo"}:
            return 1
        if normalized in {"non_offensive", "non-offensive", "nao_ofensivo"}:
            return 0
    return int(bool(value))


def _coerce_level(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        normalized = normalized.strip().lower()
        mapping = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "non_offensive": 0,
            "non-offensive": 0,
            "nao_ofensivo": 0,
            "light": 1,
            "ligeiro": 1,
            "moderate": 2,
            "moderado": 2,
            "high": 3,
            "alto": 3,
            "severe": 3,
        }
        if normalized in mapping:
            return mapping[normalized]
        if normalized.isdigit():
            return int(normalized)
    raise ValueError(f"Unsupported offensiveness level value: {value!r}")


def _apply_column_aliases(dataset: DatasetDict) -> DatasetDict:
    columns = dataset["train"].column_names
    rename_map = _build_column_rename_map(columns)
    if rename_map:
        dataset = dataset.rename_columns(rename_map)
    return dataset


def _apply_target_columns_override(dataset: DatasetDict) -> DatasetDict:
    override = os.getenv(_ENV_TARGET_COLUMNS)
    if not override:
        return dataset
    columns = dataset["train"].column_names
    names = [name.strip() for name in override.split(",") if name.strip()]
    if len(names) != len(TARGET_COLUMNS):
        raise ValueError(
            f"{_ENV_TARGET_COLUMNS} must provide {len(TARGET_COLUMNS)} columns in order: "
            f"{', '.join(TARGET_COLUMNS)}."
        )
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(
            f"{_ENV_TARGET_COLUMNS} references missing columns: {missing}. "
            f"Available columns: {sorted(columns)}."
        )
    rename_map = dict(zip(names, TARGET_COLUMNS))
    return dataset.rename_columns(rename_map)


def _resolve_env_sources(train) -> Tuple[str | None, str | None, str | None, str | None]:
    text_source = os.getenv(_ENV_TEXT_COLUMN)
    bin_source = os.getenv(_ENV_BIN_COLUMN)
    level_source = os.getenv(_ENV_LEVEL_COLUMN)
    target_source = os.getenv(_ENV_TARGET_COLUMN)
    sources = []
    for source in (text_source, bin_source, level_source, target_source):
        if source is None:
            sources.append(None)
            continue
        source = source.strip()
        sources.append(source or None)
    text_source, bin_source, level_source, target_source = sources
    for source in sources:
        if source and source not in train.column_names:
            raise ValueError(
                f"Column override {source!r} not found in dataset columns: "
                f"{sorted(train.column_names)}."
            )
    return text_source, bin_source, level_source, target_source


def _infer_numeric_target_columns(train) -> Dict[str, str]:
    columns = train.column_names
    matches: Dict[int, str] = {}
    pattern = re.compile(r"(label|labels|target|targets)[_-]?(\d+)$")
    for col in columns:
        normalized = _normalize_column_name(col)
        match = pattern.search(normalized)
        if not match:
            continue
        idx = int(match.group(2))
        matches[idx] = col
    if len(matches) != len(TARGET_COLUMNS):
        return {}
    if set(matches.keys()) == set(range(len(TARGET_COLUMNS))):
        offset = 0
    elif set(matches.keys()) == set(range(1, len(TARGET_COLUMNS) + 1)):
        offset = 1
    else:
        return {}
    return {matches[idx + offset]: TARGET_COLUMNS[idx] for idx in range(len(TARGET_COLUMNS))}


def _fallback_text_column(train, exclude: Tuple[str | None, ...]) -> str | None:
    excluded = {name for name in exclude if name}
    for col in train.column_names:
        if col in excluded or col in TARGET_COLUMNS:
            continue
        if _is_probably_id_column(col):
            continue
        return col
    return None


def _add_missing_columns(
    dataset: DatasetDict,
    text_source: str | None,
    bin_source: str | None,
    level_source: str | None,
    level_offset: int,
    target_source: str | None,
    target_label_names: List[str] | None,
) -> DatasetDict:
    train_columns = dataset["train"].column_names
    missing_text = "instagram_comments" not in train_columns
    missing_bin = "offensive_language" not in train_columns
    missing_level = "offensiveness_levels" not in train_columns
    missing_targets = [col for col in TARGET_COLUMNS if col not in train_columns]

    if missing_text and not text_source:
        warnings.warn(
            "Unable to infer the text column; filling with empty strings. "
            f"Set {_ENV_TEXT_COLUMN} to map the correct column.",
            RuntimeWarning,
        )
    if missing_bin and not bin_source and not level_source:
        warnings.warn(
            "Unable to infer offensive_language; filling with zeros. "
            f"Set {_ENV_BIN_COLUMN} to map the correct column.",
            RuntimeWarning,
        )
    if missing_level and not level_source and not bin_source:
        warnings.warn(
            "Unable to infer offensiveness_levels; filling with zeros. "
            f"Set {_ENV_LEVEL_COLUMN} to map the correct column.",
            RuntimeWarning,
        )
    if missing_targets and not target_source:
        warnings.warn(
            "Unable to infer target labels; filling with zeros. "
            f"Set {_ENV_TARGET_COLUMN} or {_ENV_TARGET_COLUMNS} to map the correct columns.",
            RuntimeWarning,
        )

    if not (missing_text or missing_bin or missing_level or missing_targets):
        return dataset

    def add_columns(example: Dict) -> Dict:
        updates: Dict[str, object] = {}
        if missing_text:
            if text_source:
                value = example.get(text_source)
                updates["instagram_comments"] = "" if value is None else str(value)
            else:
                updates["instagram_comments"] = ""
        if missing_bin:
            if bin_source:
                updates["offensive_language"] = _coerce_binary(example.get(bin_source))
            elif level_source:
                level_value = _parse_level_value(example.get(level_source))
                if level_value is None:
                    level_value = 0
                if level_offset:
                    level_value = max(0, level_value - level_offset)
                updates["offensive_language"] = int(level_value > 0)
            else:
                updates["offensive_language"] = 0
        if missing_level:
            if level_source:
                level_value = _parse_level_value(example.get(level_source))
                if level_value is None:
                    level_value = 0
                if level_offset:
                    level_value = max(0, level_value - level_offset)
                updates["offensiveness_levels"] = level_value
            elif bin_source:
                updates["offensiveness_levels"] = _coerce_binary(example.get(bin_source))
            else:
                updates["offensiveness_levels"] = 0
        if missing_targets:
            if target_source:
                target_values = _extract_targets_from_value(
                    example.get(target_source),
                    target_label_names,
                )
                updates.update({col: target_values[col] for col in missing_targets})
            else:
                updates.update({col: 0 for col in missing_targets})
        return updates

    return dataset.map(add_columns)


def _build_split_dataset(value: object) -> Dataset | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return Dataset.from_list([])
        if all(isinstance(item, dict) for item in value):
            return Dataset.from_list(value)
        return None
    if isinstance(value, dict):
        try:
            return Dataset.from_dict(value)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return _build_split_dataset(parsed)
    return None


def _maybe_expand_nested_splits(dataset: DatasetDict) -> DatasetDict:
    if not isinstance(dataset, DatasetDict):
        return dataset

    def try_expand_from_split(split) -> DatasetDict | None:
        if len(split) != 1:
            return None
        row = split[0]
        split_datasets: Dict[str, Dataset] = {}
        if any(name in split.column_names for name in ("train", "validation", "test")):
            for split_name in ("train", "validation", "test"):
                if split_name not in row:
                    continue
                split_dataset = _build_split_dataset(row.get(split_name))
                if split_dataset is not None:
                    split_datasets[split_name] = split_dataset
            if "train" in split_datasets:
                return DatasetDict(split_datasets)
        for key, value in row.items():
            if key in ("train", "validation", "test"):
                continue
            split_dataset = _build_split_dataset(value)
            if split_dataset is None or len(split_dataset) <= 1:
                continue
            return DatasetDict({"train": split_dataset})
        return None

    expanded = None
    for split in dataset.values():
        expanded = try_expand_from_split(split)
        if expanded is not None:
            break
    if expanded is None:
        return dataset
    warnings.warn(
        "Detected nested splits inside a single-row dataset; expanding into DatasetDict.",
        RuntimeWarning,
    )
    return expanded


def _is_metadata_like(dataset: DatasetDict) -> bool:
    if not isinstance(dataset, DatasetDict) or "train" not in dataset:
        return False
    train = dataset["train"]
    if len(train) > 1:
        return False
    columns = set(train.column_names)
    if columns.intersection(REQUIRED_COLUMNS):
        return False
    if {"train", "validation", "test"}.intersection(columns):
        return True
    if len(train) <= 1:
        return True
    return False


def _normalize_dataset_schema(dataset: DatasetDict) -> DatasetDict:
    dataset = _apply_target_columns_override(dataset)
    dataset = _apply_column_aliases(dataset)
    train = dataset["train"]

    text_source, bin_source, level_source, target_source = _resolve_env_sources(train)
    level_offset = 0

    if "instagram_comments" not in train.column_names and not text_source:
        text_source = _infer_text_column(train)
    if "instagram_comments" not in train.column_names and not text_source:
        text_source = _fallback_text_column(train, (bin_source, level_source, target_source))
        if text_source:
            warnings.warn(
                f"Text column inferred by fallback heuristic: {text_source!r}. "
                f"Set {_ENV_TEXT_COLUMN} to override if needed.",
                RuntimeWarning,
            )
    if "offensive_language" not in train.column_names and not bin_source:
        bin_source = _infer_binary_column(train)
    if "offensiveness_levels" not in train.column_names and not level_source:
        level_source, level_offset = _infer_level_column(train)

    missing_targets = [col for col in TARGET_COLUMNS if col not in train.column_names]
    if missing_targets:
        numeric_map = _infer_numeric_target_columns(train)
        if numeric_map:
            dataset = dataset.rename_columns(numeric_map)
            train = dataset["train"]
            missing_targets = [col for col in TARGET_COLUMNS if col not in train.column_names]

    target_label_names = None
    if missing_targets and not target_source:
        target_source, target_label_names = _infer_target_source(train)
    elif target_source:
        feature = train.features.get(target_source) if hasattr(train, "features") else None
        target_label_names = _get_label_names_from_feature(feature)

    dataset = _add_missing_columns(
        dataset,
        text_source=text_source,
        bin_source=bin_source,
        level_source=level_source,
        level_offset=level_offset,
        target_source=target_source,
        target_label_names=target_label_names,
    )
    return dataset


def _has_required_columns(dataset: DatasetDict) -> bool:
    return set(REQUIRED_COLUMNS).issubset(set(dataset["train"].column_names))


def _try_load_with_configs(
    dataset_name: str,
    cache_dir: str | None,
    force: bool,
) -> DatasetDict | None:
    try:
        from datasets import get_dataset_config_names
    except Exception:
        return None

    try:
        config_names = get_dataset_config_names(dataset_name)
    except Exception:
        return None

    for name in config_names:
        if not name:
            continue
        try:
            candidate = _load_dataset_with_cache(dataset_name, cache_dir=cache_dir, force=force, name=name)
        except Exception:
            continue
        candidate = _apply_column_aliases(candidate)
        if _has_required_columns(candidate):
            return candidate
    return None


def _load_dataset_from_repo_files(
    dataset_name: str,
    cache_dir: str | None,
    force: bool,
) -> DatasetDict:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for the HateBR fallback loader. "
            "Install it with: pip install 'huggingface_hub>=0.14'."
        ) from exc

    repo_files = list_repo_files(dataset_name, repo_type="dataset")
    candidates = _build_repo_data_candidates(repo_files)
    if not candidates:
        candidates = []

    best_dataset = None
    best_score: Tuple[int, int] = (-1, -1)
    last_error: Exception | None = None

    for builder, data_files, builder_kwargs in candidates:
        local_data_files: Dict[str, List[str]] = {}
        for split, filenames in data_files.items():
            local_data_files[split] = [
                hf_hub_download(
                    repo_id=dataset_name,
                    repo_type="dataset",
                    filename=filename,
                    cache_dir=cache_dir,
                    force_download=force,
                )
                for filename in filenames
            ]
        try:
            candidate = load_dataset(
                builder,
                data_files=local_data_files,
                cache_dir=cache_dir,
                download_mode="force_redownload" if force else "reuse_dataset_if_exists",
                **builder_kwargs,
            )
        except Exception as exc:
            last_error = exc
            continue
        candidate = _maybe_expand_nested_splits(candidate)
        if _is_metadata_like(candidate):
            continue
        train_len = len(candidate["train"]) if "train" in candidate else 0
        total_len = sum(len(split) for split in candidate.values()) if isinstance(candidate, DatasetDict) else 0
        score = (train_len, total_len)
        if score > best_score:
            best_score = score
            best_dataset = candidate
        if train_len >= _MIN_TRAIN_ROWS:
            return candidate

    if best_dataset is not None and len(best_dataset["train"]) >= 2:
        return best_dataset

    def download_snapshot(allow_patterns: List[str] | None) -> str | None:
        try:
            return snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                cache_dir=cache_dir,
                force_download=force,
                allow_patterns=allow_patterns,
            )
        except Exception:
            return None

    snapshot_dir = download_snapshot(
        [
            "dataset_dict.json",
            "**/dataset_dict.json",
            "dataset_info.json",
            "**/dataset_info.json",
            "dataset_infos.json",
            "**/dataset_infos.json",
            "indexes.json",
            "**/indexes.json",
            "state.json",
            "**/state.json",
            "*.py",
            "**/*.py",
            "*.parquet",
            "*.parquet.gz",
            "*.arrow",
            "*.json",
            "*.jsonl",
            "*.json.gz",
            "*.jsonl.gz",
            "*.csv",
            "*.csv.gz",
            "*.tsv",
            "*.tsv.gz",
            "*.txt",
            "*.txt.gz",
            "**/*.parquet",
            "**/*.parquet.gz",
            "**/*.arrow",
            "**/*.json",
            "**/*.jsonl",
            "**/*.json.gz",
            "**/*.jsonl.gz",
            "**/*.csv",
            "**/*.csv.gz",
            "**/*.tsv",
            "**/*.tsv.gz",
            "**/*.txt",
            "**/*.txt.gz",
            "**/*.zip",
            "**/*.tar.gz",
            "**/*.tgz",
        ]
    )
    if snapshot_dir is None:
        snapshot_dir = download_snapshot(None)
    if snapshot_dir is None:
        last_error = RuntimeError("snapshot_download failed")

    if snapshot_dir:
        disk_dataset = _load_from_disk_snapshot(snapshot_dir)
        if disk_dataset is not None:
            disk_dataset = _maybe_expand_nested_splits(disk_dataset)
            if not _is_metadata_like(disk_dataset):
                return disk_dataset

        script_dataset = _load_dataset_from_script(
            snapshot_dir,
            dataset_name,
            cache_dir=cache_dir,
            force=force,
        )
        if script_dataset is not None:
            script_dataset = _maybe_expand_nested_splits(script_dataset)
            if not _is_metadata_like(script_dataset):
                return script_dataset

        local_candidates = _build_local_data_candidates(snapshot_dir)
        for builder, data_files, builder_kwargs in local_candidates:
            try:
                candidate = load_dataset(
                    builder,
                    data_files=data_files,
                    cache_dir=cache_dir,
                    download_mode="force_redownload" if force else "reuse_dataset_if_exists",
                    **builder_kwargs,
                )
            except Exception as exc:
                last_error = exc
                continue
            candidate = _maybe_expand_nested_splits(candidate)
            if _is_metadata_like(candidate):
                continue
            train_len = len(candidate["train"]) if "train" in candidate else 0
            total_len = (
                sum(len(split) for split in candidate.values()) if isinstance(candidate, DatasetDict) else 0
            )
            score = (train_len, total_len)
            if score > best_score:
                best_score = score
                best_dataset = candidate
            if train_len >= _MIN_TRAIN_ROWS:
                return candidate

        if best_dataset is None:
            archives = _collect_archives(snapshot_dir)
            extracted_dir = _extract_archives(archives, os.path.join(snapshot_dir, "_extracted"))
            if extracted_dir:
                local_candidates = _build_local_data_candidates(extracted_dir)
                for builder, data_files, builder_kwargs in local_candidates:
                    try:
                        candidate = load_dataset(
                            builder,
                            data_files=data_files,
                            cache_dir=cache_dir,
                            download_mode="force_redownload" if force else "reuse_dataset_if_exists",
                            **builder_kwargs,
                        )
                    except Exception as exc:
                        last_error = exc
                        continue
                    candidate = _maybe_expand_nested_splits(candidate)
                    if _is_metadata_like(candidate):
                        continue
                    train_len = len(candidate["train"]) if "train" in candidate else 0
                    total_len = (
                        sum(len(split) for split in candidate.values())
                        if isinstance(candidate, DatasetDict)
                        else 0
                    )
                    score = (train_len, total_len)
                    if score > best_score:
                        best_score = score
                        best_dataset = candidate
                    if train_len >= _MIN_TRAIN_ROWS:
                        return candidate

    if best_dataset is not None:
        return best_dataset
    if last_error is not None:
        raise RuntimeError(f"Failed to load HateBR from repo files: {last_error}") from last_error
    raise RuntimeError("Failed to load HateBR from repo files.")


def load_hatebr_dataset(mask_urls_users: bool, model_name: str) -> DatasetBundle:
    _ensure_supported_datasets_version()
    dataset_name = "ruanchaves/hatebr"
    cache_dir: str | None = None
    force_download = False
    dataset = None
    try:
        fallback_cache = _build_isolated_cache_dir()
        dataset = _load_dataset_from_repo_files(
            dataset_name,
            cache_dir=fallback_cache,
            force=force_download,
        )
    except Exception:
        dataset = None

    if dataset is None or "train" not in dataset or len(dataset["train"]) < 2:
        try:
            dataset = _load_dataset_with_cache(dataset_name, cache_dir=cache_dir, force=force_download)
        except UnicodeDecodeError:
            _clear_dataset_cache(dataset_name)
            _clear_dataset_module_cache(dataset_name)
            try:
                force_download = True
                dataset = _load_dataset_with_cache(dataset_name, cache_dir=cache_dir, force=force_download)
            except UnicodeDecodeError as exc:
                cache_dir = _build_isolated_cache_dir()
                force_download = True
                try:
                    dataset = _load_dataset_from_repo_files(
                        dataset_name,
                        cache_dir=cache_dir,
                        force=force_download,
                    )
                except Exception as retry_exc:
                    raise RuntimeError(
                        "Failed to load HateBR due to repeated cache decode errors. "
                        "An isolated cache was created in .hf_cache and a data-file fallback was attempted, "
                        "but the load still failed. Please remove the datasets cache manually "
                        "(e.g., ~/.cache/huggingface/datasets and ~/.cache/huggingface/modules) and retry."
                    ) from retry_exc
        except RuntimeError as exc:
            message = str(exc)
            if "Dataset scripts are no longer supported" in message:
                raise RuntimeError(
                    "The HateBR dataset requires a datasets version that still supports dataset scripts. "
                    "Please install a compatible version, for example: pip install 'datasets<3.0'."
                ) from exc
            raise
    dataset = _maybe_expand_nested_splits(dataset)
    dataset = _apply_column_aliases(dataset)
    if not _has_required_columns(dataset):
        alternate = _try_load_with_configs(dataset_name, cache_dir=cache_dir, force=force_download)
        if alternate is not None:
            dataset = alternate
    if not _has_required_columns(dataset):
        try:
            fallback_cache = cache_dir or _build_isolated_cache_dir()
            dataset = _load_dataset_from_repo_files(
                dataset_name,
                cache_dir=fallback_cache,
                force=force_download,
            )
        except Exception:
            pass
    dataset = _maybe_expand_nested_splits(dataset)
    dataset = _normalize_dataset_schema(dataset)
    _validate_columns(dataset)
    if len(dataset["train"]) < _MIN_TRAIN_ROWS:
        try:
            fallback_cache = _build_isolated_cache_dir()
            fallback = _load_dataset_from_repo_files(
                dataset_name,
                cache_dir=fallback_cache,
                force=True,
            )
            fallback = _normalize_dataset_schema(fallback)
            _validate_columns(fallback)
            if len(fallback["train"]) > len(dataset["train"]):
                dataset = fallback
        except Exception:
            pass
    if len(dataset["train"]) < _MIN_TRAIN_ROWS:
        warnings.warn(
            "HateBR train split looks unusually small. "
            "This may indicate the wrong data file was loaded. "
            "Consider setting HATEBR_TEXT_COLUMN or clearing the Hugging Face cache.",
            RuntimeWarning,
        )

    def preprocess(example: Dict) -> Dict:
        text = _normalize_text(example["instagram_comments"], mask_urls_users)
        labels = _build_labels(example)
        return {"text": text, **labels}

    dataset = dataset.map(preprocess)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return DatasetBundle(
        dataset=dataset,
        tokenizer=tokenizer,
        label_names={
            "bin": ["non_offensive", "offensive"],
            "level": ["non_offensive", "light", "moderate", "high"],
            "target": TARGET_COLUMNS,
        },
    )


def _split_dataset(train, test_fraction: float, seed: int | None) -> Tuple[object, object]:
    size = len(train)
    if size < 2:
        return train, train.select([])
    if test_fraction <= 0:
        return train, train.select([])
    test_size = int(round(size * test_fraction))
    if test_size <= 0:
        test_size = 1
    if test_size >= size:
        test_size = size - 1
    split = train.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def _ensure_splits(dataset: DatasetDict, seed: int | None = None) -> DatasetDict:
    if "train" not in dataset:
        raise ValueError("Dataset must include a train split.")
    if "validation" in dataset and "test" in dataset:
        return dataset

    rng_seed = 42 if seed is None else seed
    train = dataset["train"]
    validation = dataset.get("validation")
    test = dataset.get("test")

    if validation is not None and test is None:
        validation, test = _split_dataset(validation, test_fraction=0.5, seed=rng_seed)
        return DatasetDict(train=train, validation=validation, test=test)

    if validation is None and test is not None:
        train, validation = _split_dataset(train, test_fraction=0.1, seed=rng_seed)
        return DatasetDict(train=train, validation=validation, test=test)

    if validation is None and test is None:
        train, holdout = _split_dataset(train, test_fraction=0.2, seed=rng_seed)
        validation, test = _split_dataset(holdout, test_fraction=0.5, seed=rng_seed)
        return DatasetDict(train=train, validation=validation, test=test)

    return dataset


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int,
    seed: int | None = None,
) -> DatasetDict:
    dataset = _ensure_splits(dataset, seed=seed)

    def tokenize(batch: Dict) -> Dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def clean_split(split: object) -> object:
        if not hasattr(split, "column_names"):
            return split
        drop_cols = [
            name
            for name in ("train", "validation", "test")
            if name in split.column_names
            and name not in REQUIRED_COLUMNS
            and name not in TARGET_COLUMNS
            and name != "text"
        ]
        if drop_cols:
            split = split.remove_columns(drop_cols)
        return split

    tokenized_splits: Dict[str, Dataset] = {}
    for split_name, split in dataset.items():
        split = clean_split(split)
        if len(split) > 0:
            if "input_ids" not in split.column_names or "attention_mask" not in split.column_names:
                split = split.map(tokenize, batched=True)
        tokenized_splits[split_name] = split

    tokenized = DatasetDict(tokenized_splits)
    train_features = tokenized["train"].features if "train" in tokenized else None
    if train_features:
        for split_name, split in tokenized.items():
            if len(split) == 0 and (
                "input_ids" not in split.column_names or "attention_mask" not in split.column_names
            ):
                empty_data = {name: [] for name in train_features}
                tokenized[split_name] = Dataset.from_dict(empty_data, features=train_features)

    missing_after = [
        name
        for name in ("input_ids", "attention_mask")
        if name not in tokenized["train"].column_names
    ]
    if missing_after:
        raise ValueError(f"Tokenization failed to create columns: {missing_after}.")

    columns = ["input_ids", "attention_mask", "labels_bin", "labels_level", "labels_target"]
    tokenized.set_format(type="torch", columns=columns)
    return tokenized


def build_dataloaders(
    tokenized: DatasetDict,
    batch_size: int,
    num_workers: int,
    seed_worker_fn,
    generator: torch.Generator,
) -> DataLoaders:
    train_loader = torch.utils.data.DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker_fn,
        generator=generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        tokenized["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        tokenized["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker_fn,
    )
    return DataLoaders(train=train_loader, validation=validation_loader, test=test_loader)


def get_target_columns() -> Tuple[str, ...]:
    return tuple(TARGET_COLUMNS)
