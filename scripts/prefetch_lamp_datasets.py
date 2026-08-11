#!/usr/bin/env python3
"""
Download and preprocess LaMP/LongLaMP datasets into the local dataset root.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

try:
    from datasets import DownloadConfig, load_dataset
except Exception:  # pragma: no cover - optional dependency at runtime
    DownloadConfig = None
    load_dataset = None

BASE_URL = "https://ciir.cs.umass.edu/downloads/LaMP"
LAMP_PATHS = {
    "LaMP-1": "LaMP_1",
    "LaMP-2": "LaMP_2/new",
    "LaMP-3": "LaMP_3",
    "LaMP-4": "LaMP_4",
    "LaMP-5": "LaMP_5",
    "LaMP-7": "LaMP_7",
}

DEFAULT_LAMP_TASKS = ["LaMP-1", "LaMP-2", "LaMP-3", "LaMP-4", "LaMP-5", "LaMP-7"]
DEFAULT_LONGLAMP_TASKS = ["LongLaMP-2", "LongLaMP-3", "LongLaMP-4"]
OFFLINE_ENV_VARS = (
    "HF_OFFLINE",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)
LONGLAMP_CONFIGS = {
    "LongLaMP-2": "abstract_generation_user",
    "LongLaMP-3": "topic_writing_user",
    "LongLaMP-4": "product_review_user",
}


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"[skip] {dest}")
        return
    if _is_offline():
        raise RuntimeError(
            f"Offline mode is enabled but missing required file: {dest}. "
            "Disable offline mode to download the dataset."
        )
    _ensure_parent(dest)
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def download_lamp(task: str, splits: Iterable[str], dataset_root: Path, force: bool) -> None:
    if task not in LAMP_PATHS:
        raise ValueError(f"Unsupported LaMP task: {task}")
    task_dir = dataset_root / task
    task_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = LAMP_PATHS[task]
    for split in splits:
        suffixes = ("questions",) if split == "test" else ("questions", "outputs")
        for suffix in suffixes:
            filename = f"{split}_{suffix}.json"
            url = f"{BASE_URL}/{dataset_path}/{split}/{filename}"
            dest = task_dir / filename
            _download(url, dest, force)


def _is_offline() -> bool:
    return any(
        os.environ.get(var, "").strip().lower() in {"1", "true", "yes"}
        for var in OFFLINE_ENV_VARS
    )


def download_longlamp(task: str, splits: Iterable[str], force: bool) -> None:
    if load_dataset is None or DownloadConfig is None:
        raise RuntimeError(
            "The 'datasets' package is required to prefetch LongLaMP. "
            "Install it or skip LongLaMP prefetching."
        )
    if task not in LONGLAMP_CONFIGS:
        supported = ", ".join(sorted(LONGLAMP_CONFIGS))
        raise ValueError(f"Unsupported LongLaMP task: {task}. Supported: {supported}")

    name = LONGLAMP_CONFIGS[task]
    offline = _is_offline()
    download_config = DownloadConfig(local_files_only=offline)
    download_mode = "force_redownload" if force else "reuse_dataset_if_exists"

    for split in splits:
        split_name = split
        try:
            print(f"[longlamp] {task} {split_name}")
            load_dataset(
                "LongLaMP/LongLaMP",
                name=name,
                split=split_name,
                download_config=download_config,
                download_mode=download_mode,
            )
        except Exception as exc:
            if split == "dev":
                split_name = "test"
                print(f"[longlamp] {task} dev not found, caching test split instead")
                load_dataset(
                    "LongLaMP/LongLaMP",
                    name=name,
                    split=split_name,
                    download_config=download_config,
                    download_mode=download_mode,
                )
            else:
                raise RuntimeError(
                    f"Failed to prefetch {task} {split}: {exc}"
                ) from exc


def _get_lamp_loader():
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    try:
        from src.lamp_benchmark import load_lamp_dataset
    except Exception as exc:
        print(f"[warn] Could not import load_lamp_dataset: {exc}")
        return None
    return load_lamp_dataset


def materialize_to_disk(
    tasks: Iterable[str],
    splits: Iterable[str],
    load_lamp_dataset_fn,
) -> None:
    if load_lamp_dataset_fn is None:
        return
    for task in tasks:
        for split in splits:
            try:
                load_lamp_dataset_fn(task, split=split)
                print(f"[materialize] {task} {split}")
            except Exception as exc:
                print(f"[warn] Failed to materialize {task} {split}: {exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch LaMP/LongLaMP datasets for offline runs."
    )
    parser.add_argument(
        "--splits",
        default="train,dev",
        help="Comma-separated splits to download (default: train,dev).",
    )
    parser.add_argument(
        "--lamp_tasks",
        default=",".join(DEFAULT_LAMP_TASKS),
        help="Comma-separated LaMP tasks (default: all supported).",
    )
    parser.add_argument(
        "--longlamp_tasks",
        default=",".join(DEFAULT_LONGLAMP_TASKS),
        help="Comma-separated LongLaMP tasks (default: all supported).",
    )
    parser.add_argument(
        "--dataset_root",
        default=os.environ.get("LAMP_DATA_ROOT", "./data/lamp"),
        help="Root directory to store LaMP JSON files.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files.")
    parser.add_argument("--skip_lamp", action="store_true", help="Skip LaMP.")
    parser.add_argument("--skip_longlamp", action="store_true", help="Skip LongLaMP.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    splits = _split_list(args.splits)
    lamp_tasks = _split_list(args.lamp_tasks)
    longlamp_tasks = _split_list(args.longlamp_tasks)

    if not splits:
        raise ValueError("No splits specified. Use --splits train,dev (etc).")

    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    if _is_offline():
        print("[offline] HF offline mode detected; only cached data will be used.")

    load_lamp_dataset_fn = _get_lamp_loader()

    if not args.skip_lamp:
        for task in lamp_tasks:
            download_lamp(task, splits, dataset_root, args.force)
        materialize_to_disk(lamp_tasks, splits, load_lamp_dataset_fn)

    if not args.skip_longlamp:
        for task in longlamp_tasks:
            download_longlamp(task, splits, args.force)
        materialize_to_disk(longlamp_tasks, splits, load_lamp_dataset_fn)


if __name__ == "__main__":
    main()
