#!/usr/bin/env python3
"""
Compute per-task K given a target group size.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.lamp_benchmark import load_lamp_dataset


def compute_k(task: str, split: str, target_group: int) -> int:
    if target_group <= 0:
        raise ValueError("target_group must be > 0")

    dataset = load_lamp_dataset(task, split=split)

    n = len(dataset)
    k = int(n / target_group + 0.5)
    if k < 1:
        k = 1
    if k > n:
        k = n
    return k


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute K from target group size")
    parser.add_argument("--task", required=True, help="LaMP-1, LaMP-2, LongLaMP-2, ...")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--target_group", type=int, default=100)
    args = parser.parse_args()

    k = compute_k(args.task, args.split, args.target_group)
    print(k)


if __name__ == "__main__":
    main()
