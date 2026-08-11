#!/usr/bin/env python3
"""
Build a manifest for cluster-head detection array jobs.
Each line: task<TAB>k<TAB>cluster_start<TAB>cluster_end
"""

import argparse
from pathlib import Path

from compute_k import compute_k


DEFAULT_TASKS = [
    "LaMP-1",
    "LaMP-2",
    "LaMP-3",
    "LaMP-4",
    "LaMP-5",
    "LaMP-7",
    "LongLaMP-2",
    "LongLaMP-3",
    "LongLaMP-4",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cluster head manifest")
    parser.add_argument("--target_group", type=int, default=100)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument(
        "--output",
        default="preference_head/cluster_head_manifest.tsv",
        help="Output manifest path",
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for task in DEFAULT_TASKS:
        k = compute_k(task, args.split, args.target_group)
        chunks = (k + args.chunk_size - 1) // args.chunk_size
        for chunk_idx in range(chunks):
            start = chunk_idx * args.chunk_size
            end = min(k - 1, start + args.chunk_size - 1)
            lines.append(f"{task}\t{k}\t{start}\t{end}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} jobs to {out_path}")


if __name__ == "__main__":
    main()
