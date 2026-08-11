#!/usr/bin/env python3
"""
Aggregate per-cluster head weights into a single top-heads file.

This is useful for validation scripts that expect a global top_heads.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate cluster head weights")
    parser.add_argument("--cluster_heads_dir", required=True, help="cluster_heads dir")
    parser.add_argument("--top_percent", type=float, default=0.04)
    parser.add_argument("--output_file", required=True, help="Output top_heads.json")
    args = parser.parse_args()

    cluster_dir = Path(args.cluster_heads_dir)
    cluster_dirs = sorted([p for p in cluster_dir.glob("cluster_*") if p.is_dir()])
    if not cluster_dirs:
        raise FileNotFoundError(f"No cluster_* dirs found in {cluster_dir}")

    weights_list = []
    meta = {}
    for cluster in cluster_dirs:
        weight_path = cluster / "head_weights.json"
        if not weight_path.exists():
            print(f"Skipping {cluster} (missing head_weights.json)")
            continue
        with open(weight_path, "r") as f:
            payload = json.load(f)
        if not meta:
            meta = {
                "model": payload.get("model", ""),
                "task": payload.get("task", ""),
            }
        weights = np.array(payload["head_weights"], dtype=np.float32)
        weights_list.append(weights)

    if not weights_list:
        raise RuntimeError(f"No head weights found in {cluster_dir}")

    avg_weights = np.mean(np.stack(weights_list, axis=0), axis=0)
    num_layers, num_heads = avg_weights.shape
    total_heads = num_layers * num_heads
    num_to_select = int(total_heads * args.top_percent)
    if num_to_select < 1:
        num_to_select = 1

    flat = avg_weights.flatten()
    top_indices = np.argsort(flat)[::-1][:num_to_select]
    top_heads = [[int(idx // num_heads), int(idx % num_heads)] for idx in top_indices]

    output = {
        "model": meta.get("model", ""),
        "task": meta.get("task", ""),
        "top_percent": args.top_percent,
        "num_heads_selected": len(top_heads),
        "preference_heads": top_heads,
        "aggregation": "mean",
        "source_dir": str(cluster_dir),
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved aggregated heads to: {out_path}")


if __name__ == "__main__":
    main()
