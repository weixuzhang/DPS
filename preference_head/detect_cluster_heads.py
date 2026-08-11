#!/usr/bin/env python3
"""
Detect preference heads per cluster and export head weight matrices.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

# Reuse detector without modifying original
from preference_head_detection import PreferenceHeadConfig, PreferenceHeadDetector

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.lamp_benchmark import load_lamp_dataset


class ClusterPreferenceHeadDetector(PreferenceHeadDetector):
    def __init__(self, config: PreferenceHeadConfig, cluster_indices: List[int]):
        self.cluster_indices = cluster_indices
        super().__init__(config)

    def _load_dataset(self):
        dataset = load_lamp_dataset(self.config.task, split=self.config.split)

        dataset = dataset.select(self.cluster_indices)

        if self.config.num_samples > 0 and self.config.num_samples < len(dataset):
            indices = np.linspace(0, len(dataset) - 1, self.config.num_samples, dtype=int)
            dataset = dataset.select(indices.tolist())

        return dataset


def compute_head_weights(
    head_scores: dict,
    num_layers: int,
    num_heads: int,
    norm: str,
    min_pcs: float,
    power: float,
    top_percent: float,
) -> np.ndarray:
    ranked = []
    for (layer, head), scores in head_scores.items():
        avg_pcs = float(np.mean(scores))
        ranked.append((layer, head, avg_pcs))

    ranked.sort(key=lambda x: x[2], reverse=True)

    total_heads = num_layers * num_heads
    keep_heads = int(total_heads * top_percent)
    if keep_heads < 1:
        keep_heads = 1

    keep_set = set((layer, head) for layer, head, _ in ranked[:keep_heads])

    weights = np.zeros((num_layers, num_heads), dtype=np.float32)
    for layer, head, avg_pcs in ranked:
        if (layer, head) not in keep_set:
            continue
        if avg_pcs < min_pcs:
            continue
        weights[layer, head] = max(avg_pcs, 0.0)

    if norm == "max":
        denom = float(weights.max()) if weights.max() > 0 else 1.0
        weights = weights / denom
    elif norm == "sum":
        denom = float(weights.sum()) if weights.sum() > 0 else 1.0
        weights = weights / denom
    elif norm == "none":
        pass
    else:
        raise ValueError(f"Unknown norm: {norm}")

    if power != 1.0:
        weights = np.power(weights, power)

    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-aware preference head detection")
    parser.add_argument("--cluster_file", required=True, help="Path to clusters.json")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--task", required=True, help="LaMP-1, LaMP-2, ... or LongLaMP-1")
    parser.add_argument("--num_samples", type=int, default=200, help="Samples per cluster")
    parser.add_argument("--split", default="train", help="Dataset split for detection (default: train)")
    parser.add_argument("--max_samples_per_cluster", type=int, default=200)
    parser.add_argument("--top_percent", type=float, default=0.04)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--save_dir", required=True, help="Root directory for cluster outputs")

    parser.add_argument("--pcs_norm", default="max", choices=["max", "sum", "none"])
    parser.add_argument("--pcs_min", type=float, default=0.0, help="Minimum PCS to keep")
    parser.add_argument("--pcs_power", type=float, default=1.0, help="Exponent for PCS weights")
    parser.add_argument("--cluster_start", type=int, default=0, help="Start cluster id (inclusive)")
    parser.add_argument("--cluster_end", type=int, default=-1, help="End cluster id (inclusive)")

    args = parser.parse_args()

    with open(args.cluster_file, "r") as f:
        cluster_data = json.load(f)

    assignments = cluster_data["cluster_assignments"]
    k = int(cluster_data["k"])

    cluster_indices = {i: [] for i in range(k)}
    for idx, cid in enumerate(assignments):
        cluster_indices[int(cid)].append(idx)

    if args.cluster_end < 0 or args.cluster_end >= k:
        args.cluster_end = k - 1
    if args.cluster_start < 0 or args.cluster_start >= k:
        raise ValueError(f"cluster_start must be in [0, {k-1}]")
    if args.cluster_start > args.cluster_end:
        raise ValueError("cluster_start must be <= cluster_end")

    root_dir = Path(args.save_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    for cluster_id in range(k):
        if cluster_id < args.cluster_start or cluster_id > args.cluster_end:
            continue
        indices = cluster_indices[cluster_id]
        if not indices:
            print(f"Cluster {cluster_id}: empty, skipping")
            continue

        print(f"\n=== Cluster {cluster_id} ({len(indices)} samples) ===")
        num_samples = min(args.num_samples, args.max_samples_per_cluster, len(indices))

        config = PreferenceHeadConfig(
            model_path=args.model_path,
            task=args.task,
            num_samples=num_samples,
            split=args.split,
            device=args.device,
            torch_dtype=args.dtype,
            top_percent=args.top_percent,
            save_dir=str(root_dir / f"cluster_{cluster_id:02d}"),
        )

        detector = ClusterPreferenceHeadDetector(config, indices)
        detector.detect_preference_heads()
        detector.save_results()

        weights = compute_head_weights(
            detector.head_scores,
            detector.num_layers,
            detector.num_heads,
            args.pcs_norm,
            args.pcs_min,
            args.pcs_power,
            args.top_percent,
        )

        weight_payload = {
            "model": Path(args.model_path).name,
            "task": args.task,
            "cluster_id": cluster_id,
            "num_layers": detector.num_layers,
            "num_heads": detector.num_heads,
            "top_percent": args.top_percent,
            "pcs_norm": args.pcs_norm,
            "pcs_min": args.pcs_min,
            "pcs_power": args.pcs_power,
            "head_weights": weights.tolist(),
        }

        weight_path = Path(config.save_dir) / "head_weights.json"
        with open(weight_path, "w") as f:
            json.dump(weight_payload, f, indent=2)

        npy_path = Path(config.save_dir) / "head_weights.npy"
        np.save(npy_path, weights)

        print(f"Saved head weights to: {weight_path}")


if __name__ == "__main__":
    main()
