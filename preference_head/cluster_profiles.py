#!/usr/bin/env python3
"""
Cluster LaMP/LongLaMP users by profile embeddings.

Outputs:
  - clusters.json (assignments + centroids)
  - embeddings.npy (optional, for routing reuse)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Add path for LAMP dataset
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.lamp_benchmark import load_lamp_dataset


def _is_offline() -> bool:
    offline_vars = (
        os.environ.get("HF_OFFLINE", ""),
        os.environ.get("HF_HUB_OFFLINE", ""),
        os.environ.get("HF_DATASETS_OFFLINE", ""),
        os.environ.get("TRANSFORMERS_OFFLINE", ""),
    )
    return any(val.strip().lower() in {"1", "true", "yes"} for val in offline_vars)


def _profile_to_text(profile: Dict) -> str:
    if not isinstance(profile, dict):
        return ""

    if "text" in profile and profile["text"]:
        return str(profile["text"])

    parts: List[str] = []
    preferred_keys = (
        "title",
        "abstract",
        "description",
        "content",
        "summary",
        "reviewText",
    )
    for key in preferred_keys:
        if key in profile and profile[key]:
            parts.append(str(profile[key]))
    if not parts:
        for key, value in profile.items():
            if key == "id":
                continue
            if isinstance(value, str) and value.strip():
                parts.append(value)
    return " ".join(parts)


def build_profile_text(sample: Dict, max_profiles: int) -> str:
    profiles = sample.get("profiles", [])
    texts = []
    for prof in profiles[:max_profiles]:
        text = _profile_to_text(prof)
        if text:
            texts.append(text)
    return "\n".join(texts)


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def embed_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    normalize: bool,
) -> np.ndarray:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def kmeans_pp_init(embeddings: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n, dim = embeddings.shape
    centroids = np.empty((k, dim), dtype=np.float32)
    first_idx = rng.integers(0, n)
    centroids[0] = embeddings[first_idx]
    closest_dist_sq = np.sum((embeddings - centroids[0]) ** 2, axis=1)
    for c in range(1, k):
        closest_dist_sq = np.clip(closest_dist_sq, 0.0, None)
        total = float(np.sum(closest_dist_sq))
        if not np.isfinite(total) or total <= 1e-12:
            idx = rng.integers(0, n)
        else:
            probs = closest_dist_sq / total
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs_sum = float(np.sum(probs))
            if probs_sum <= 1e-12:
                idx = rng.integers(0, n)
            else:
                probs = probs / probs_sum
                idx = rng.choice(n, p=probs)
        centroids[c] = embeddings[idx]
        dist_sq = np.sum((embeddings - centroids[c]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
    return centroids


def kmeans(
    embeddings: np.ndarray,
    k: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centroids = kmeans_pp_init(embeddings, k, rng)

    for _ in range(max_iter):
        distances = np.sum((embeddings[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            cluster_points = embeddings[labels == c]
            if len(cluster_points) == 0:
                new_centroids[c] = embeddings[rng.integers(0, embeddings.shape[0])]
            else:
                new_centroids[c] = cluster_points.mean(axis=0)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    return labels, centroids


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster user profiles for LaMP/LongLaMP")
    parser.add_argument("--task", required=True, help="Task name, e.g., LaMP-1 or LongLaMP-2")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_profiles", type=int, default=5, help="Profiles per user to embed")
    parser.add_argument("--max_length", type=int, default=256, help="Max tokens per profile batch")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    offline = _is_offline()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset {args.task} ({args.split})...")
    dataset = load_lamp_dataset(args.task, split=args.split)

    if args.k <= 0 or args.k > len(dataset):
        raise ValueError(f"k must be in [1, {len(dataset)}], got {args.k}")

    print(f"Loading embedding model {args.embedding_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, local_files_only=offline)
    model = AutoModel.from_pretrained(args.embedding_model, local_files_only=offline).to(device).eval()

    print("Building profile texts...")
    profile_texts = [
        build_profile_text(sample, args.max_profiles) for sample in dataset
    ]

    print("Embedding profiles...")
    embeddings = embed_texts(
        profile_texts,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
        args.normalize,
    )
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Clustering into k={args.k} groups...")
    labels, centroids = kmeans(
        embeddings.astype(np.float32),
        args.k,
        args.max_iter,
        args.tol,
        args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_embeddings:
        emb_path = out_dir / "embeddings.npy"
        np.save(emb_path, embeddings)
    else:
        emb_path = None

    cluster_sizes = [int(np.sum(labels == i)) for i in range(args.k)]
    cluster_data = {
        "task": args.task,
        "split": args.split,
        "k": args.k,
        "embedding_model": args.embedding_model,
        "max_profiles": args.max_profiles,
        "max_length": args.max_length,
        "normalize": args.normalize,
        "num_samples": len(dataset),
        "cluster_sizes": cluster_sizes,
        "cluster_assignments": labels.tolist(),
        "centroids": centroids.tolist(),
        "embeddings_path": str(emb_path) if emb_path else None,
    }

    cluster_path = out_dir / "clusters.json"
    with open(cluster_path, "w") as f:
        json.dump(cluster_data, f, indent=2)

    print(f"Saved cluster assignments to: {cluster_path}")
    if emb_path:
        print(f"Saved embeddings to: {emb_path}")


if __name__ == "__main__":
    main()
