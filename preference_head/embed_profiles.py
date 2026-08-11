#!/usr/bin/env python3
"""
Generate profile embeddings for a dataset split without clustering.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.lamp_benchmark import load_lamp_dataset

from cluster_profiles import _is_offline, build_profile_text, embed_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed LaMP/LongLaMP profiles")
    parser.add_argument("--task", required=True, help="Task name, e.g., LaMP-1 or LongLaMP-2")
    parser.add_argument("--split", default="dev", help="Dataset split (default: dev)")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_profiles", type=int, default=5, help="Profiles per user to embed")
    parser.add_argument("--max_length", type=int, default=256, help="Max tokens per profile batch")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
    parser.add_argument("--output_file", required=True, help="Path to save embeddings (.npy)")
    parser.add_argument("--meta_file", default=None, help="Optional JSON metadata output")

    args = parser.parse_args()

    offline = _is_offline()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset {args.task} ({args.split})...")
    dataset = load_lamp_dataset(args.task, split=args.split)

    print(f"Loading embedding model {args.embedding_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, local_files_only=offline)
    model = AutoModel.from_pretrained(args.embedding_model, local_files_only=offline).to(device).eval()

    print("Building profile texts...")
    profile_texts = [build_profile_text(sample, args.max_profiles) for sample in dataset]

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

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    print(f"Saved embeddings to: {out_path}")

    if args.meta_file:
        meta_path = Path(args.meta_file)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "task": args.task,
            "split": args.split,
            "embedding_model": args.embedding_model,
            "max_profiles": args.max_profiles,
            "max_length": args.max_length,
            "normalize": args.normalize,
            "num_samples": len(dataset),
            "embeddings_path": str(out_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
