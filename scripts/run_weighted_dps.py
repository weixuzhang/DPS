#!/usr/bin/env python3
"""
Weighted DPS with soft routing over profile clusters.

Flow:
  1) Load cluster assignments + centroids
  2) Load per-cluster head weights
  3) For each sample: compute cluster probs -> head weights -> head scales
  4) Run DPS with scaled heads on the depersonalized pass
"""

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.configs import DataConfigs, DecoderConfigs, ModelConfigs
from src.datasets.lamp import LAMP
from src.models.base_model import BaseModel


def _is_offline() -> bool:
    offline_vars = (
        os.environ.get("HF_OFFLINE", ""),
        os.environ.get("HF_HUB_OFFLINE", ""),
        os.environ.get("HF_DATASETS_OFFLINE", ""),
        os.environ.get("TRANSFORMERS_OFFLINE", ""),
    )
    return any(val.strip().lower() in {"1", "true", "yes"} for val in offline_vars)


def _normalize_task(task: str) -> str:
    if task.startswith("LAMP_") or task.startswith("LongLaMP_"):
        return task
    if task.startswith("LaMP-"):
        return f"LAMP_{task.split('-')[1]}"
    if task.startswith("LongLaMP-"):
        return f"LongLaMP_{task.split('-')[1]}"
    raise ValueError(f"Unrecognized task format: {task}")


def _default_data_dir(task: str) -> str:
    env_root = os.environ.get("LAMP_DATA_ROOT")
    if env_root:
        base = Path(env_root)
    else:
        base = REPO_ROOT / "data" / "lamp"
    if task.startswith("LAMP_"):
        return str(base / f"LaMP-{task.split('_')[1]}")
    if task.startswith("LongLaMP_"):
        return str(base / f"LongLaMP-{task.split('_')[1]}")
    return str(base)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


class WeightedDPSModel(BaseModel):
    def __init__(self, model_configs: ModelConfigs, decoder_configs: DecoderConfigs):
        super().__init__(model_configs, decoder_configs)
        self.alpha_cap = decoder_configs.configs.get("alpha_cap")
        self.scale_alpha = decoder_configs.configs.get("scale_alpha", False)
        self.alpha = decoder_configs.configs.get("alpha")

        self._apply_head_scale = False
        self._head_scale = None
        self._head_scale_hooks = []
        self._install_head_scale_hooks()

    def _install_head_scale_hooks(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.layers):
            hook = layer.self_attn.q_proj.register_forward_hook(
                lambda module, inputs, output, idx=layer_idx: self._scale_q_proj(output, idx)
            )
            self._head_scale_hooks.append(hook)

    def _scale_q_proj(self, output: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not self._apply_head_scale or self._head_scale is None:
            return output
        scale = self._head_scale[layer_idx]
        if scale is None:
            return output
        if not torch.is_tensor(output):
            return output

        bsz, seq_len, hidden = output.shape
        num_heads = self.model.config.num_attention_heads
        head_dim = hidden // num_heads
        if head_dim * num_heads != hidden:
            return output

        if not torch.is_tensor(scale):
            scale = torch.tensor(scale, dtype=output.dtype, device=output.device)
        else:
            scale = scale.to(device=output.device, dtype=output.dtype)

        out = output.view(bsz, seq_len, num_heads, head_dim)
        out = out * scale.view(1, 1, num_heads, 1)
        return out.view(bsz, seq_len, hidden)

    def _calculate_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        if self.scale_alpha:
            entropy = entropy / np.log(probs.shape[-1])
        return entropy

    def _set_head_scale(self, head_scale: np.ndarray) -> None:
        self._head_scale = [head_scale[layer_idx] for layer_idx in range(head_scale.shape[0])]

    def _head_scale_context(self, head_scale: np.ndarray):
        class _Context:
            def __init__(self, model: "WeightedDPSModel", scale: np.ndarray):
                self.model = model
                self.scale = scale

            def __enter__(self):
                self.model._set_head_scale(self.scale)
                self.model._apply_head_scale = True

            def __exit__(self, exc_type, exc, tb):
                self.model._apply_head_scale = False

        return _Context(self, head_scale)

    def generate_with_head_scale(self, inputs: Dict, head_scale: np.ndarray) -> Dict:
        prompt = inputs["prompted_question"][0]
        use_system_prompt = bool(inputs["verbalised_instruction"][0])

        tokenised_inputs = self._verbalise_input(
            prompt, use_system_prompt=use_system_prompt
        ).to(self.model.device)

        with torch.inference_mode():
            input_logits = self.model(
                input_ids=tokenised_inputs[:, :-1], use_cache=True, return_dict=True
            )
            generated_ids: List[int] = []
            last_input_token = tokenised_inputs[:, -1]
            base_past_kv = copy.deepcopy(input_logits.past_key_values)
            dep_past_kv = copy.deepcopy(input_logits.past_key_values)
            alphas: List[float] = []

            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)

                base_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=base_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                )

                with self._head_scale_context(head_scale):
                    dep_outputs = self.model(
                        input_ids=last_input_token,
                        past_key_values=dep_past_kv,
                        use_cache=True,
                        attn_mode=self.attn_mode,
                    )

                base_past_kv = base_outputs.past_key_values
                dep_past_kv = dep_outputs.past_key_values

                if self.alpha is None:
                    alpha = self._calculate_entropy(base_outputs.logits[0, -1])
                    alpha_value = alpha.item()
                else:
                    alpha = torch.tensor(
                        self.alpha,
                        device=base_outputs.logits.device,
                        dtype=base_outputs.logits.dtype,
                    )
                    alpha_value = float(self.alpha)
                if self.alpha_cap:
                    alpha = torch.min(
                        alpha, torch.tensor(self.alpha_cap).to(alpha.device)
                    )
                    alpha_value = min(alpha_value, float(self.alpha_cap))
                alphas.append(alpha_value)

                base_logits = base_outputs.logits[0, -1].log_softmax(dim=-1)
                dep_logits = dep_outputs.logits[0, -1].log_softmax(dim=-1)

                next_token_logits = (1 + alpha) * base_logits - alpha * dep_logits
                last_input_token = next_token_logits.argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break

            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {"decoded_text": decoded_text, "alphas": alphas, "attentions": {}}

    def generate(self, inputs, return_attentions: bool = False):
        raise NotImplementedError("Use generate_with_head_scale for weighted DPS.")

    def lm_score(self, logits):
        raise NotImplementedError("Weighted DPS scoring is not implemented.")


class ClusterRouter:
    def __init__(
        self,
        centroids: np.ndarray,
        embeddings: np.ndarray,
        routing: str,
        temperature: float,
    ):
        self.centroids = centroids
        self.embeddings = embeddings
        self.routing = routing
        self.temperature = temperature

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / np.sum(exp)

    def cluster_probs(self, idx: int) -> np.ndarray:
        emb = self.embeddings[idx]
        dists = np.linalg.norm(self.centroids - emb, axis=1)
        if self.routing == "hard":
            probs = np.zeros_like(dists)
            probs[int(np.argmin(dists))] = 1.0
            return probs
        logits = -dists / max(self.temperature, 1e-6)
        return self._softmax(logits)


def load_cluster_weights(cluster_dir: Path, k: int) -> np.ndarray:
    weights = []
    for cluster_id in range(k):
        weight_path = cluster_dir / f"cluster_{cluster_id:02d}" / "head_weights.json"
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing head_weights.json: {weight_path}")
        with open(weight_path, "r") as f:
            payload = json.load(f)
        weights.append(np.array(payload["head_weights"], dtype=np.float32))
    return np.stack(weights, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weighted DPS with soft routing")
    parser.add_argument("--task", required=True, help="LAMP_1, LaMP-1, LongLaMP-1, ...")
    parser.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--model_name", default="LLaMA3-8b-Instruct")
    parser.add_argument("--model_type", default="instruct", choices=["instruct", "base"])
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=32)

    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--num_retrieve", type=int, default=5)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--num_samples", type=int, default=-1)

    parser.add_argument("--cluster_file", required=True)
    parser.add_argument("--cluster_heads_dir", required=True)
    parser.add_argument("--embeddings_file", default=None)
    parser.add_argument("--routing", choices=["soft", "hard"], default="soft")
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--alpha_cap", type=float, default=None)
    parser.add_argument("--scale_alpha", action="store_true")
    parser.add_argument("--alpha", type=float, default=None)

    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--run_dir", default=None)

    args = parser.parse_args()

    task = _normalize_task(args.task)
    data_dir = _default_data_dir(task)

    with open(args.cluster_file, "r") as f:
        cluster_data = json.load(f)
    k = int(cluster_data["k"])
    centroids = np.array(cluster_data["centroids"], dtype=np.float32)

    if args.embeddings_file:
        emb_path = args.embeddings_file
    else:
        emb_path = cluster_data.get("embeddings_path")
    if not emb_path:
        raise ValueError("Embeddings file not found. Provide --embeddings_file or save embeddings in cluster step.")
    embeddings = np.load(emb_path)

    cluster_dir = Path(args.cluster_heads_dir)
    head_weights = load_cluster_weights(cluster_dir, k)

    data_configs = DataConfigs(
        name=task,
        data_dir=data_dir,
        num_samples=args.num_samples,
        variation=None,
        retriever=args.retriever,
        num_retrieve=args.num_retrieve,
        max_prompt_length=args.max_prompt_length,
    )
    dataset = LAMP(
        data_configs,
        model_name_or_path=args.model_path,
        use_chat_template=(args.model_type == "instruct"),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if embeddings.shape[0] != len(dataset):
        raise ValueError(
            f"Embeddings length {embeddings.shape[0]} != dataset length {len(dataset)}"
        )

    model_configs = ModelConfigs(
        name=args.model_name,
        model_type=args.model_type,
        configs={
            "model_name_or_path": args.model_path,
            "max_seq_len": args.max_seq_len,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    decoder_configs = DecoderConfigs(
        name="DPSWeighted",
        method="DPSWeighted",
        configs={
            "alpha_cap": args.alpha_cap,
            "scale_alpha": args.scale_alpha,
            "alpha": args.alpha,
        },
    )
    model = WeightedDPSModel(model_configs, decoder_configs)

    router = ClusterRouter(
        centroids=centroids,
        embeddings=embeddings,
        routing=args.routing,
        temperature=args.temperature,
    )

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        run_dir = Path(args.output_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    method_name = f"DPSWeighted{args.routing.capitalize()}"
    pred_filename = f"pred_{task}_{args.model_name}__{method_name}.json"
    pred_path = run_dir / pred_filename

    print(f"Saving predictions to: {pred_path}")
    print(f"Routing: {args.routing} | k={k}")

    for step, batch in enumerate(dataloader):
        sample_idx = batch.get("idx", [step])[0]
        try:
            sample_idx = int(sample_idx)
        except Exception:
            sample_idx = step

        cluster_probs = router.cluster_probs(sample_idx)
        weighted_heads = np.tensordot(cluster_probs, head_weights, axes=(0, 0))
        head_scale = np.clip(1.0 - weighted_heads, 0.0, 1.0)

        prediction = model.generate_with_head_scale(batch, head_scale)
        batch["predicted_answer"] = prediction["decoded_text"]
        batch["alphas"] = prediction["alphas"]

        batch = _to_serializable(batch)
        with open(pred_path, "a") as f:
            f.write(json.dumps(batch) + "\n")

    print("Done.")
    print(f"Offline mode: {_is_offline()}")


if __name__ == "__main__":
    main()
