#!/usr/bin/env python3
"""
Preference Head Detection for LLMs

This script detects attention heads that causally encode and inject user-specific 
preferences (style, tone, vocabulary) into model outputs. It uses the Preference
Contribution Score (PCS) method described in preference_head.md.

Adapted from retrieval_head_detection.py to detect preference heads instead of
retrieval heads.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

# Add paths for LAMP dataset
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.lamp_benchmark import load_lamp_dataset


def _is_offline() -> bool:
    """Return True if HF offline mode is enabled via env vars."""
    offline_vars = (
        os.environ.get("HF_OFFLINE", ""),
        os.environ.get("HF_HUB_OFFLINE", ""),
        os.environ.get("HF_DATASETS_OFFLINE", ""),
        os.environ.get("TRANSFORMERS_OFFLINE", ""),
    )
    return any(val.strip().lower() in {"1", "true", "yes"} for val in offline_vars)


def _raise_offline_model_error(model_path: str, exc: Exception) -> None:
    raise RuntimeError(
        f"HF offline mode is enabled but model '{model_path}' is not in cache. "
        "Set HF_OFFLINE=false (or HF_HUB_OFFLINE=0) to download, "
        "or pre-download into the cache."
    ) from exc


@dataclass
class PreferenceHeadConfig:
    """Configuration for preference head detection."""
    model_path: str
    task: str = "LaMP-1"  # LaMP task to use
    split: str = "train"  # Dataset split for detection
    num_samples: int = 400  # Number of samples for detection
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 32
    temperature: float = 0.7
    top_k: int = 1  # Top-k heads to track per generation step
    save_dir: str = "./preference_scores"
    
    # Scoring method
    score_method: str = "nll"  # "nll" (negative log-likelihood) or "style" (style similarity)
    
    # Head selection
    top_percent: float = 0.04  # Top 4% of heads


class PreferenceHeadDetector:
    """
    Detects preference heads in transformer models using ablation and activation patching.
    
    The detection process:
    1. Compute baseline personalization score with full model
    2. For each head, ablate (zero out) and recompute score
    3. Calculate Preference Contribution Score (PCS) = baseline - ablated_score
    4. Rank heads by PCS and select top candidates
    """
    
    def __init__(self, config: PreferenceHeadConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.offline = _is_offline()
        
        # Load model and tokenizer
        print(f"Loading model from {config.model_path}...")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        self.dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_path, local_files_only=self.offline
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=self.dtype,
                device_map="auto",
                attn_implementation="eager",  # Need eager for attention manipulation
                local_files_only=self.offline,
            )
        except OSError as exc:
            if self.offline:
                _raise_offline_model_error(config.model_path, exc)
            raise
        self.model.eval()
        
        # Get model dimensions
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        
        print(f"Model loaded: {self.num_layers} layers, {self.num_heads} heads per layer")
        
        # Initialize score tracking
        self.head_scores = defaultdict(list)  # {(layer, head): [scores]}
        self.baseline_scores = []
        
        # Load dataset
        print(f"Loading {config.task} dataset...")
        self.dataset = self._load_dataset()
        print(f"Loaded {len(self.dataset)} samples")
        
    def _load_dataset(self):
        """Load LaMP dataset for preference detection."""
        dataset = load_lamp_dataset(self.config.task, split=self.config.split)
        
        # Limit to num_samples
        if self.config.num_samples > 0 and self.config.num_samples < len(dataset):
            # Sample uniformly across the dataset
            indices = np.linspace(0, len(dataset) - 1, self.config.num_samples, dtype=int)
            dataset = dataset.select(indices.tolist())
        
        return dataset
    
    def _format_prompt(self, sample: Dict, include_profile: bool = True) -> str:
        """
        Format a prompt from a LaMP sample.
        
        Args:
            sample: LaMP dataset sample
            include_profile: Whether to include user profile (clean) or not (corrupted)
        
        Returns:
            Formatted prompt string
        """
        source = sample['source']
        query = sample.get('query', '')
        
        if include_profile:
            profiles = sample['profiles']
            # Use first 5 profiles for context (to keep it manageable)
            profile_text = ""
            for i, prof in enumerate(profiles[:5]):
                if 'text' in prof:
                    profile_text += f"History {i+1}: {prof['text']}\n"
                elif 'title' in prof:
                    profile_text += f"History {i+1}: {prof['title']}\n"
            
            prompt = f"{profile_text}\n{source}"
        else:
            # Corrupted: no profile
            prompt = source
        
        return prompt
    
    def _compute_personalization_score(
        self, 
        prompt: str, 
        target: str,
        ablate_head: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        Compute personalization score for a prompt-target pair.
        
        Args:
            prompt: Input prompt
            target: Target completion
            ablate_head: Optional (layer, head) tuple to ablate
        
        Returns:
            Personalization score (negative log-likelihood)
        """
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        target_ids = self.tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        # Prepare full input
        full_ids = torch.cat([input_ids, target_ids], dim=1)
        
        # Forward pass with optional ablation
        with torch.no_grad():
            if ablate_head is not None:
                # Register hook to ablate specific head
                hooks = []
                layer_idx, head_idx = ablate_head
                
                def ablation_hook(module, args, output):
                    # output is tuple (hidden_states, attention_weights, ...)
                    # For LLaMA, o_proj output is (batch, seq_len, hidden_size)
                    # We need to reshape and zero out the specific head
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    batch, seq_len, hidden_size = hidden_states.shape
                    head_dim = hidden_size // self.num_heads
                    
                    # Reshape to (batch, seq_len, num_heads, head_dim)
                    hidden_states = hidden_states.view(batch, seq_len, self.num_heads, head_dim)
                    
                    # Zero out the target head
                    hidden_states[:, :, head_idx, :] = 0
                    
                    # Reshape back
                    hidden_states = hidden_states.view(batch, seq_len, hidden_size)
                    
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    return hidden_states
                
                # Hook the o_proj of the target layer
                target_layer = self.model.model.layers[layer_idx]
                hook = target_layer.self_attn.o_proj.register_forward_hook(ablation_hook)
                hooks.append(hook)
                
                try:
                    outputs = self.model(full_ids, labels=full_ids)
                finally:
                    # Remove hooks
                    for h in hooks:
                        h.remove()
            else:
                outputs = self.model(full_ids, labels=full_ids)
            
            # Compute NLL for target tokens only
            logits = outputs.logits
            shift_logits = logits[:, input_ids.shape[1]-1:-1, :].contiguous()
            shift_labels = target_ids.contiguous()
            
            # Compute cross-entropy loss
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            
            # Return negative log-likelihood (lower is better)
            return loss.item()
    
    def compute_baseline_score(self, sample_idx: int) -> float:
        """Compute baseline personalization score with full model."""
        sample = self.dataset[sample_idx]
        prompt = self._format_prompt(sample, include_profile=True)
        target = sample['target']
        
        score = self._compute_personalization_score(prompt, target)
        return score
    
    def compute_ablation_score(
        self, 
        sample_idx: int, 
        layer_idx: int, 
        head_idx: int
    ) -> float:
        """Compute personalization score with specific head ablated."""
        sample = self.dataset[sample_idx]
        prompt = self._format_prompt(sample, include_profile=True)
        target = sample['target']
        
        score = self._compute_personalization_score(
            prompt, 
            target, 
            ablate_head=(layer_idx, head_idx)
        )
        return score
    
    def compute_pcs(
        self, 
        baseline_score: float, 
        ablated_score: float
    ) -> float:
        """
        Compute Preference Contribution Score (PCS).
        
        PCS = ablated_score - baseline_score
        
        Higher PCS means the head is important for personalization
        (removing it increases loss/decreases personalization)
        """
        return ablated_score - baseline_score
    
    def detect_preference_heads(self):
        """
        Main detection loop using head-wise ablation.
        
        For efficiency, we use a two-stage approach:
        1. Stage 1: Coarse detection on subset of samples
        2. Stage 2: Fine-grained analysis on top candidates
        """
        print("\n" + "="*80)
        print("Stage 1: Baseline Score Computation")
        print("="*80)
        
        # Stage 1: Compute baseline scores
        baseline_scores = []
        num_samples = min(self.config.num_samples, len(self.dataset))
        
        for i in tqdm(range(num_samples), desc="Computing baselines"):
            try:
                score = self.compute_baseline_score(i)
                baseline_scores.append((i, score))
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue
        
        self.baseline_scores = baseline_scores
        avg_baseline = np.mean([s[1] for s in baseline_scores])
        print(f"Average baseline NLL: {avg_baseline:.4f}")
        
        print("\n" + "="*80)
        print("Stage 2: Head-wise Ablation (PCS Computation)")
        print("="*80)
        print(f"Testing {self.num_layers * self.num_heads} heads across {num_samples} samples...")
        print("This will take a while...\n")
        
        # Stage 2: Ablate each head and compute PCS
        # For efficiency, sample a subset of examples per head
        samples_per_head = min(50, num_samples)
        sample_indices = np.random.choice(num_samples, samples_per_head, replace=False)
        
        for layer_idx in range(self.num_layers):
            print(f"\nLayer {layer_idx}/{self.num_layers}")
            
            for head_idx in tqdm(range(self.num_heads), desc=f"  Heads"):
                pcs_scores = []
                
                for sample_idx in sample_indices:
                    try:
                        baseline_score = baseline_scores[sample_idx][1]
                        ablated_score = self.compute_ablation_score(
                            sample_idx, layer_idx, head_idx
                        )
                        pcs = self.compute_pcs(baseline_score, ablated_score)
                        pcs_scores.append(pcs)
                    except Exception as e:
                        print(f"Error on layer {layer_idx}, head {head_idx}, sample {sample_idx}: {e}")
                        continue
                
                # Store average PCS for this head
                if pcs_scores:
                    self.head_scores[(layer_idx, head_idx)] = pcs_scores
        
        print("\n" + "="*80)
        print("Detection Complete!")
        print("="*80)
    
    def rank_heads(self) -> List[Tuple[Tuple[int, int], float]]:
        """
        Rank heads by average PCS.
        
        Returns:
            List of ((layer, head), avg_pcs) sorted by PCS descending
        """
        ranked = []
        for (layer, head), scores in self.head_scores.items():
            avg_pcs = np.mean(scores)
            ranked.append(((layer, head), avg_pcs))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def get_top_preference_heads(self, top_percent: Optional[float] = None) -> List[List[int]]:
        """
        Get top preference heads.
        
        Args:
            top_percent: Percentage of heads to select (default from config)
        
        Returns:
            List of [layer, head] pairs
        """
        if top_percent is None:
            top_percent = self.config.top_percent
        
        ranked = self.rank_heads()
        num_to_select = int(len(ranked) * top_percent)
        
        top_heads = [list(head) for head, _ in ranked[:num_to_select]]
        return top_heads
    
    def save_results(self):
        """Save detection results to disk."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Get model name
        model_name = Path(self.config.model_path).name
        task_name = self.config.task.replace("-", "_")
        
        # Save full head scores
        scores_file = Path(self.config.save_dir) / f"{model_name}_{task_name}_pcs.json"
        
        # Convert to serializable format
        scores_dict = {
            f"{layer}-{head}": scores 
            for (layer, head), scores in self.head_scores.items()
        }
        
        with open(scores_file, 'w') as f:
            json.dump(scores_dict, f, indent=2)
        
        print(f"\n💾 Full PCS scores saved to: {scores_file}")
        
        # Save ranked heads
        ranked = self.rank_heads()
        ranked_file = Path(self.config.save_dir) / f"{model_name}_{task_name}_ranked.json"
        
        ranked_dict = {
            "model": model_name,
            "task": self.config.task,
            "num_samples": self.config.num_samples,
            "ranked_heads": [
                {
                    "layer": layer,
                    "head": head,
                    "avg_pcs": float(avg_pcs),
                    "rank": i + 1
                }
                for i, ((layer, head), avg_pcs) in enumerate(ranked)
            ]
        }
        
        with open(ranked_file, 'w') as f:
            json.dump(ranked_dict, f, indent=2)
        
        print(f"💾 Ranked heads saved to: {ranked_file}")
        
        # Save top preference heads (for easy loading in DPS)
        top_heads = self.get_top_preference_heads()
        top_file = Path(self.config.save_dir) / f"{model_name}_{task_name}_top_heads.json"
        
        top_dict = {
            "model": model_name,
            "task": self.config.task,
            "top_percent": self.config.top_percent,
            "num_heads_selected": len(top_heads),
            "preference_heads": top_heads
        }
        
        with open(top_file, 'w') as f:
            json.dump(top_dict, f, indent=2)
        
        print(f"💾 Top preference heads saved to: {top_file}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print detection summary."""
        ranked = self.rank_heads()
        top_heads = self.get_top_preference_heads()
        
        print("\n" + "="*80)
        print("Preference Head Detection Summary")
        print("="*80)
        print(f"Model: {Path(self.config.model_path).name}")
        print(f"Task: {self.config.task}")
        print(f"Samples analyzed: {self.config.num_samples}")
        print(f"Total heads: {len(ranked)}")
        print(f"Top {self.config.top_percent*100:.1f}% heads selected: {len(top_heads)}")
        
        print(f"\n🏆 Top 10 Preference Heads:")
        print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Avg PCS':<12}")
        print("-" * 40)
        for i, ((layer, head), avg_pcs) in enumerate(ranked[:10], 1):
            print(f"{i:<6} {layer:<8} {head:<8} {avg_pcs:>10.4f}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Detect preference heads in LLMs")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model")
    parser.add_argument("--task", type=str, default="LaMP-1",
                       help="LaMP task to use (LaMP-1 to LaMP-7)")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split for detection (default: train)")
    parser.add_argument("--num_samples", type=int, default=400,
                       help="Number of samples for detection")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Model dtype")
    
    # Detection arguments
    parser.add_argument("--top_percent", type=float, default=0.04,
                       help="Percentage of top heads to select (default: 0.04 = 4%%)")
    parser.add_argument("--save_dir", type=str, default="./preference_scores",
                       help="Directory to save results")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=32,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    
    args = parser.parse_args()
    
    # Create config
    config = PreferenceHeadConfig(
        model_path=args.model_path,
        task=args.task,
        num_samples=args.num_samples,
        split=args.split,
        device=args.device,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_percent=args.top_percent,
        save_dir=args.save_dir
    )
    
    # Initialize detector
    detector = PreferenceHeadDetector(config)
    
    # Run detection
    detector.detect_preference_heads()
    
    # Save results
    detector.save_results()


if __name__ == "__main__":
    main()
