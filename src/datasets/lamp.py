import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.lamp_benchmark import load_lamp_dataset, create_prompt_generator
from src.configs import DataConfigs
from src.datasets.base_dataset import BaseDataset
from transformers import AutoTokenizer


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


class LAMP(BaseDataset):
    """
    LAMP dataset with RAG-based profile retrieval.
    
    Uses the lamp_benchmark retrieval system to select the most relevant user profiles
    instead of using all profiles, following LAMP benchmark's intended approach.
    
    Retrieval options (set via data_configs or kwargs):
    - 'bm25': Fast token-based retrieval (no GPU needed, recommended)
    - 'contriever': Dense retrieval with facebook/contriever model (better quality)
    - 'first_k': No retrieval, just use first k profiles
    - 'random': Random selection (for ablation)
    """
    
    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs
    ):
        super().__init__(data_configs, **kwargs)
        
        # LAMP task configuration
        # Convert LAMP_1 to LaMP-1 or LongLaMP_1 to LongLaMP-1
        if data_configs.name.startswith("LongLaMP_"):
            self.task = data_configs.name.replace("LongLaMP_", "LongLaMP-")
        else:
            self.task = data_configs.name.replace("LAMP_", "LaMP-")
        self.split = "dev"  # Default to dev split for evaluation
        
        # RAG configuration - get from data_configs or kwargs
        self.retriever_type = getattr(data_configs, 'retriever', None) or kwargs.get('retriever', 'bm25')
        self.num_retrieve = getattr(data_configs, 'num_retrieve', None) or kwargs.get('num_retrieve', 5)
        self.max_prompt_length = getattr(data_configs, 'max_prompt_length', None) or kwargs.get('max_seq_len', 2048)
        
        # Get model name for tokenizer
        model_name = kwargs.get('model_name_or_path') or 'meta-llama/Meta-Llama-3-8B-Instruct'
        offline = _is_offline()
        
        print(f"Initializing LAMP dataset with RAG:")
        print(f"  Task: {self.task}")
        print(f"  Retriever: {self.retriever_type}")
        print(f"  Num profiles to retrieve: {self.num_retrieve}")
        print(f"  Max prompt length: {self.max_prompt_length} tokens")
        
        # Load LAMP dataset and initialize RAG components
        try:
            # Load dataset
            self.dataset = load_lamp_dataset(self.task, self.split)
            print(f"Loaded {self.task} dataset with {len(self.dataset)} examples")

            # Initialize tokenizer for prompt generation
            # Offline mode is controlled by HF_OFFLINE environment variable (set in run scripts)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, local_files_only=offline
                )
            except OSError as exc:
                if offline:
                    _raise_offline_model_error(model_name, exc)
                raise

            # Create prompt generator with RAG retrieval
            self.prompt_generator = create_prompt_generator(
                task=self.task,
                retriever=self.retriever_type,
                num_retrieve=self.num_retrieve,
                max_length=self.max_prompt_length,
                tokenizer=self.tokenizer
            )
            print(f"Initialized {self.retriever_type} retriever successfully")
        except Exception as e:
            print(f"Failed to load LAMP dataset {self.task}: {e}")
            raise e
            
        # Parse data for BaseDataset compatibility
        self.data = self.parse_data()

    def parse_data(self):
        """Convert LAMP dataset to DeCoRe format with RAG-based profile retrieval"""
        data = []
        
        for idx, item in enumerate(self.dataset):
            # Extract components from the LaMP dataset
            source = item['source']
            profiles = item['profiles']
            query = item.get('query', '')
            corpus = item.get('corpus', [])
            target = item['target']

            # Use RAG to generate prompt with retrieved profiles
            # This automatically:
            # 1. Retrieves top-k most relevant profiles using the configured retriever
            # 2. Truncates each profile to fit within token budget
            # 3. Formats according to LAMP task specification
            try:
                instruction = self.prompt_generator(
                    source=source,
                    profiles=profiles,
                    query=query,
                    corpus=corpus
                )
            except Exception as e:
                print(f"Warning: Failed to generate RAG prompt for sample {idx}: {e}")
                print(f"Falling back to simple truncation for this sample")
                # Fallback to basic approach if RAG fails
                instruction = self._get_instruction_fallback(self.task, profiles, query, corpus)

            # Context is integrated into instruction
            context = ""

            # Determine answer prefix based on task type
            answer_prefix = self._get_answer_prefix(self.task)

            data.append(
                {
                    "idx": idx,
                    "instruction": instruction,
                    "instruction_wo_context": source,
                    "icl_demo": "",  # No in-context learning demos
                    "contexts": context,
                    "question": query,
                    "answer_prefix": answer_prefix,
                    "answers": [target],
                    "task": self.task
                }
            )
            
            if self.num_samples != -1 and idx + 1 >= self.num_samples:
                break
                
        return data

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Build the prompt following DeCORE format
        instruction = sample["instruction"]
        icl_demo = sample["icl_demo"]
        contexts = sample["contexts"]
        question = sample["question"]
        answer_prefix = sample["answer_prefix"]
        
        # Construct the full prompt
        verbalised_instruction = instruction
        verbalised_icl_demo = icl_demo
        verbalised_contexts = contexts
        verbalised_question = question
        verbalised_answer_prefix = answer_prefix
        
        # The profile-free prompt is the raw task source (no retrieved user
        # profiles), used by contrast methods such as CAD as the "no context"
        # branch. Falls back to the full instruction for older cached rows.
        instruction_wo_context = sample.get("instruction_wo_context", instruction)

        # Create the prompted question based on whether using chat template
        if self.kwargs.get("use_chat_template", False):
            # For chat template, use list format
            prompted_question = [[instruction]]
            prompted_question_wo_context = [[instruction_wo_context]]
        else:
            # For regular format, concatenate all parts
            prompted_question = instruction
            prompted_question_wo_context = instruction_wo_context

        # Return the sample in the expected format
        sample_output = {
            "verbalised_instruction": verbalised_instruction,
            "verbalised_icl_demo": verbalised_icl_demo,
            "verbalised_contexts": verbalised_contexts,
            "verbalised_question": verbalised_question,
            "verbalised_answer_prefix": verbalised_answer_prefix,
            "prompted_question": prompted_question,
            "prompted_question_wo_context": prompted_question_wo_context,
        }
        
        # Add original sample data for metrics
        sample_output.update(sample)
        
        return sample_output
    
    def __len__(self):
        return len(self.data)

    def _get_instruction_fallback(self, task: str, profiles: List[Dict], query: str, corpus: List[str]) -> str:
        """
        Fallback instruction generation with simple truncation.
        Used if RAG retrieval fails for any reason.
        """
        MAX_PROFILE_TOKENS = 2048
        CHARS_PER_TOKEN = 4
        max_profile_chars = MAX_PROFILE_TOKENS * CHARS_PER_TOKEN
        
        # Collect all profile texts
        profile_texts = []
        for p in profiles:
            if task == "LaMP-1":
                text = p.get('title', '')
            elif task == "LaMP-2":
                text = p.get('description', '')
            elif task == "LaMP-3":
                text = p.get('text', '')
            elif task == "LaMP-4":
                text = f"{p.get('title', '')} {p.get('text', '')}"
            else:
                text = p.get('text', '')
            if text:
                profile_texts.append(text)
        
        full_profile_text = "\n".join(profile_texts)
        
        # Truncate if exceeds limit
        if len(full_profile_text) > max_profile_chars:
            truncated_text = full_profile_text[:max_profile_chars]
            last_period = truncated_text.rfind('.')
            if last_period > max_profile_chars * 0.8:
                truncated_text = truncated_text[:last_period + 1]
            profile_text = truncated_text + "\n[... profile truncated ...]"
        else:
            profile_text = full_profile_text
        
        if profile_text:
            profile_section = f"User Profile:\n{profile_text}\n\n"
        else:
            profile_section = ""

        if task == "LaMP-1":
            return f"{profile_section}Given the following references, which one is more relevant to the query?\nQuery: {query}\nReferences:\n{corpus[0]}\n{corpus[1]}\nAnswer:"
        elif task == "LaMP-2":
            return f"{profile_section}Predict the tags for the following movie description:\nDescription: {query}\nAnswer:"
        elif task == "LaMP-3":
            return f"{profile_section}Predict the rating (1-5) for the following review:\nReview: {query}\nAnswer:"
        elif task == "LaMP-4":
            return f"{profile_section}Generate a news headline for the following article:\nArticle: {query}\nAnswer:"
        else:
            return f"{profile_section}Task: {task}\nInput: {query}\nAnswer:"

    def _get_answer_prefix(self, task: str) -> str:
        """Get the expected answer prefix based on the task."""
        if task == "LaMP-1":
            return "["
        elif task == "LaMP-2":
            return "" # Tags can be comma separated
        elif task == "LaMP-3":
            return "" # Rating is a number
        elif task == "LaMP-4":
            return "" # Generated text
        else:
            return ""
