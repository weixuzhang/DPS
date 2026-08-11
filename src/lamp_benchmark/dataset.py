# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/data/datasets.py
# and https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/prompts/prompts.py
import json
import os
from pathlib import Path

from datasets import Dataset, DownloadConfig, load_dataset, load_from_disk

from .data_types import Profile


def _is_offline() -> bool:
    """Return True if HF offline mode is enabled via env vars."""
    offline_vars = (
        os.environ.get("HF_OFFLINE", ""),
        os.environ.get("HF_HUB_OFFLINE", ""),
        os.environ.get("HF_DATASETS_OFFLINE", ""),
        os.environ.get("TRANSFORMERS_OFFLINE", ""),
    )
    return any(val.strip().lower() in {"1", "true", "yes"} for val in offline_vars)


def _maybe_limit_dataset_threads() -> None:
    """Optionally force datasets.load_from_disk to use a single thread."""
    if os.environ.get("HF_DATASETS_SINGLE_THREAD", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        return
    try:
        import datasets.arrow_dataset as arrow_dataset
        from tqdm.contrib.concurrent import thread_map as tqdm_thread_map

        def _thread_map_single(fn, *iterables, **tqdm_kwargs):
            tqdm_kwargs.setdefault("max_workers", 1)
            return tqdm_thread_map(fn, *iterables, **tqdm_kwargs)

        arrow_dataset.thread_map = _thread_map_single
    except Exception:
        # Best-effort: fall back to default threading behavior.
        return


def get_dataset_root() -> Path:
    """Return dataset root path (LAMP_DATA_ROOT, defaulting to ./data/lamp)."""
    env_root = os.environ.get("LAMP_DATA_ROOT")
    if env_root:
        return Path(env_root)
    return Path("./data/lamp")


def load_lamp_dataset(task: str, split: str) -> Dataset:
    dataset_root = get_dataset_root()
    dataset_dir = dataset_root / task / split

    if not dataset_dir.exists():
        if task.startswith('LaMP'):
            _process_lamp_dataset(task, split, dataset_root)
        elif task.startswith('LongLaMP'):
            _process_long_lamp_dataset(task, split, dataset_root)
        else:
            raise ValueError(f'Invalid task: {task}')

    _maybe_limit_dataset_threads()
    return load_from_disk(str(dataset_dir))


def _process_lamp_dataset(task: str, split: str, dataset_root: Path) -> None:
    task_dir = dataset_root / task
    questions_file = task_dir / f"{split}_questions.json"
    outputs_file = task_dir / f"{split}_outputs.json"

    with questions_file.open('r') as file:
        questions = json.load(file)

    with outputs_file.open('r') as file:
        outputs = {gold['id']: gold['output'] for gold in json.load(file)['golds']}

    query_corpus_generator = {
        'LaMP-1': _generate_query_corpus_classification_citation,
        'LaMP-2': _generate_query_corpus_classification_movies,
        'LaMP-3': _generate_query_corpus_regression_review,
        'LaMP-4': _generate_query_corpus_generation_news,
        'LaMP-5': _generate_query_corpus_generation_paper,
        'LaMP-6': _generate_query_corpus_generation_avocado,
        'LaMP-7': _generate_query_corpus_generation_tweet
    }[task]
    examples = []

    for question in questions:
        source = question['input']
        profiles = question['profile']
        target = outputs[question['id']]
        query, corpus = query_corpus_generator(source, profiles)

        example = {
            'source': source,
            'profiles': profiles,
            'query': query,
            'corpus': corpus,
            'target': target
        }
        examples.append(example)

    dataset_dir = task_dir / split
    Dataset.from_list(examples).save_to_disk(str(dataset_dir))


def _process_long_lamp_dataset(task: str, split: str, dataset_root: Path) -> None:
    name_map = {
        'LongLaMP-1': None,
        'LongLaMP-2': 'abstract_generation_user',
        'LongLaMP-3': 'topic_writing_user',
        'LongLaMP-4': 'product_review_user',
    }
    name = name_map.get(task)
    if name is None:
        available = ", ".join(
            v for v in name_map.values() if v is not None
        )
        raise ValueError(
            f"{task} is not available in the LongLaMP/LongLaMP dataset configs. "
            f"Available configs: {available}"
        )

    offline = _is_offline()
    download_config = DownloadConfig(local_files_only=offline)
    try:
        dataset = load_dataset(
            'LongLaMP/LongLaMP',
            name=name,
            split=split,
            download_config=download_config,
        )
    except Exception as exc:
        if split == "dev":
            dataset = load_dataset(
                'LongLaMP/LongLaMP',
                name=name,
                split="test",
                download_config=download_config,
            )
            print(f"LongLaMP split 'dev' not found, using 'test' for {task}")
        elif offline:
            raise ConnectionError(
                "HF offline mode is enabled but LongLaMP data is not cached. "
                "Set HF_OFFLINE=false (or HF_DATASETS_OFFLINE=0) to download, "
                f"or pre-download with: load_dataset('LongLaMP/LongLaMP', name='{name}')"
            ) from exc
        else:
            raise

    query_corpus_generator = {
        'LongLaMP-1': _generate_query_corpus_generation_email,
        'LongLaMP-2': _generate_query_corpus_generation_abstract,
        'LongLaMP-3': _generate_query_corpus_generation_topic,
        'LongLaMP-4': _generate_query_corpus_generation_review
    }[task]
    examples = []

    for row in dataset:
        source = row['input']
        profiles = row['profile']
        target = row['output']
        query, corpus = query_corpus_generator(source, profiles)

        example = {
            'source': source,
            'profiles': profiles,
            'query': query,
            'corpus': corpus,
            'target': target
        }
        examples.append(example)

    dataset_dir = dataset_root / task / split
    Dataset.from_list(examples).save_to_disk(str(dataset_dir))


# =============================   LaMP 1: Personalized Citation Identification   =============================
def _generate_query_corpus_classification_citation(source: str, profiles: list[Profile]) -> (
    tuple[str, list[str]]
):
    reference_1, reference_2 = _extract_references(source)
    query = f'{reference_1} {reference_2}'
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


# =============================        LaMP 2: Personalized Movie Tagging        =============================
def _generate_query_corpus_classification_movies(source: str, profiles: list[Profile]) -> (
    tuple[str, list[str]]
):
    query = _extract_string_after_keyword(source, 'description: ')
    corpus = [profile['description'] for profile in profiles]
    return query, corpus


# =============================       LaMP 3: Personalized Product Rating       =============================
def _generate_query_corpus_regression_review(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, 'review: ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


# ==============================  LaMP 4: Personalized News Headline Generation  =============================
def _generate_query_corpus_generation_news(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, 'article: ')
    corpus = [f'{profile["title"]} {profile["text"]}' for profile in profiles]
    return query, corpus


# ============================= LaMP 5: Personalized Scholarly Title Generation =============================
def _generate_query_corpus_generation_paper(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, 'paper: ')
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


# =============================  LaMP 6: Personalized Email Subject Generation  =============================
def _generate_query_corpus_generation_avocado(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


# =============================     LaMP 7: Personalized Tweet Paraphrasing     =============================
def _generate_query_corpus_generation_tweet(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


# ============================     LongLaMP 1: Personalized Email Completion     ============================
def _generate_query_corpus_generation_email(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


# ===========================     LongLaMP 2: Personalized Abstract Generation     ===========================
def _generate_query_corpus_generation_abstract(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, 'items: ')
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


# ============================     LongLaMP 3: Personalized Topic Generation     =============================
def _generate_query_corpus_generation_topic(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    corpus = [f'{profile["content"]} {profile["summary"]}' for profile in profiles]
    return source, corpus


# ========================     LongLaMP 4: Personalized Product Review Generation     ========================
def _generate_query_corpus_generation_review(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    corpus = [
        f'{profile["overall"]} {profile["summary"]} {profile["description"]} {profile["reviewText"]}'
        for profile in profiles
    ]
    return source, corpus


# =============================                Utility Functions                =============================
def _extract_references(source: str) -> tuple[str, str]:
    template_1 = 'Just answer with [1] or [2] without explanation. [1]: "'
    template_2 = '" [2]: "'

    template_1_start = source.find(template_1)
    template_2_start = source.find(template_2)
    assert template_1_start != -1 and template_2_start != -1 and source.endswith('"')

    reference_1 = source[template_1_start + len(template_1) : template_2_start]
    reference_2 = source[template_2_start + len(template_2) : -1]
    return reference_1, reference_2


def _extract_string_after_keyword(source: str, keyword: str) -> str:
    keyword_start = source.find(keyword)

    if keyword_start == -1:
        raise ValueError(f'Keyword "{keyword}" not found in input')

    return source[keyword_start + len(keyword):]
