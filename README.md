# Differential Preference Steering (DPS)

Official implementation of **Differential Preference Steering (DPS)**. DPS
detects preference-sensitive attention heads and steers them during decoding to
personalize LLM outputs.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Datasets are not included. Point `LAMP_DATA_ROOT` at a directory with one
subdirectory per task (`LaMP-1/` … `LongLaMP-4/`), or run
`python scripts/prefetch_lamp_datasets.py` to download the raw files. LaMP
loading, prompting, retrieval, and metrics are implemented in
`src/lamp_benchmark/`.

## Usage

Run a Hydra-configured evaluation:

```bash
python scripts/main.py data=lamp_1 model=llama3_8b_instruct decoder=dps
```

Build preference-head artifacts:

```bash
python preference_head/cluster_profiles.py --task LaMP-1 --split dev --k 25 \
  --output_dir artifacts/cluster_runs/lamp1_k25 --save_embeddings
python preference_head/preference_head_detection.py --task LaMP-1 --split dev \
  --save_dir artifacts/preference_heads
```

Run weighted cluster-routed DPS:

```bash
python scripts/run_weighted_dps.py --task LaMP-1 \
  --cluster_file artifacts/cluster_runs/lamp1_k25/clusters.json \
  --cluster_heads_dir artifacts/cluster_heads/lamp1_k25 \
  --embeddings_file artifacts/cluster_runs/lamp1_k25/embeddings.npy
```

## Layout

- `src/` — datasets, metrics, model wrappers, and decoding methods
  (`src/lamp_benchmark/` holds the LaMP utilities).
- `preference_head/` — profile clustering and preference-head detection.
- `configs/` — Hydra configs for data, models, and decoders.
- `scripts/` — entry points for evaluation, weighted DPS, and data prefetch.
