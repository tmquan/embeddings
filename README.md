# Nemotron Datasets Embedding Extraction Pipeline

A pipeline for downloading, exploring, and extracting dense embeddings from NVIDIA Nemotron post-training datasets using `nvidia/llama-embed-nemotron-8b` with multi-GPU inference.

Two extraction backends are provided:

| | HuggingFace (`02_embedding_extraction_huggingface.py`) | NeMo Curator (`02_embedding_extraction_nemocurator.py`) |
|---|---|---|
| **Multi-GPU** | Manual `torch.multiprocessing` + LPT scheduling | [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) Pipeline + Ray executor |
| **Model loading** | Direct `AutoModel` / `AutoTokenizer` | Managed by `EmbeddingCreatorStage` |
| **OOM handling** | Automatic batch-size halving | Managed by Curator internals |
| **Output format** | Chunked `.npy` (float32) | Parquet with `embeddings` column (float32) |
| **Dependencies** | PyTorch + Transformers only | `nemo-curator[ray]` + Ray |

## Overview

This pipeline processes **11 Nemotron post-training datasets** (~48.2M records) and extracts 4096-dimensional embeddings using all 8 available GPUs. The NeMo Curator backend covers all 11 datasets (25 work units); the first 8 datasets (20 work units, 13,505,137 docs) completed in ~41.2 h at 92.3 docs/s on 2026-02-14.

The three additional datasets added on 2026-02-15:

| Dataset | HuggingFace Repo | Records | Splits |
|---------|-----------------|---------|--------|
| Llama-Nemotron-Post-Training-Dataset | `nvidia/Llama-Nemotron-Post-Training-Dataset` | ~16.2M | SFT (chat, code_v1/v1.1, math_v1/v1.1, safety, science), RL, train |
| Nemotron-Post-Training-Dataset-v1 | `nvidia/Nemotron-Post-Training-Dataset-v1` | ~25.7M | code, math, stem, tool, chat |
| Nemotron-Post-Training-Dataset-v2 | `nvidia/Nemotron-Post-Training-Dataset-v2` | ~6.3M | chat, code, math, stem, multilingual (de/es/fr/it/ja) |

The **embedding reduction** script (`04_embedding_reduction_nemocurator.py`) processes each dataset/split independently on CPU (sklearn t-SNE + umap-learn), so each split is reduced separately and memory is bounded per-split. Resume-safe: existing output is skipped.

| Property | Value |
|----------|-------|
| Model | `nvidia/llama-embed-nemotron-8b` (8B params, bidirectional Llama) |
| Embedding dim | 4096 |
| Output format | `.npy` float32 (HuggingFace) / Parquet (NeMo Curator) |
| GPUs | 8 x NVIDIA B300 SXM6 AC (275 GB each) |
| Multi-GPU strategy | `torch.multiprocessing` + LPT (HuggingFace) / Ray executor (NeMo Curator) |
| Attention | Eager (bidirectional, non-causal) |
| Pooling | Mean pooling with attention mask, L2-normalised |

## Quick Start

### 1. Environment Setup

```bash
conda activate nemotron
pip install -r requirements.txt
```

> PyTorch with CUDA must be installed separately:
> ```bash
> pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 --index-url https://download.pytorch.org/whl/cu130
> ```

### 2. Download Datasets

```bash
python 00_download_nemotron_datasets.py
```

Downloads 11 post-training datasets from HuggingFace to `/raid/datasets/`.  Already-downloaded datasets are skipped automatically.

### 3. Explore Datasets

```bash
python 01_explore_nemotron_datasets.py
```

Inspects each dataset's structure, schema, text-length statistics, and generates a markdown report at `doc/dataset_exploration_summary.md`.

### 4a. Extract Embeddings (HuggingFace — manual multi-GPU)

```bash
# Dry-run: preview work plan without processing
python 02_embedding_extraction_huggingface.py --dry-run

# Process all datasets (8 GPUs, default settings)
python 02_embedding_extraction_huggingface.py

# Process a specific dataset
python 02_embedding_extraction_huggingface.py --datasets Nemotron-Science-v1

# Tune batch size and sequence length
python 02_embedding_extraction_huggingface.py --batch-size 4 --max-length 4096
```

### 4b. Extract Embeddings (NeMo Curator — Ray-distributed pipeline)

```bash
# Dry-run: preview work plan without processing
python 02_embedding_extraction_nemocurator.py --dry-run

# Process all datasets (8 GPUs, default settings)
python 02_embedding_extraction_nemocurator.py

# Process a specific dataset
python 02_embedding_extraction_nemocurator.py --datasets Nemotron-Science-v1

# Tune batch size and sequence length
python 02_embedding_extraction_nemocurator.py --batch-size 16 --max-seq-length 4096
```

<details>
<summary>Sample NeMo Curator pipeline output (discovery, 13.2M records, 8 GPUs)</summary>

```
2026-02-12 08:46:55.876 | INFO     | ========================================================================
2026-02-12 08:46:55.876 | INFO     | Nemotron Embedding Extraction Pipeline (NeMo Curator)
...
2026-02-12 08:46:56.032 | INFO     | Found 20 work units (13,207,217 total records)
2026-02-12 08:46:56.032 | INFO     | Initialising ray_data executor with 8 GPUs ...
```

</details>

<details>
<summary>NeMo Curator pipeline run completion (2026-02-14)</summary>

Full run completed successfully. Final task (Nemotron-SWE-v1/r2e_gym) and summary:

```
2026-02-14 18:57:25.404 | INFO     | nemo_curator.backends.experimental.ray_data.executor:execute:97 - Pipeline completed. Final results: 6 tasks
2026-02-14 18:57:28.084 | INFO     | __main__:_rename_output_files:646 -   Renamed 6 output file(s) → part_XXXXX_of_00006.parquet (matched to input order)
2026-02-14 18:57:28.084 | SUCCESS  | __main__:run_embedding_pipeline:770 - Pipeline completed in 18384.51s – 51,029 documents, 2.8 docs/s
2026-02-14 18:57:28.084 | INFO     | __main__:main:1087 -   Metadata saved → /raid/embeddings_curator/Nemotron-SWE-v1/r2e_gym/metadata.json
2026-02-14 18:57:28.085 | INFO     | __main__:main:1092 - ========================================================================
2026-02-14 18:57:28.085 | INFO     | __main__:main:1093 - Pipeline complete
2026-02-14 18:57:28.085 | INFO     | __main__:main:1094 -   Total wall-clock time : 148355.7s (2472.6 min)
2026-02-14 18:57:28.085 | INFO     | __main__:main:1095 -   Output directory      : /raid/embeddings_curator
2026-02-14 18:57:28.085 | INFO     | __main__:main:1102 -   Results: 20 succeeded, 0 skipped, 0 empty, 0 errors
2026-02-14 18:57:28.085 | INFO     | __main__:main:1123 -   Total documents processed : 13,505,137
2026-02-14 18:57:28.085 | INFO     | __main__:main:1128 -   Aggregate throughput      : 92.3 docs/s
2026-02-14 18:57:28.085 | INFO     | __main__:main:1133 - ========================================================================
```

| Metric | Value |
|--------|-------|
| Work units | 20 succeeded |
| Total documents processed | 13,505,137 |
| Wall-clock time | 148,355.7 s (~2473 min / ~41.2 h) |
| Aggregate throughput | 92.3 docs/s |
| Output directory | `/raid/embeddings_curator` |

</details>

## Dataset Catalog

All 11 datasets are post-training format (JSONL or Parquet), stored at `/raid/datasets/`.

| Dataset | HuggingFace Repo | Rows | Text Strategy |
|---------|-----------------|------|---------------|
| Llama-Nemotron-Post-Training-Dataset | `nvidia/Llama-Nemotron-Post-Training-Dataset` | ~16.2M | `messages_concat` |
| Nemotron-Post-Training-Dataset-v1 | `nvidia/Nemotron-Post-Training-Dataset-v1` | ~25.7M | `messages_list` |
| Nemotron-Post-Training-Dataset-v2 | `nvidia/Nemotron-Post-Training-Dataset-v2` | ~6.3M | `messages_list` |
| Nemotron-Science-v1 | `nvidia/Nemotron-Science-v1` | 226K | `messages_list` |
| Nemotron-Instruction-Following-Chat-v1 | `nvidia/Nemotron-Instruction-Following-Chat-v1` | 431K | `messages_list` |
| Nemotron-3-Nano-RL-Training-Blend | `nvidia/Nemotron-3-Nano-RL-Training-Blend` | 93K | `rl_blend` |
| Nemotron-Math-Proofs-v1 | `nvidia/Nemotron-Math-Proofs-v1` | 1.4M | `math_proof` |
| Nemotron-Agentic-v1 | `nvidia/Nemotron-Agentic-v1` | 335K | `agentic` |
| Nemotron-Competitive-Programming-v1 | `nvidia/Nemotron-Competitive-Programming-v1` | 3.9M | `messages_list` |
| Nemotron-Math-v2 | `nvidia/Nemotron-Math-v2` | 7.1M | `math_v2` |
| Nemotron-SWE-v1 | `nvidia/Nemotron-SWE-v1` | 51K | `agentic` |

**Total: ~48.2M records across 25 work units (11 datasets)**

## Pipeline Scripts

### `00_download_nemotron_datasets.py`

Downloads datasets from HuggingFace using the `hf` CLI to `/raid/datasets/`.

- Automatically skips already-downloaded datasets
- Configurable dataset list (comment/uncomment in script)
- Pretraining datasets available but commented out (~24 TB)

### `01_explore_nemotron_datasets.py`

Analyzes dataset structure and generates extraction strategies.

- Inspects JSONL/parquet schemas and row counts
- Computes text-length statistics (chars, estimated tokens)
- Designs per-dataset text concatenation strategies
- Outputs markdown report to `doc/dataset_exploration_summary.md`

### `02_embedding_extraction_huggingface.py`

Extracts embeddings using manual multi-GPU distributed inference with `torch.multiprocessing`.

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets` | all | Filter by dataset key substring |
| `--datasets-dir` | `/raid/datasets` | Source data directory |
| `--embeddings-dir` | `/raid/embeddings` | Output directory |
| `--num-gpus` | all available (8) | Number of GPUs |
| `--batch-size` | 8 | Batch size per GPU |
| `--max-length` | 8192 | Max tokens (truncation limit) |
| `--chunk-size` | 50,000 | Records per output `.npy` chunk |
| `--dtype` | bfloat16 | Model compute dtype (output is always float32) |
| `--dry-run` | — | Show work plan without processing |

**Key features:**
- **LPT scheduling**: Distributes work units across GPUs using Longest-Processing-Time-first for optimal load balance
- **Resume-safe**: Existing chunk files are detected and skipped automatically
- **OOM recovery**: Batch size is automatically halved on CUDA out-of-memory errors
- **Streaming I/O**: Records are streamed line-by-line, keeping memory bounded

### `02_embedding_extraction_nemocurator.py`

Extracts embeddings using [NVIDIA NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) Pipeline with a Ray executor for automatic GPU distribution.

**Pipeline stages:** `ParquetReader` → `EmbeddingCreatorStage` → `ParquetWriter`

Source datasets (JSONL/Parquet) are first pre-processed into standardised Parquet files with a `text` column, then fed through the Curator pipeline which handles tokenisation, batching, GPU allocation, and embedding generation.

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets` | all | Filter by dataset key substring |
| `--datasets-dir` | `/raid/datasets` | Source data directory |
| `--embeddings-dir` | `/raid/embeddings_curator` | Output directory |
| `--num-gpus` | all available (8) | Number of GPUs |
| `--batch-size` | 8 | Model inference batch size |
| `--max-seq-length` | 8192 | Max sequence length for tokenisation |
| `--chunk-size` | 10,000 | Records per preprocessed Parquet partition |
| `--files-per-partition` | 1 | Input files per pipeline partition |
| `--executor` | ray_data | Executor backend |
| `--dry-run` | — | Show work plan without processing |

**Key features:**
- **Ray-distributed**: NeMo Curator + Ray handle GPU scheduling and data distribution automatically
- **Resume-safe**: Existing preprocessed and embedding directories are detected and skipped
- **Curator pipeline**: Uses `EmbeddingCreatorStage` with mean pooling, bfloat16 compute, and autocast for efficient inference
- **float32 Parquet output**: Embeddings stored as `list<element: float>` in columnar Parquet format (patched from default float64)
- **Indexed output naming**: Output files are renamed from hash-based names to `part_XXXXX_of_XXXXX.parquet`
- **Patched NeMo Curator v1.0**: Installed source files patched for `trust_remote_code=True`, `torch_dtype=bfloat16`, and `attn_implementation="eager"` (see [Troubleshooting](#nemo-curator-trust_remote_code-error))

### `04_embedding_reduction_nemocurator.py`

Reduces NeMo Curator embeddings to 2D/3D with t-SNE and UMAP (scikit-learn + umap-learn, CPU). Processes each dataset/split **independently** so memory is bounded per-split (not all 13M+ at once). Resume-safe: existing output is skipped.

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--embeddings-dir` | `/raid/embeddings_curator` | Root with `{dataset_key}/{sub_label}/embeddings/` |
| `--output-dir` | `/raid/embeddings_reduced` | Output Parquet/CSV per split |
| `--datasets` | all | Filter by dataset key substring |
| `--max-points-for-reduction` | 0 (all rows) | Per-split cap; splits with more rows are subsampled |
| `--tsne-perplexity` | 30 | t-SNE perplexity |
| `--umap-neighbors` | 15 | UMAP n_neighbors |
| `--umap-min-dist` | 0.1 | UMAP min_dist |
| `--no-csv` | — | Skip CSV output |
| `--dry-run` | — | List splits and row counts only |

Output structure mirrors the embeddings directory:

```
/raid/embeddings_reduced/
├── Nemotron-Science-v1/
│   ├── MCQ/
│   │   ├── reduced_2d_3d.parquet
│   │   ├── reduced_2d_3d.csv
│   │   └── metadata.json
│   └── RQA/
│       └── ...
└── ...
```

## Output Format

### HuggingFace backend → chunked `.npy`

Embeddings are saved as chunked NumPy arrays under `/raid/embeddings/`:

```
/raid/embeddings/
├── Nemotron-Science-v1/
│   ├── MCQ/
│   │   ├── embeddings_00000.npy   # [50000, 4096] float32
│   │   ├── embeddings_00001.npy   # [50000, 4096] float32
│   │   ├── embeddings_00002.npy   # [remaining, 4096] float32
│   │   └── metadata.json
│   └── RQA/
│       ├── embeddings_00000.npy
│       └── metadata.json
├── Nemotron-Math-v2/
│   ├── low/
│   ├── medium/
│   ├── high_part0/
│   ├── high_part1/
│   └── high_part2/
└── ...
```

Each `metadata.json` contains:

```json
{
  "dataset_key": "Nemotron-Science-v1",
  "hf_name": "nvidia/Nemotron-Science-v1",
  "sub_label": "MCQ",
  "total_records": 174200,
  "num_chunks": 4,
  "chunk_size": 50000,
  "embedding_dim": 4096,
  "model_id": "nvidia/llama-embed-nemotron-8b",
  "max_length": 8192,
  "dtype": "float32",
  "elapsed_seconds": 1234.5,
  "records_per_second": 141.1
}
```

### NeMo Curator backend → Parquet

Embeddings are saved as Parquet files under `/raid/embeddings_curator/`:

```
/raid/embeddings_curator/
├── Nemotron-Science-v1/
│   ├── MCQ/
│   │   ├── preprocessed/              # Intermediate: Parquet with "text" column
│   │   │   ├── part_000000_of_000018.parquet
│   │   │   ├── part_000001_of_000018.parquet
│   │   │   └── ...
│   │   ├── embeddings/                # Output: Parquet with "embeddings" column (float32)
│   │   │   ├── part_00000_of_00018.parquet
│   │   │   ├── part_00001_of_00018.parquet
│   │   │   └── ...
│   │   └── metadata.json
│   └── RQA/
│       ├── preprocessed/
│       ├── embeddings/
│       └── metadata.json
└── ...
```

```python
import pandas as pd

df = pd.read_parquet("/raid/embeddings_curator/Nemotron-Science-v1/MCQ/embeddings/")
print(df["embeddings"].iloc[0][:5])  # first 5 dims of first embedding
```

## Text Extraction Strategies

Each dataset has a tailored text concatenation strategy:

| Strategy | Datasets | Description |
|----------|----------|-------------|
| `messages_list` | Science, Instruction-Following, Competitive-Programming | Flatten `messages` list: `Role: content` blocks joined by newlines |
| `rl_blend` | Nano-RL-Training-Blend | Extract `input` from nested `responses_create_params` dict, append `ground_truth` |
| `math_proof` | Math-Proofs-v1 | Concatenate `problem` + `formal_statement` (Lean 4) + `messages` |
| `math_v2` | Math-v2 | Prefer `messages`, fallback to `problem` field |
| `agentic` | Agentic-v1, SWE-v1 | Prepend serialised `tools` definitions, then flatten `messages` |

## Disk Space & Time Estimates

**Estimated output sizes** (4096 dim x 4 bytes per record):

| Dataset | Records | Output Size |
|---------|---------|-------------|
| Llama-Nemotron-Post-Training-Dataset | ~16.2M | ~252 GB |
| Nemotron-Post-Training-Dataset-v1 | ~25.7M | ~399 GB |
| Nemotron-Post-Training-Dataset-v2 | ~6.3M | ~98 GB |
| Nemotron-Science-v1 | 226K | ~3.5 GB |
| Nemotron-Instruction-Following-Chat-v1 | 431K | ~6.7 GB |
| Nemotron-3-Nano-RL-Training-Blend | 93K | ~1.5 GB |
| Nemotron-Math-Proofs-v1 | 1.4M | ~22 GB |
| Nemotron-Agentic-v1 | 335K | ~5.2 GB |
| Nemotron-Competitive-Programming-v1 | 3.9M | ~61 GB |
| Nemotron-Math-v2 | 7.1M | ~110 GB |
| Nemotron-SWE-v1 | 51K | ~0.8 GB |
| **Total** | **~48.2M** | **~960 GB** |

## Troubleshooting

### CUDA Out of Memory

- **HuggingFace**: Reduce `--batch-size` (try 4 or 2); the script automatically halves batch size on OOM and retries
- **NeMo Curator**: Reduce `--batch-size` (try 4 or 2) and/or `--max-seq-length` (try 4096); ensure the model loads in bfloat16 (see [patching instructions](#nemo-curator-trust_remote_code-error))

### NeMo Curator Only Using 1 GPU

Ray Data autoscales GPU actors between `(1, num_gpus)`.  It starts with 1 actor and only scales up when there are enough data blocks flowing through the pipeline.  If your preprocessed Parquet files are too large (few files → few partitions → few blocks), Ray will never spin up additional actors.

**Fix**: Use a smaller `--chunk-size` to create more preprocessed partitions.  The default is 10,000 rows per file.  If you previously ran with a larger chunk size, regenerate the preprocessed data:

```bash
python 02_embedding_extraction_nemocurator.py --force-preprocess
```

Rule of thumb: aim for **at least 2–4× more partitions than GPUs** (e.g. ≥16–32 files for 8 GPUs).

### SWE-v1 Slow Processing

SWE-v1 has very long conversations (median ~34K tokens, 55% exceed 32K).  Use a lower sequence length to truncate:

```bash
python 02_embedding_extraction_huggingface.py --datasets SWE --max-length 4096
python 02_embedding_extraction_nemocurator.py --datasets SWE --max-seq-length 4096
```

### NeMo Curator `trust_remote_code` Error

If you see:

```
ValueError: The repository nvidia/llama-embed-nemotron-8b contains custom code which must be executed
to correctly load the model. Please pass the argument `trust_remote_code=True` to allow custom code to be run.
```

NeMo Curator v1.0 has several issues with custom HF models like `nvidia/llama-embed-nemotron-8b`:

- `AutoModel.from_pretrained(...)` called without `trust_remote_code=True`
- Model loads in **float32** instead of bfloat16 (doubles GPU memory)
- Embeddings written as **float64** instead of float32 (50% wasted disk space)

Since Ray workers import the package fresh (monkey-patching in the main process has no effect), you must patch the installed source files directly:

```bash
SITE=$(python -c "import nemo_curator; print(nemo_curator.__file__.rsplit('/', 1)[0])")

# 1. EmbeddingModelStage.setup – trust_remote_code, bfloat16, eager attention
sed -i 's|self.model = AutoModel.from_pretrained(self.model_identifier, local_files_only=True)|self.model = AutoModel.from_pretrained(\n            self.model_identifier,\n            local_files_only=True,\n            trust_remote_code=True,\n            torch_dtype=torch.bfloat16,\n            attn_implementation="eager",\n        )|' \
  "$SITE/stages/text/embedders/base.py"

# 2. EmbeddingModelStage.collect_outputs – float32 instead of float64
#    Add "import numpy as np" to imports
sed -i '/^import torch$/i import numpy as np' "$SITE/stages/text/embedders/base.py"
#    Replace .tolist() with .float().numpy() to keep float32
sed -i 's/return torch.cat(processed_outputs, dim=0).numpy().tolist()/return torch.cat(processed_outputs, dim=0).float().numpy()/' \
  "$SITE/stages/text/embedders/base.py"
#    Update create_output_dataframe to split 2D array into list of 1D arrays
sed -i 's/return df_cpu.assign(\*\*{self.embedding_field: collected_output})/embeddings_list = [collected_output[i] for i in range(len(collected_output))]\n        return df_cpu.assign(**{self.embedding_field: embeddings_list})/' \
  "$SITE/stages/text/embedders/base.py"

# 3. TokenizerStage.load_cfg – AutoConfig.from_pretrained
sed -i 's/self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only)/self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only, trust_remote_code=True)/' \
  "$SITE/stages/text/models/tokenizer.py"

# 4. TokenizerStage._setup – AutoTokenizer.from_pretrained
sed -i '/local_files_only=local_files_only,/{/trust_remote_code/!s/local_files_only=local_files_only,/local_files_only=local_files_only,\n            trust_remote_code=True,/}' \
  "$SITE/stages/text/models/tokenizer.py"
```

### Resume After Interruption

Simply re-run the same command.  Both backends detect and skip existing outputs.

### Verifying Output

**HuggingFace backend (`.npy`):**

```python
import numpy as np

emb = np.load("/raid/embeddings/Nemotron-Science-v1/MCQ/embeddings_00000.npy")
print(emb.shape)   # (50000, 4096)
print(emb.dtype)   # float32
print(np.linalg.norm(emb[0]))  # ~1.0 (L2-normalised)
```

**NeMo Curator backend (Parquet):**

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("/raid/embeddings_curator/Nemotron-Science-v1/MCQ/embeddings/")
print(len(df))                        # number of embedded records
print(len(df["embeddings"].iloc[0]))  # 4096
emb = np.array(df["embeddings"].iloc[0])
print(emb.dtype)                      # float32
print(np.linalg.norm(emb))           # ~1.0 (L2-normalised)
```

## License

See `LICENSE` file for dataset and code licensing information.

The embedding model (`nvidia/llama-embed-nemotron-8b`) is released under the customised NSCL v1 license with Meta's Llama 3.1 Community License terms.
