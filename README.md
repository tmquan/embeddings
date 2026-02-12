# Nemotron Datasets Embedding Extraction Pipeline

A pipeline for downloading, exploring, and extracting dense embeddings from NVIDIA Nemotron post-training datasets using `nvidia/llama-embed-nemotron-8b` with multi-GPU inference.

Two extraction backends are provided:

| | HuggingFace (`02_embedding_extraction_huggingface.py`) | NeMo Curator (`02_embedding_extraction_nemocurator.py`) |
|---|---|---|
| **Multi-GPU** | Manual `torch.multiprocessing` + LPT scheduling | [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) Pipeline + Ray executor |
| **Model loading** | Direct `AutoModel` / `AutoTokenizer` | Managed by `EmbeddingCreatorStage` |
| **OOM handling** | Automatic batch-size halving | Managed by Curator internals |
| **Output format** | Chunked `.npy` (float32) | Parquet with `embeddings` column |
| **Dependencies** | PyTorch + Transformers only | `nemo-curator[ray]` + Ray |

## Overview

This pipeline processes **8 Nemotron post-training datasets** (~13.2M records, ~380 GB on disk) and extracts 4096-dimensional embeddings using all 8 available GPUs.

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

Downloads 8 post-training datasets from HuggingFace to `/raid/datasets/`.  Already-downloaded datasets are skipped automatically.

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
python 02_embedding_extraction_nemocurator.py --batch-size 512 --max-seq-length 4096
```

<details>
<summary>Sample NeMo Curator pipeline output (13.2M records, 8 GPUs)</summary>

```
2026-02-12 08:46:55.876 | INFO     | ========================================================================
2026-02-12 08:46:55.876 | INFO     | Nemotron Embedding Extraction Pipeline (NeMo Curator)
2026-02-12 08:46:55.876 | INFO     | ========================================================================
2026-02-12 08:46:55.876 | INFO     |   Model             : nvidia/llama-embed-nemotron-8b
2026-02-12 08:46:55.876 | INFO     |   Embedding dim     : 4096
2026-02-12 08:46:55.877 | INFO     |   GPUs              : 8
2026-02-12 08:46:55.877 | INFO     |   Batch size        : 1024
2026-02-12 08:46:55.877 | INFO     |   Max seq length    : 8192 tokens
2026-02-12 08:46:55.877 | INFO     |   Chunk size        : 1,000 records
2026-02-12 08:46:55.877 | INFO     |   Executor          : ray_data
2026-02-12 08:46:55.877 | INFO     |   Datasets dir      : /raid/datasets
2026-02-12 08:46:55.877 | INFO     |   Embeddings dir    : /raid/embeddings_curator
2026-02-12 08:46:55.877 | INFO     |   Dataset filter    : (all)
2026-02-12 08:46:55.877 | INFO     | ------------------------------------------------------------------------
2026-02-12 08:46:55.877 | INFO     | Discovering datasets and files ...
2026-02-12 08:46:55.889 | INFO     |   [Nemotron-3-Nano-RL-Training-Blend/train] 1 file(s), ~592,353 records
2026-02-12 08:46:55.901 | INFO     |   [Nemotron-Science-v1/MCQ] 1 file(s), ~174,530 records
2026-02-12 08:46:55.907 | INFO     |   [Nemotron-Science-v1/RQA] 1 file(s), ~55,409 records
2026-02-12 08:46:55.914 | INFO     |   [Nemotron-Instruction-Following-Chat-v1/chat_if] 1 file(s), ~417,251 records
2026-02-12 08:46:55.921 | INFO     |   [Nemotron-Instruction-Following-Chat-v1/structured_outputs] 1 file(s), ~4,979 records
2026-02-12 08:46:55.929 | INFO     |   [Nemotron-Math-Proofs-v1/lean] 1 file(s), ~1,613,418 records
2026-02-12 08:46:55.935 | INFO     |   [Nemotron-Agentic-v1/tool_calling] 1 file(s), ~325,317 records
2026-02-12 08:46:55.942 | INFO     |   [Nemotron-Agentic-v1/interactive_agent] 1 file(s), ~31,913 records
2026-02-12 08:46:55.949 | INFO     |   [Nemotron-Competitive-Programming-v1/cpp_part0] 1 file(s), ~486,006 records
2026-02-12 08:46:55.956 | INFO     |   [Nemotron-Competitive-Programming-v1/cpp_part1] 1 file(s), ~472,163 records
2026-02-12 08:46:55.962 | INFO     |   [Nemotron-Competitive-Programming-v1/python_part0] 1 file(s), ~891,829 records
2026-02-12 08:46:55.969 | INFO     |   [Nemotron-Competitive-Programming-v1/python_part1] 1 file(s), ~907,526 records
2026-02-12 08:46:55.976 | INFO     |   [Nemotron-Competitive-Programming-v1/infinibyte_part0] 1 file(s), ~574,913 records
2026-02-12 08:46:55.983 | INFO     |   [Nemotron-Competitive-Programming-v1/infinibyte_part1] 1 file(s), ~592,189 records
2026-02-12 08:46:55.990 | INFO     |   [Nemotron-Math-v2/low] 1 file(s), ~1,626,858 records
2026-02-12 08:46:55.997 | INFO     |   [Nemotron-Math-v2/medium] 1 file(s), ~1,762,371 records
2026-02-12 08:46:56.004 | INFO     |   [Nemotron-Math-v2/high_part0] 1 file(s), ~666,099 records
2026-02-12 08:46:56.015 | INFO     |   [Nemotron-Math-v2/high_part1] 1 file(s), ~784,780 records
2026-02-12 08:46:56.025 | INFO     |   [Nemotron-Math-v2/high_part2] 1 file(s), ~1,176,313 records
2026-02-12 08:46:56.032 | INFO     |   [Nemotron-SWE-v1/r2e_gym] 1 file(s), ~51,000 records
2026-02-12 08:46:56.032 | INFO     | Found 20 work units (13,207,217 total records)
2026-02-12 08:46:56.032 | INFO     | ------------------------------------------------------------------------
2026-02-12 08:46:56.032 | INFO     | Initialising ray_data executor with 8 GPUs ...
```

</details>

## Dataset Catalog

All 8 datasets are post-training JSONL format, stored at `/raid/datasets/`.

| Dataset | HuggingFace Repo | Rows | Disk Size | Text Strategy |
|---------|-----------------|------|-----------|---------------|
| Nemotron-Science-v1 | `nvidia/Nemotron-Science-v1` | 226K | 2.5 GB | `messages_list` |
| Nemotron-Instruction-Following-Chat-v1 | `nvidia/Nemotron-Instruction-Following-Chat-v1` | 431K | 6.7 GB | `messages_list` |
| Nemotron-3-Nano-RL-Training-Blend | `nvidia/Nemotron-3-Nano-RL-Training-Blend` | 93K | 6.5 GB | `rl_blend` |
| Nemotron-Math-Proofs-v1 | `nvidia/Nemotron-Math-Proofs-v1` | 1.4M | 28 GB | `math_proof` |
| Nemotron-Agentic-v1 | `nvidia/Nemotron-Agentic-v1` | 335K | 5.4 GB | `agentic` |
| Nemotron-Competitive-Programming-v1 | `nvidia/Nemotron-Competitive-Programming-v1` | 3.9M | 177 GB | `messages_list` |
| Nemotron-Math-v2 | `nvidia/Nemotron-Math-v2` | 7.1M | 142 GB | `math_v2` |
| Nemotron-SWE-v1 | `nvidia/Nemotron-SWE-v1` | 51K | 11 GB | `agentic` |

**Total: ~13.2M records, ~380 GB source data**

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
- **Curator pipeline**: Uses `EmbeddingCreatorStage` with mean pooling and autocast for efficient inference
- **Parquet output**: Embeddings stored in columnar Parquet format for efficient downstream analytics
- **Monkey-patched `trust_remote_code`**: Patches NeMo Curator v1.0 to support custom HF models like `nvidia/llama-embed-nemotron-8b`

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
│   │   ├── embeddings/                # Output: Parquet with "embeddings" column
│   │   │   ├── part_00000.parquet
│   │   │   └── part_00001.parquet
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
| Nemotron-Science-v1 | 226K | ~3.5 GB |
| Nemotron-Instruction-Following-Chat-v1 | 431K | ~6.7 GB |
| Nemotron-3-Nano-RL-Training-Blend | 93K | ~1.5 GB |
| Nemotron-Math-Proofs-v1 | 1.4M | ~22 GB |
| Nemotron-Agentic-v1 | 335K | ~5.2 GB |
| Nemotron-Competitive-Programming-v1 | 3.9M | ~61 GB |
| Nemotron-Math-v2 | 7.1M | ~110 GB |
| Nemotron-SWE-v1 | 51K | ~0.8 GB |
| **Total** | **~13.2M** | **~216 GB** |

## Troubleshooting

### CUDA Out of Memory

- **HuggingFace**: Reduce `--batch-size` (try 4 or 2); the script automatically halves batch size on OOM and retries
- **NeMo Curator**: Reduce `--batch-size` (try 512 or 256) and/or `--max-seq-length` (try 4096)
- Reduce `--max-length` / `--max-seq-length` (try 4096)

### NeMo Curator Only Using 1 GPU

Ray Data autoscales GPU actors between `(1, num_gpus)`.  It starts with 1 actor and only scales up when there are enough data blocks flowing through the pipeline.  If your preprocessed Parquet files are too large (few files → few partitions → few blocks), Ray will never spin up additional actors.

**Fix**: Use a smaller `--chunk-size` to create more preprocessed partitions.  The default is 1,000 rows per file.  If you previously ran with a larger chunk size, regenerate the preprocessed data:

```bash
python 02_embedding_extraction_nemocurator.py --force-preprocess --chunk-size 10000
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

NeMo Curator v1.0 hard-codes `AutoModel.from_pretrained(...)` without `trust_remote_code=True`, and loads the model in float32 instead of bfloat16. Since Ray workers import the package fresh (monkey-patching in the main process has no effect), you must patch the installed source files directly:

```bash
SITE=$(python -c "import nemo_curator; print(nemo_curator.__file__.rsplit('/', 1)[0])")

# 1. EmbeddingModelStage.setup – add trust_remote_code, bfloat16, eager attention
sed -i 's|self.model = AutoModel.from_pretrained(self.model_identifier, local_files_only=True)|self.model = AutoModel.from_pretrained(\n            self.model_identifier,\n            local_files_only=True,\n            trust_remote_code=True,\n            torch_dtype=torch.bfloat16,\n            attn_implementation="eager",\n        )|' \
  "$SITE/stages/text/embedders/base.py"

# 2. TokenizerStage.load_cfg – AutoConfig.from_pretrained
sed -i 's/self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only)/self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only, trust_remote_code=True)/' \
  "$SITE/stages/text/models/tokenizer.py"

# 3. TokenizerStage._setup – AutoTokenizer.from_pretrained  (add after local_files_only line)
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

df = pd.read_parquet("/raid/embeddings_curator/Nemotron-Science-v1/MCQ/embeddings/")
print(len(df))                        # number of embedded records
print(len(df["embeddings"].iloc[0]))  # 4096
```

## License

See `LICENSE` file for dataset and code licensing information.

The embedding model (`nvidia/llama-embed-nemotron-8b`) is released under the customised NSCL v1 license with Meta's Llama 3.1 Community License terms.
