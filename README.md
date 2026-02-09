# Nemotron Datasets Embedding Extraction Pipeline

A pipeline for downloading, exploring, and extracting dense embeddings from NVIDIA Nemotron post-training datasets using `nvidia/llama-embed-nemotron-8b` with multi-GPU inference.

## Overview

This pipeline processes **8 Nemotron post-training datasets** (~13.2M records, ~380 GB on disk) and extracts 4096-dimensional embeddings using all 8 available GPUs.  Embeddings are saved as chunked `.npy` files in float32 format.

| Property | Value |
|----------|-------|
| Model | `nvidia/llama-embed-nemotron-8b` (8B params, bidirectional Llama) |
| Embedding dim | 4096 |
| Output format | `.npy` float32 |
| GPUs | 8 x NVIDIA B300 SXM6 AC (275 GB each) |
| Multi-GPU strategy | `torch.multiprocessing` with LPT scheduling |
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

### 4. Extract Embeddings

```bash
# Dry-run: preview work plan without processing
python 02_embedding_extraction.py --dry-run

# Process all datasets (8 GPUs, default settings)
python 02_embedding_extraction.py

# Process a specific dataset
python 02_embedding_extraction.py --datasets Nemotron-Science-v1

# Tune batch size and sequence length
python 02_embedding_extraction.py --batch-size 4 --max-length 4096
```

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

### `02_embedding_extraction.py`

Extracts embeddings using multi-GPU distributed inference.

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

## Output Format

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

- Reduce `--batch-size` (try 4 or 2)
- Reduce `--max-length` (try 4096)
- The script automatically halves batch size on OOM and retries

### SWE-v1 Slow Processing

SWE-v1 has very long conversations (median ~34K tokens, 55% exceed 32K).  Use a lower `--max-length` to truncate:

```bash
python 02_embedding_extraction.py --datasets SWE --max-length 4096
```

### Resume After Interruption

Simply re-run the same command.  Existing `.npy` chunks are detected and skipped.

### Verifying Output

```python
import numpy as np

emb = np.load("/raid/embeddings/Nemotron-Science-v1/MCQ/embeddings_00000.npy")
print(emb.shape)   # (50000, 4096)
print(emb.dtype)   # float32
print(np.linalg.norm(emb[0]))  # ~1.0 (L2-normalised)
```

## License

See `LICENSE` file for dataset and code licensing information.

The embedding model (`nvidia/llama-embed-nemotron-8b`) is released under the customised NSCL v1 license with Meta's Llama 3.1 Community License terms.
