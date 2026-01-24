# Nemotron Datasets Embedding Extraction Pipeline

A comprehensive pipeline for downloading, exploring, and extracting embeddings from NVIDIA Nemotron datasets using multi-GPU processing.

## Overview

This pipeline processes **21 Nemotron datasets** (12 post-training + 9 pretraining) and extracts embeddings using the `nvidia/llama-embed-nemotron-8b` model with distributed multi-GPU inference.

## Features

- ✅ **21 Datasets**: Full Nemotron collection (post-training + pretraining)
- ✅ **Multi-GPU**: Distributed processing with Ray
- ✅ **Auto-Detection**: Smart format detection (conversational vs. pretraining)
- ✅ **Shard-Preserving**: Maintains original dataset shard structure
- ✅ **Scalable**: Handles datasets from 27k to 8.79B samples
- ✅ **Flash Attention 2**: Optimized inference (with SDPA fallback)

## Quick Start

### 0. Setup Environment (First Time Only)

See [SETUP.md](SETUP.md) for detailed environment setup instructions.

**Quick setup:**
```bash
# Using existing embeddings environment
conda activate embeddings

# Install/update compatible versions
./check_and_fix_versions.sh

# Or manually install
pip install -r requirements.txt
```

### 1. Check Dataset Status

```bash
python 01_verify_nemotron_dataset_status.py
```

Shows which datasets are already downloaded and their sizes.

### 2. Download Datasets

```bash
python 00_download_nemotron_datasets.py
```

**Start small** with the sample dataset first to test the pipeline!

### 3. Explore Datasets

```bash
python 01_explore_nemotron_datasets.py
```

Generates:
- `dataset_exploration_summary.json` - Quick overview
- `dataset_exploration_detailed.json` - Full details with samples
- `embedding_extraction_functions.py` - Auto-generated extraction functions

### 4. Extract Embeddings

```bash
# Test with sample dataset (27.7k samples) - RECOMMENDED FIRST
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-Dataset-sample \
  --num-gpus 8 \
  --batch-size 8

# Process all datasets (WARNING: requires TB of disk space and days of processing!)
python 02_extract_nemotron_embeddings.py --num-gpus 8 --batch-size 8
```

## Dataset Catalog

### Post-Training Datasets (12)

| Dataset | Key | Samples | Description |
|---------|-----|---------|-------------|
| Nemotron-Post-Training-Dataset-v1 | `v1` | ~100k | Version 1 |
| Nemotron-Post-Training-Dataset-v2 | `v2` | ~200k | Version 2 |
| Llama-Nemotron (SFT) | `llama-sft` | ~300k | Supervised fine-tuning |
| Llama-Nemotron (RL) | `llama-rl` | ~100k | Reinforcement learning |
| Nemotron-Science-v1 | `v3-science` | ~150k | Scientific domain |
| Nemotron-Instruction-Following-Chat-v1 | `v3-instruction-chat` | ~200k | Instruction following |
| Nemotron-Math-Proofs-v1 | `v3-math-proofs` | ~50k | Mathematical proofs |
| Nemotron-3-Nano-RL-Training-Blend | `v3-rl-blend` | ~80k | RL training blend |
| Nemotron-Agentic-v1 | `v3-agentic` | ~100k | Agentic tasks |
| Nemotron-Competitive-Programming-v1 | `v3-competitive-programming` | ~60k | Programming contests |
| Nemotron-Math-v2 | `v3-math` | ~180k | Mathematics v2 |
| Nemotron-SWE-v1 | `v3-swe` | ~120k | Software engineering |

### Pretraining Datasets (9)

| Dataset | Key | Samples | Description |
|---------|-----|---------|-------------|
| Nemotron-Pretraining-Dataset-sample | `pretrain-sample` | 27.7k | **Sample for testing** |
| Nemotron-CC-Code-v1 | `pretrain-cc-code-v1` | 216M | Code from Common Crawl |
| Nemotron-CC-v2.1 | `pretrain-cc-v2.1` | 3.8B | Common Crawl v2.1 |
| Nemotron-Pretraining-Code-v2 | `pretrain-code-v2` | 836M | Code pretraining v2 |
| Nemotron-Pretraining-Specialized-v1 | `pretrain-specialized-v1` | 60.7M | Specialized domains |
| Nemotron-CC-Math-v1 | `pretrain-cc-math-v1` | 190M | Math content |
| Nemotron-CC-v2 | `pretrain-cc-v2` | 8.79B | **Largest dataset!** |
| Nemotron-Pretraining-SFT-v1 | `pretrain-sft-v1` | 299M | SFT pretraining |
| Nemotron-Pretraining-Code-v1 | `pretrain-code-v1` | 936M | Code pretraining v1 |

## Dataset-Specific Extraction Commands

### Post-Training Datasets

#### Nemotron v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Post-Training-Dataset-v1 \
  --num-gpus 8 --batch-size 8
```

#### Nemotron v2
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Post-Training-Dataset-v2 \
  --num-gpus 8 --batch-size 8
```

#### Llama-Nemotron (SFT)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Llama-Nemotron-Post-Training-Dataset \
  --splits SFT \
  --num-gpus 8 --batch-size 8
```

#### Llama-Nemotron (RL)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Llama-Nemotron-Post-Training-Dataset \
  --splits RL \
  --num-gpus 8 --batch-size 8
```

#### Nemotron Science v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Science-v1 \
  --num-gpus 8 --batch-size 8
```

#### Nemotron Instruction-Following Chat v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Instruction-Following-Chat-v1 \
  --num-gpus 8 --batch-size 8
```

#### Nemotron Math-Proofs v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Math-Proofs-v1 \
  --num-gpus 8 --batch-size 8
```

#### Nemotron 3 Nano RL Training Blend
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-3-Nano-RL-Training-Blend \
  --num-gpus 8 --batch-size 8
```

#### Nemotron Agentic v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Agentic-v1 \
  --num-gpus 8 --batch-size 8
```

#### Nemotron Competitive Programming v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Competitive-Programming-v1 \
  --num-gpus 8 --batch-size 8
```

#### Nemotron Math v2
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Math-v2 \
  --num-gpus 8 --batch-size 8
```

#### Nemotron SWE v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-SWE-v1 \
  --num-gpus 8 --batch-size 8
```

### Pretraining Datasets

#### Pretraining Sample (⭐ START HERE - 27.7k samples)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-Dataset-sample \
  --num-gpus 8 --batch-size 8
```

#### Common Crawl Code v1 (216M samples)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-CC-Code-v1 \
  --num-gpus 8 --batch-size 8
```

#### Common Crawl v2.1 (3.8B samples - LARGE!)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-CC-v2.1 \
  --num-gpus 8 --batch-size 8 \
  --chunk-size 10000
```

#### Pretraining Code v2 (836M samples)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-Code-v2 \
  --num-gpus 8 --batch-size 8
```

#### Pretraining Specialized v1 (60.7M samples)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-Specialized-v1 \
  --num-gpus 8 --batch-size 8
```

#### Common Crawl Math v1 (190M samples)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-CC-Math-v1 \
  --num-gpus 8 --batch-size 8
```

#### Common Crawl v2 (8.79B samples - LARGEST!)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-CC-v2 \
  --num-gpus 8 --batch-size 8 \
  --chunk-size 10000
```

#### Pretraining SFT v1 (299M samples)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-SFT-v1 \
  --num-gpus 8 --batch-size 8
```

#### Pretraining Code v1 (936M samples)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-Code-v1 \
  --num-gpus 8 --batch-size 8
```

### Batch Processing Multiple Datasets

#### All Post-Training Datasets
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets \
  nvidia/Nemotron-Post-Training-Dataset-v1 \
  nvidia/Nemotron-Post-Training-Dataset-v2 \
  nvidia/Llama-Nemotron-Post-Training-Dataset \
  nvidia/Nemotron-Science-v1 \
  nvidia/Nemotron-Instruction-Following-Chat-v1 \
  nvidia/Nemotron-Math-Proofs-v1 \
  nvidia/Nemotron-3-Nano-RL-Training-Blend \
  nvidia/Nemotron-Agentic-v1 \
  nvidia/Nemotron-Competitive-Programming-v1 \
  nvidia/Nemotron-Math-v2 \
  nvidia/Nemotron-SWE-v1 \
  --num-gpus 8 --batch-size 8
```

#### Small Pretraining Datasets (Sample + Specialized)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-Dataset-sample \
            nvidia/Nemotron-Pretraining-Specialized-v1 \
  --num-gpus 8 --batch-size 8
```

#### Medium Pretraining Datasets (Code + Math)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-CC-Code-v1 \
            nvidia/Nemotron-CC-Math-v1 \
            nvidia/Nemotron-Pretraining-SFT-v1 \
  --num-gpus 8 --batch-size 8
```

#### Large Pretraining Datasets (Code v1 & v2)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-Code-v1 \
            nvidia/Nemotron-Pretraining-Code-v2 \
  --num-gpus 8 --batch-size 8
```

#### Very Large Common Crawl Datasets (WARNING: Multi-day processing!)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-CC-v2 \
            nvidia/Nemotron-CC-v2.1 \
  --num-gpus 8 --batch-size 8 \
  --chunk-size 10000
```

## Pipeline Scripts

### `00_download_nemotron_datasets.py`

Downloads datasets from HuggingFace to `/raid/datasets`.

**Features:**
- Automatic skip if already downloaded
- Progress logging
- Error handling and retry
- Summary report

**Configuration:**
- Edit `DATASETS_DIR` in script to change download location
- Comment out datasets you don't need

### `01_explore_nemotron_datasets.py`

Analyzes dataset structures and generates extraction strategies.

**What it does:**
- Scans all splits and columns
- Identifies text columns
- Detects conversational formats
- Determines optimal extraction strategy
- Generates custom extraction functions

**Outputs:**
- `dataset_exploration_summary.json`
- `dataset_exploration_detailed.json`
- `embedding_extraction_functions.py`

### `02_extract_nemotron_embeddings.py`

Extracts embeddings using multi-GPU distributed processing.

**Key Parameters:**
- `--datasets`: Filter specific datasets
- `--splits`: Filter specific splits
- `--num-gpus`: Number of GPUs (default: all available)
- `--batch-size`: Batch size per GPU (default: 8)
- `--chunk-size`: Samples per shard (default: 10000)
- `--dtype`: Model precision (bfloat16/float16/float32)
- `--no-flash-attention`: Disable Flash Attention 2

**Example:**
```bash
# Process specific datasets with 8 GPUs
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Science-v1 \
            nvidia/Nemotron-Math-v2 \
  --num-gpus 8 \
  --batch-size 8 \
  --dtype bfloat16
```

## Configuration (`config.py`)

Central configuration for all scripts:

```python
# Model configuration
MODEL_ID_DEFAULT = "nvidia/llama-embed-nemotron-8b"
MODEL_EMBED_SIZE = 4096
MODEL_MAX_TOKENS = 32768

# Storage paths
DEFAULT_DATASETS_DIR = "/raid/datasets"
DEFAULT_EMBEDDINGS_DIR = "/raid/embeddings"

# Processing defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_DTYPE = "bfloat16"
```

## Output Format

Embeddings are saved in NumPy format matching the original dataset shard structure:

```
/raid/embeddings/
├── pretraining/
│   └── sample/
│       └── train/
│           ├── data-00000-of-00003.npy  # [N, 4096] float32
│           ├── data-00001-of-00003.npy
│           ├── data-00002-of-00003.npy
│           └── embedding_metadata.json
└── nemotron-v3/
    └── science/
        └── train/
            ├── data-00000-of-00010.npy
            └── ...
```

Each `embedding_metadata.json` contains:
```json
{
  "dataset_name": "nvidia/Nemotron-Science-v1",
  "split_name": "train",
  "num_samples": 150000,
  "num_shards": 10,
  "embedding_dim": 4096,
  "model_id": "nvidia/llama-embed-nemotron-8b"
}
```

## Disk Space Requirements

| Dataset Type | Estimated Size |
|-------------|----------------|
| Sample | ~100 MB |
| Small (60-200M) | 5-20 GB |
| Medium (300-900M) | 30-100 GB |
| Large (3.8B) | 500-800 GB |
| Very Large (8.79B) | 1-2 TB |

**Total for all datasets: ~3-4 TB**

## Processing Time (8 x H100 GPUs)

| Dataset Size | Estimated Time |
|-------------|----------------|
| 27.7k | ~5 min |
| 60M | ~2-4 hours |
| 300M | ~8-12 hours |
| 3.8B | ~100+ hours |
| 8.79B | ~250+ hours |

## Best Practices

1. **Start Small**: Test with `Nemotron-Pretraining-Dataset-sample` first
2. **Monitor Resources**: Check disk space and GPU memory
3. **Process Strategically**: Prioritize important datasets
4. **Use Checkpointing**: Scripts resume from existing shards
5. **Validate Output**: Check embedding quality on samples

## Data Formats

### Post-Training (Conversational)
```json
{
  "messages": [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is..."}
  ]
}
```

### Pretraining (Simple Text)
```json
{
  "text": "The quick brown fox jumps over the lazy dog...",
  "metadata": {"domain": "wikipedia", "quality_score": 0.95}
}
```

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (try 1)
- Use `--dtype float16` or `bfloat16`
- Process smaller datasets first

### Slow Processing
- Enable Flash Attention 2 (requires compatible GPU)
- Increase `--num-gpus`
- Adjust `--chunk-size` for optimal I/O

### Disk Space Issues
- Process datasets selectively using `--datasets`
- Clean up intermediate files
- Use external storage for large datasets

## Recent Updates

**January 24, 2026**: Added support for 9 pretraining datasets
- See [`doc/PRETRAINING_DATASETS_UPDATE.md`](doc/PRETRAINING_DATASETS_UPDATE.md) for details
- Enhanced extraction logic for simple text formats
- Improved auto-detection capabilities

## License

See `LICENSE` file for dataset and code licensing information.

## Citation

If you use these embeddings, please cite the original Nemotron datasets:

```bibtex
@misc{nvidia-nemotron-2025,
  title={Nemotron Training Datasets},
  author={NVIDIA},
  year={2025},
  url={https://huggingface.co/collections/nvidia}
}
```

## Additional Documentation

### Setup and Getting Started
- [`SETUP.md`](SETUP.md) - **Environment setup guide** (Python, conda, pip, dependencies)
- [`requirements.txt`](requirements.txt) - Python package dependencies (compatible versions)

### Dataset Guides
- [`doc/POST_TRAINING_COMMANDS.md`](doc/POST_TRAINING_COMMANDS.md) - **Complete guide for Post-Training datasets** (download, verify, explore, extract)
- [`doc/QUICK_REFERENCE.md`](doc/QUICK_REFERENCE.md) - Quick command reference and workflow
- [`doc/README.md`](doc/README.md) - Dataset distribution overview with sizes in rows, bytes, chars, tokens

### Technical Documentation
- [`doc/PRETRAINING_DATASETS_UPDATE.md`](doc/PRETRAINING_DATASETS_UPDATE.md) - Pretraining datasets update details
- [`doc/CHANGES_SUMMARY.md`](doc/CHANGES_SUMMARY.md) - Complete change log
- [`doc/SHARD_FORMAT.md`](doc/SHARD_FORMAT.md) - Shard format documentation
- [`doc/00_download_nemotron_datasets.md`](doc/00_download_nemotron_datasets.md) - Download script documentation
- [`doc/01_explore_nemotron_datasets.md`](doc/01_explore_nemotron_datasets.md) - Exploration script documentation

## Support

For issues or questions:
1. Check [`doc/PRETRAINING_DATASETS_UPDATE.md`](doc/PRETRAINING_DATASETS_UPDATE.md) for recent changes
2. Review [`doc/QUICK_REFERENCE.md`](doc/QUICK_REFERENCE.md) for troubleshooting tips
3. Review exploration outputs for dataset-specific details
4. Examine extraction logs for processing errors
