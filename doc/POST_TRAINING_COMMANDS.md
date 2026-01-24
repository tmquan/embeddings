# Post-Training Datasets - Common Commands

Quick reference for working with **Post-Training datasets only** (12 datasets, ~1.64M rows, ~4.3 GB total).

## Overview

Post-Training datasets contain conversational/instruction-following data in multi-turn message format. These are smaller, high-quality datasets ideal for fine-tuning and evaluation.

### Post-Training Dataset List (12)

1. `nvidia/Nemotron-Post-Training-Dataset-v1` (~100k rows, ~200 MB)
2. `nvidia/Nemotron-Post-Training-Dataset-v2` (~200k rows, ~400 MB)
3. `nvidia/Llama-Nemotron-Post-Training-Dataset` (SFT: ~300k, RL: ~100k, ~800 MB)
4. `nvidia/Nemotron-Science-v1` (~150k rows, ~450 MB)
5. `nvidia/Nemotron-Instruction-Following-Chat-v1` (~200k rows, ~500 MB)
6. `nvidia/Nemotron-Math-Proofs-v1` (~50k rows, ~250 MB)
7. `nvidia/Nemotron-3-Nano-RL-Training-Blend` (~80k rows, ~160 MB)
8. `nvidia/Nemotron-Agentic-v1` (~100k rows, ~300 MB)
9. `nvidia/Nemotron-Competitive-Programming-v1` (~60k rows, ~240 MB)
10. `nvidia/Nemotron-Math-v2` (~180k rows, ~540 MB)
11. `nvidia/Nemotron-SWE-v1` (~120k rows, ~480 MB)

**Total**: ~1.64M rows, ~4.3 GB

---

## 1. Download Post-Training Datasets

### Download All Post-Training Datasets

Edit `00_download_nemotron_datasets.py` to include only post-training datasets:

```python
DATASETS = [
    # Post-Training Datasets
    "nvidia/Nemotron-Post-Training-Dataset-v1",
    "nvidia/Nemotron-Post-Training-Dataset-v2",
    "nvidia/Llama-Nemotron-Post-Training-Dataset",
    "nvidia/Nemotron-Science-v1",
    "nvidia/Nemotron-Instruction-Following-Chat-v1",
    "nvidia/Nemotron-Math-Proofs-v1",
    "nvidia/Nemotron-3-Nano-RL-Training-Blend",
    "nvidia/Nemotron-Agentic-v1",
    "nvidia/Nemotron-Competitive-Programming-v1",
    "nvidia/Nemotron-Math-v2",
    "nvidia/Nemotron-SWE-v1",
]
```

Then run:
```bash
python 00_download_nemotron_datasets.py
```

### Download Individual Post-Training Datasets

You can also use HuggingFace CLI directly:

```bash
# Nemotron v1
hf download nvidia/Nemotron-Post-Training-Dataset-v1 \
  --repo-type dataset \
  --local-dir /raid/datasets/nvidia_Nemotron-Post-Training-Dataset-v1

# Nemotron v2
hf download nvidia/Nemotron-Post-Training-Dataset-v2 \
  --repo-type dataset \
  --local-dir /raid/datasets/nvidia_Nemotron-Post-Training-Dataset-v2

# Llama-Nemotron (includes both SFT and RL)
hf download nvidia/Llama-Nemotron-Post-Training-Dataset \
  --repo-type dataset \
  --local-dir /raid/datasets/nvidia_Llama-Nemotron-Post-Training-Dataset

# Science v1
hf download nvidia/Nemotron-Science-v1 \
  --repo-type dataset \
  --local-dir /raid/datasets/nvidia_Nemotron-Science-v1

# Instruction-Following Chat v1
hf download nvidia/Nemotron-Instruction-Following-Chat-v1 \
  --repo-type dataset \
  --local-dir /raid/datasets/nvidia_Nemotron-Instruction-Following-Chat-v1

# Math-Proofs v1
hf download nvidia/Nemotron-Math-Proofs-v1 \
  --repo-type dataset \
  --local-dir /raid/datasets/nvidia_Nemotron-Math-Proofs-v1

# 3-Nano RL Training Blend
hf download nvidia/Nemotron-3-Nano-RL-Training-Blend \
  --repo-type dataset \
  --local-dir /raid/datasets/nvidia_Nemotron-3-Nano-RL-Training-Blend

# Agentic v1
hf download nvidia/Nemotron-Agentic-v1 \
  --repo-type dataset \
  --local-dir /raid/datasets/nvidia_Nemotron-Agentic-v1

# Competitive Programming v1
hf download nvidia/Nemotron-Competitive-Programming-v1 \
  --repo-type dataset \
  --local-dir /raid/datasets/nvidia_Nemotron-Competitive-Programming-v1

# Math v2
hf download nvidia/Nemotron-Math-v2 \
  --repo-type dataset \
  --local-dir /raid/datasets/nvidia_Nemotron-Math-v2

# SWE v1
hf download nvidia/Nemotron-SWE-v1 \
  --repo-type dataset \
  --local-dir /raid/datasets/nvidia_Nemotron-SWE-v1
```

---

## 2. Verify Downloaded Datasets

### Check Status of Post-Training Datasets

```bash
python 01_verify_nemotron_dataset_status.py
```

This will show which post-training datasets are downloaded, their sizes, and split information.

### Manual Verification

```bash
# List all downloaded datasets
ls -lh /raid/datasets/ | grep -E "(Nemotron|Llama-Nemotron)" | grep -v "Pretraining\|CC-"

# Check specific dataset structure
ls -lh /raid/datasets/nvidia_Nemotron-Science-v1/

# Count total size of post-training datasets
du -sh /raid/datasets/nvidia_Nemotron-* /raid/datasets/nvidia_Llama-Nemotron-* 2>/dev/null | grep -v "Pretraining\|CC-"
```

---

## 3. Explore Post-Training Datasets

### Explore All Post-Training Datasets

```bash
python 01_explore_nemotron_datasets.py
```

This generates:
- `dataset_exploration_summary.json` - Quick statistics
- `dataset_exploration_detailed.json` - Full details with samples
- `embedding_extraction_functions.py` - Auto-generated extraction functions

### Filter Exploration Results for Post-Training Only

```bash
# View summary for post-training datasets
python -c "
import json
with open('dataset_exploration_summary.json', 'r') as f:
    data = json.load(f)
post_training = [d for d in data if 'Pretraining' not in d['name'] and 'CC-' not in d['name']]
print(json.dumps(post_training, indent=2))
" | less
```

### Quick Dataset Inspection

```bash
# View first few records of a dataset (Python)
python -c "
from datasets import load_from_disk
ds = load_from_disk('/raid/datasets/nvidia_Nemotron-Science-v1/train')
print(f'Columns: {ds.column_names}')
print(f'Num samples: {len(ds)}')
print(f'First sample: {ds[0]}')
"
```

---

## 4. Extract Embeddings - All Post-Training Datasets

### Extract All Post-Training Datasets (One Command)

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
  --num-gpus 8 \
  --batch-size 8
```

**Estimated time**: ~2-4 hours on 8 GPUs (total ~1.64M samples)

---

## 5. Extract Embeddings - Individual Datasets

### General Datasets

#### Nemotron v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Post-Training-Dataset-v1 \
  --num-gpus 8 --batch-size 8
```
**Est. time**: ~15 minutes (100k samples)

#### Nemotron v2
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Post-Training-Dataset-v2 \
  --num-gpus 8 --batch-size 8
```
**Est. time**: ~30 minutes (200k samples)

#### Llama-Nemotron (Both SFT and RL)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Llama-Nemotron-Post-Training-Dataset \
  --num-gpus 8 --batch-size 8
```
**Est. time**: ~1 hour (400k samples total)

#### Llama-Nemotron SFT Only
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Llama-Nemotron-Post-Training-Dataset \
  --splits SFT \
  --num-gpus 8 --batch-size 8
```

#### Llama-Nemotron RL Only
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Llama-Nemotron-Post-Training-Dataset \
  --splits RL \
  --num-gpus 8 --batch-size 8
```

### Domain-Specific Datasets

#### Science v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Science-v1 \
  --num-gpus 8 --batch-size 8
```
**Est. time**: ~25 minutes (150k samples)

#### Instruction-Following Chat v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Instruction-Following-Chat-v1 \
  --num-gpus 8 --batch-size 8
```
**Est. time**: ~30 minutes (200k samples)

#### Math-Proofs v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Math-Proofs-v1 \
  --num-gpus 8 --batch-size 8
```
**Est. time**: ~10 minutes (50k samples)

#### Math v2
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Math-v2 \
  --num-gpus 8 --batch-size 8
```
**Est. time**: ~30 minutes (180k samples)

#### 3-Nano RL Training Blend
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-3-Nano-RL-Training-Blend \
  --num-gpus 8 --batch-size 8
```
**Est. time**: ~15 minutes (80k samples)

### Task-Specific Datasets

#### Agentic v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Agentic-v1 \
  --num-gpus 8 --batch-size 8
```
**Est. time**: ~15 minutes (100k samples)

#### Competitive Programming v1
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Competitive-Programming-v1 \
  --num-gpus 8 --batch-size 8
```
**Est. time**: ~12 minutes (60k samples)

#### SWE v1 (Software Engineering)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-SWE-v1 \
  --num-gpus 8 --batch-size 8
```
**Est. time**: ~20 minutes (120k samples)

---

## 6. Grouped Extraction by Category

### Extract All Math-Related Datasets
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets \
  nvidia/Nemotron-Math-Proofs-v1 \
  nvidia/Nemotron-Math-v2 \
  --num-gpus 8 --batch-size 8
```
**Total**: ~230k samples, ~45 minutes

### Extract All Code-Related Datasets
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets \
  nvidia/Nemotron-Competitive-Programming-v1 \
  nvidia/Nemotron-SWE-v1 \
  --num-gpus 8 --batch-size 8
```
**Total**: ~180k samples, ~35 minutes

### Extract All v3 Datasets
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets \
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
**Total**: ~1.04M samples, ~2.5 hours

### Extract Foundation Datasets (v1, v2, Llama)
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets \
  nvidia/Nemotron-Post-Training-Dataset-v1 \
  nvidia/Nemotron-Post-Training-Dataset-v2 \
  nvidia/Llama-Nemotron-Post-Training-Dataset \
  --num-gpus 8 --batch-size 8
```
**Total**: ~600k samples, ~1.5 hours

---

## 7. Batch Processing Script

### Create a Shell Script for Sequential Processing

```bash
#!/bin/bash
# process_post_training.sh

set -e

DATASETS=(
  "nvidia/Nemotron-Post-Training-Dataset-v1"
  "nvidia/Nemotron-Post-Training-Dataset-v2"
  "nvidia/Llama-Nemotron-Post-Training-Dataset"
  "nvidia/Nemotron-Science-v1"
  "nvidia/Nemotron-Instruction-Following-Chat-v1"
  "nvidia/Nemotron-Math-Proofs-v1"
  "nvidia/Nemotron-3-Nano-RL-Training-Blend"
  "nvidia/Nemotron-Agentic-v1"
  "nvidia/Nemotron-Competitive-Programming-v1"
  "nvidia/Nemotron-Math-v2"
  "nvidia/Nemotron-SWE-v1"
)

NUM_GPUS=8
BATCH_SIZE=8

echo "Processing ${#DATASETS[@]} post-training datasets..."

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "=========================================="
  echo "Processing: $dataset"
  echo "=========================================="
  
  python 02_extract_nemotron_embeddings.py \
    --datasets "$dataset" \
    --num-gpus $NUM_GPUS \
    --batch-size $BATCH_SIZE
  
  echo "✓ Completed: $dataset"
done

echo ""
echo "=========================================="
echo "All post-training datasets processed!"
echo "=========================================="
```

Usage:
```bash
chmod +x process_post_training.sh
./process_post_training.sh
```

---

## 8. Verify Embeddings Output

### Check Embeddings Directory Structure

```bash
# List all post-training embeddings
ls -lh /raid/embeddings/nemotron-* /raid/embeddings/llama-nemotron 2>/dev/null

# Check specific dataset embeddings
ls -lh /raid/embeddings/nemotron-v3/science/

# View embedding metadata
cat /raid/embeddings/nemotron-v3/science/train/embedding_metadata.json | python -m json.tool
```

### Verify Embedding Counts

```bash
# Python script to verify embeddings match dataset size
python -c "
import json
import numpy as np
from pathlib import Path

embedding_dir = Path('/raid/embeddings/nemotron-v3/science/train')
metadata = json.loads((embedding_dir / 'embedding_metadata.json').read_text())

print(f'Dataset: {metadata[\"dataset_name\"]}')
print(f'Expected samples: {metadata[\"num_samples\"]}')
print(f'Number of shards: {metadata[\"num_shards\"]}')

# Count actual embeddings
total = 0
for shard_file in sorted(embedding_dir.glob('*.npy')):
    embeddings = np.load(shard_file)
    total += len(embeddings)
    print(f'  {shard_file.name}: {len(embeddings)} embeddings')

print(f'Total embeddings: {total}')
print(f'Match: {\"✓\" if total == metadata[\"num_samples\"] else \"✗\"}')"
```

---

## 9. Storage Requirements

### Post-Training Datasets

| Category | Datasets | Raw Data | Embeddings (4096-dim) | Total |
|----------|----------|----------|----------------------|-------|
| Foundation | v1, v2, Llama | ~1.4 GB | ~9.6 GB | ~11 GB |
| v3 Collection | 8 datasets | ~2.9 GB | ~16.6 GB | ~19.5 GB |
| **TOTAL** | **12 datasets** | **~4.3 GB** | **~26.2 GB** | **~30.5 GB** |

**Embedding size calculation**: 
- Each embedding: 4096 dimensions × 4 bytes (float32) = 16,384 bytes (~16 KB)
- 1.64M samples × 16 KB ≈ 26.2 GB

---

## 10. Quality Checks

### Sample a Few Embeddings

```python
# sample_embeddings.py
import numpy as np
from pathlib import Path

# Load embeddings from a shard
embedding_file = Path('/raid/embeddings/nemotron-v3/science/train/data-00000-of-00010.npy')
embeddings = np.load(embedding_file)

print(f'Shape: {embeddings.shape}')
print(f'Dtype: {embeddings.dtype}')
print(f'First embedding (first 10 dims): {embeddings[0][:10]}')
print(f'Embedding norms (should be ~1.0): {np.linalg.norm(embeddings[:5], axis=1)}')
```

---

## 11. Common Issues and Solutions

### Issue: Out of Memory

**Solution**: Reduce batch size
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Science-v1 \
  --num-gpus 8 \
  --batch-size 2  # Reduced from 8
```

### Issue: Dataset Not Found

**Solution**: Check download location
```bash
ls -la /raid/datasets/nvidia_Nemotron-Science-v1/
# If missing, download again
python 00_download_nemotron_datasets.py
```

### Issue: Slow Processing

**Solution**: Enable Flash Attention (default) or increase GPUs
```bash
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Science-v1 \
  --num-gpus 8 \
  --batch-size 8
  # Flash Attention is enabled by default
```

---

## 12. Summary

### Complete Workflow for Post-Training Datasets

```bash
# 1. Verify status
python 01_verify_nemotron_dataset_status.py

# 2. Download (if needed)
python 00_download_nemotron_datasets.py

# 3. Explore
python 01_explore_nemotron_datasets.py

# 4. Extract embeddings (all post-training)
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
  --num-gpus 8 \
  --batch-size 8

# 5. Verify output
ls -lh /raid/embeddings/nemotron-*/
```

**Total time estimate**: 2-4 hours for all post-training datasets on 8 GPUs

---

## Quick Reference Table

| Dataset | Samples | Est. Time (8 GPUs) | Command |
|---------|---------|-------------------|---------|
| Nemotron-v1 | 100k | 15 min | `--datasets nvidia/Nemotron-Post-Training-Dataset-v1` |
| Nemotron-v2 | 200k | 30 min | `--datasets nvidia/Nemotron-Post-Training-Dataset-v2` |
| Llama-Nemotron | 400k | 60 min | `--datasets nvidia/Llama-Nemotron-Post-Training-Dataset` |
| Science-v1 | 150k | 25 min | `--datasets nvidia/Nemotron-Science-v1` |
| Instruction-Chat-v1 | 200k | 30 min | `--datasets nvidia/Nemotron-Instruction-Following-Chat-v1` |
| Math-Proofs-v1 | 50k | 10 min | `--datasets nvidia/Nemotron-Math-Proofs-v1` |
| 3-Nano-RL-Blend | 80k | 15 min | `--datasets nvidia/Nemotron-3-Nano-RL-Training-Blend` |
| Agentic-v1 | 100k | 15 min | `--datasets nvidia/Nemotron-Agentic-v1` |
| Competitive-Prog-v1 | 60k | 12 min | `--datasets nvidia/Nemotron-Competitive-Programming-v1` |
| Math-v2 | 180k | 30 min | `--datasets nvidia/Nemotron-Math-v2` |
| SWE-v1 | 120k | 20 min | `--datasets nvidia/Nemotron-SWE-v1` |
| **ALL POST-TRAINING** | **1.64M** | **2-4 hours** | *(See Section 4)* |

---

**Last Updated**: January 24, 2026  
**For pretraining datasets**: See main [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md)  
**For all datasets**: See main [`../README.md`](../README.md)
