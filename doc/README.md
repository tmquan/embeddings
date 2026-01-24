# Documentation

This folder contains detailed documentation for the Nemotron Embeddings Pipeline.

## Documentation Files

### Quick Reference
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheat sheet, common workflows, and troubleshooting tips
- **[POST_TRAINING_COMMANDS.md](POST_TRAINING_COMMANDS.md)** - Complete guide for Post-Training datasets only (12 datasets)

### Updates and Changes
- **[PRETRAINING_DATASETS_UPDATE.md](PRETRAINING_DATASETS_UPDATE.md)** - Details on the 9 new pretraining datasets added
- **[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)** - Complete changelog and impact summary


### Technical Details
- **[SHARD_FORMAT.md](SHARD_FORMAT.md)** - Dataset shard format and structure

## Dataset Distribution Overview

### Post-Training Datasets (Conversational Format)

| Dataset | Topic/Domain | Rows | Storage Size | Avg Chars/Row | Est. Tokens | Notes |
|---------|--------------|------|--------------|---------------|-------------|-------|
| Nemotron-v1 | General chat | ~100k | ~200 MB | ~2,000 | ~25M | Multi-turn conversations |
| Nemotron-v2 | General chat | ~200k | ~400 MB | ~2,000 | ~50M | Enhanced v1 |
| Llama-Nemotron SFT | Instruction following | ~300k | ~600 MB | ~2,000 | ~75M | Supervised fine-tuning |
| Llama-Nemotron RL | Reward modeling | ~100k | ~200 MB | ~2,000 | ~25M | Reinforcement learning |
| Science-v1 | Scientific | ~150k | ~450 MB | ~3,000 | ~56M | STEM domains |
| Instruction-Chat-v1 | Instructions | ~200k | ~500 MB | ~2,500 | ~63M | Task completion |
| Math-Proofs-v1 | Mathematics | ~50k | ~250 MB | ~5,000 | ~31M | Formal proofs |
| 3-Nano-RL-Blend | Mixed RL | ~80k | ~160 MB | ~2,000 | ~20M | Training blend |
| Agentic-v1 | Tool use | ~100k | ~300 MB | ~3,000 | ~38M | Agent interactions |
| Competitive-Prog-v1 | Programming | ~60k | ~240 MB | ~4,000 | ~30M | Contest problems |
| Math-v2 | Mathematics | ~180k | ~540 MB | ~3,000 | ~68M | Enhanced math |
| SWE-v1 | Software Eng | ~120k | ~480 MB | ~4,000 | ~60M | Code + docs |
| **TOTAL** | **Mixed** | **~1.64M** | **~4.3 GB** | **~2,800 avg** | **~541M** | **12 datasets** |

### Pretraining Datasets (Raw Text Format)

| Dataset | Topic/Domain | Rows | Storage Size | Avg Chars/Row | Est. Tokens | Notes |
|---------|--------------|------|--------------|---------------|-------------|-------|
| Pretrain-Sample | Mixed | 27.7k | ~50 MB | ~1,800 | ~0.6M | **Test dataset** |
| CC-Code-v1 | Code | 216M | ~350 GB | ~1,600 | ~43B | Common Crawl code |
| CC-v2.1 | General web | 3.8B | ~6 TB | ~1,600 | ~761B | **Very large!** |
| Pretrain-Code-v2 | Code | 836M | ~1.3 TB | ~1,600 | ~167B | Multi-language |
| Pretrain-Specialized-v1 | Domain-specific | 60.7M | ~100 GB | ~1,650 | ~13B | Curated domains |
| CC-Math-v1 | Mathematics | 190M | ~310 GB | ~1,650 | ~39B | Math content |
| CC-v2 | General web | 8.79B | ~14 TB | ~1,600 | ~1.76T | **Largest!** |
| Pretrain-SFT-v1 | Instruction | 299M | ~480 GB | ~1,600 | ~60B | SFT-formatted |
| Pretrain-Code-v1 | Code | 936M | ~1.5 TB | ~1,600 | ~187B | Legacy version |
| **TOTAL** | **Mixed** | **~15.1B** | **~24.0 TB** | **~1,605 avg** | **~3.03T** | **9 datasets** |

### Combined Statistics

| Category | Datasets | Total Rows | Total Size | Total Tokens (Est.) |
|----------|----------|------------|------------|---------------------|
| Post-Training | 12 | ~1.64M | ~4.3 GB | ~541M |
| Pretraining | 9 | ~15.1B | ~24.0 TB | ~3.03T |
| **GRAND TOTAL** | **21** | **~15.1B** | **~24 TB** | **~3.03T** |

### Size Estimates Methodology

**Character to Token Ratio**: ~4 chars per token (English text average)
- Post-training: Conversational data with rich context
- Pretraining: Raw web text, code, and domain-specific content

**Storage Calculations**:
- Based on typical JSON/Parquet compression ratios
- Post-training: ~2 KB per conversation (avg)
- Pretraining: ~1.6 KB per document (avg)

**Actual Sizes**: Run exploration script for exact measurements:
```bash
python 01_explore_nemotron_datasets.py
# Generates dataset_exploration_summary.json with exact statistics
```

### Dataset Categories by Size

#### Tiny (< 100 MB)
- Pretrain-Sample (27.7k rows) - Perfect for testing

#### Small (100 MB - 1 GB)
- All post-training datasets except Science, Math-Proofs, Agentic

#### Medium (1 GB - 100 GB)
- Pretrain-Specialized-v1 (60.7M rows, ~100 GB)

#### Large (100 GB - 1 TB)
- CC-Code-v1 (216M rows, ~350 GB)
- CC-Math-v1 (190M rows, ~310 GB)
- Pretrain-SFT-v1 (299M rows, ~480 GB)

#### Very Large (1 TB - 10 TB)
- Pretrain-Code-v2 (836M rows, ~1.3 TB)
- Pretrain-Code-v1 (936M rows, ~1.5 TB)
- CC-v2.1 (3.8B rows, ~6 TB)

#### Massive (> 10 TB)
- CC-v2 (8.79B rows, ~14 TB) ⚠️ **Requires significant infrastructure**

### Token Distribution by Domain

| Domain | Estimated Tokens | Percentage | Datasets |
|--------|-----------------|------------|----------|
| General Web | ~2.52T | 83.2% | CC-v2, CC-v2.1 |
| Code | ~397B | 13.1% | CC-Code-v1, Pretrain-Code-v1/v2 |
| Mathematics | ~78B | 2.6% | CC-Math-v1, Math-v1/v2, Math-Proofs |
| Specialized | ~13B | 0.4% | Pretrain-Specialized-v1 |
| Science | ~56M | <0.1% | Science-v1 |
| Other Post-training | ~412M | <0.1% | Remaining 10 datasets |

### Recommended Processing Order by Size

1. **Testing Phase** (< 1 GB total)
   - Pretrain-Sample (50 MB)
   - 2-3 small post-training datasets (~1 GB)

2. **Initial Production** (< 100 GB total)
   - All post-training datasets (~4.3 GB)
   - Pretrain-Specialized-v1 (~100 GB)

3. **Scale-up Phase** (< 1 TB total)
   - CC-Code-v1 (~350 GB)
   - CC-Math-v1 (~310 GB)
   - Pretrain-SFT-v1 (~480 GB)

4. **Large-scale Phase** (1-10 TB)
   - Pretrain-Code-v1 (~1.5 TB)
   - Pretrain-Code-v2 (~1.3 TB)
   - CC-v2.1 (~6 TB)

5. **Massive-scale Phase** (10+ TB)
   - CC-v2 (~14 TB) - Process last or skip if storage-constrained

### Notes on Estimates

- **Estimates vs Actuals**: Sizes are estimates based on typical dataset characteristics
- **Compression**: Actual disk usage depends on file format and compression
- **Run Exploration**: For exact statistics, run `python 01_explore_nemotron_datasets.py`
- **Token Counts**: Based on ~4 chars/token ratio; actual varies by content type
- **Storage Planning**: Add 20-30% overhead for embeddings and metadata

## Main Documentation

For the main README with quick start guide and dataset-specific extraction commands, see:
- **[../README.md](../README.md)** - Main documentation in the root embeddings folder

## Getting Started

### For Post-Training Datasets Only
1. Read [POST_TRAINING_COMMANDS.md](POST_TRAINING_COMMANDS.md) - Complete workflow for 12 post-training datasets
2. Estimated time: 2-4 hours for all datasets on 8 GPUs
3. Storage needed: ~30 GB (4.3 GB data + 26 GB embeddings)

### For All Datasets (Post-Training + Pretraining)
1. Read [../README.md](../README.md) for quick start and overview
2. Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for command examples
3. Review [PRETRAINING_DATASETS_UPDATE.md](PRETRAINING_DATASETS_UPDATE.md) for recent changes
4. See [README.md](README.md) dataset distribution section for size planning