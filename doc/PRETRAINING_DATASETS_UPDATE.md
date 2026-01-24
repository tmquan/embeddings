# Pretraining Datasets Integration

**Date**: January 24, 2026

## Summary

Updated the Nemotron datasets pipeline to support 9 new pretraining datasets in addition to the existing post-training datasets.

## Datasets Added

### Pretraining Datasets (9 new)
1. `nvidia/Nemotron-Pretraining-Dataset-sample` (27.7k) - Sample dataset for testing
2. `nvidia/Nemotron-CC-Code-v1` (216M) - Code from Common Crawl
3. `nvidia/Nemotron-CC-v2.1` (3.8B) - Common Crawl v2.1
4. `nvidia/Nemotron-Pretraining-Code-v2` (836M) - Code pretraining v2
5. `nvidia/Nemotron-Pretraining-Specialized-v1` (60.7M) - Specialized domains
6. `nvidia/Nemotron-CC-Math-v1` (190M) - Math from Common Crawl
7. `nvidia/Nemotron-CC-v2` (8.79B) - Common Crawl v2 (largest!)
8. `nvidia/Nemotron-Pretraining-SFT-v1` (299M) - SFT pretraining
9. `nvidia/Nemotron-Pretraining-Code-v1` (936M) - Code pretraining v1

### Post-Training Dataset Added
- `nvidia/Nemotron-SWE-v1` - Software Engineering dataset (was missing from config)

## Files Modified

### 1. `config.py`
**Changes:**
- Added 9 pretraining dataset configurations under `pretraining/` subdirectory structure
- Added missing `v3-swe` post-training dataset
- Total datasets: 21 (12 post-training + 9 pretraining)

**Directory Structure:**
```
embeddings/
├── nemotron-v1/           # Post-training v1
├── nemotron-v2/           # Post-training v2
├── llama-nemotron/        # Llama variants
├── nemotron-v3/           # Post-training v3 collection
│   ├── science/
│   ├── instruction-chat/
│   ├── math-proofs/
│   ├── rl-blend/
│   ├── agentic/
│   ├── competitive-programming/
│   ├── math-v2/
│   └── swe/
└── pretraining/           # NEW: Pretraining datasets
    ├── sample/
    ├── cc-code-v1/
    ├── cc-v2.1/
    ├── code-v2/
    ├── specialized-v1/
    ├── cc-math-v1/
    ├── cc-v2/
    ├── sft-v1/
    └── code-v1/
```

### 2. `01_explore_nemotron_datasets.py`
**Enhancements:**
- Updated `is_text_column()` to recognize primary pretraining columns (`text`, `content`, `document`, `passage`)
- Added more metadata columns to skip list (quality scores, timestamps, etc.)
- Enhanced `determine_embedding_strategy()` with new "direct_text" strategy for simple pretraining formats
- Updated `generate_extraction_functions()` to generate optimized extraction code for pretraining datasets

**New Strategy:**
- `direct_text`: Efficiently handles pretraining datasets with single text column (most common case)

### 3. `02_extract_nemotron_embeddings.py`
**Enhancements:**
- Reorganized `extract_text_from_example()` with priority-based approach:
  1. **Pretraining datasets** (simple text extraction) - FIRST
  2. **Post-training datasets** (conversational formats)
  3. **Generic fallback** (expanded field list)

- Added intelligent pretraining detection:
  - Checks for 'pretraining' or 'cc' in dataset name
  - Directly extracts from 'text' or 'content' columns
  - Optionally includes domain/source metadata if available

- Expanded generic fallback fields: `['text', 'content', 'prompt', 'response', 'document', 'passage']`

### 4. `00_download_nemotron_datasets.py`
**Changes:**
- Added all 9 pretraining datasets to download list
- Organized with comments separating post-training and pretraining datasets
- Added size information as comments for disk space planning

## Key Features

### 1. **Efficient Pretraining Data Handling**
- Pretraining datasets are processed FIRST (before checking complex conversational formats)
- Direct text extraction without unnecessary processing
- Metadata preservation when available (domain, source info)

### 2. **Backward Compatibility**
- All existing post-training dataset handling remains unchanged
- Conversational format detection still works as before
- No breaking changes to existing functionality

### 3. **Scalability**
- Handles both small (27.7k) and massive (8.79B) datasets
- Shard-based processing for large datasets
- Multi-GPU distributed extraction support

### 4. **Automatic Detection**
- Scripts automatically detect dataset type (pretraining vs post-training)
- Smart column identification
- Adaptive extraction strategies

## Usage

### Download Datasets
```bash
cd /localhome/local-tranminhq/embeddings
python 00_download_nemotron_datasets.py
```

**Note**: The large CC datasets (CC-v2 at 8.79B, CC-v2.1 at 3.8B) require significant disk space!

### Explore Datasets
```bash
python 01_explore_nemotron_datasets.py
```

This will:
- Analyze all 21 datasets
- Identify text columns
- Determine optimal extraction strategies
- Generate `dataset_exploration_summary.json` and `dataset_exploration_detailed.json`
- Auto-generate extraction functions in `embedding_extraction_functions.py`

### Extract Embeddings (Multi-GPU)
```bash
# Extract from specific pretraining dataset
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-Dataset-sample \
  --num-gpus 8 \
  --batch-size 4

# Extract from all datasets
python 02_extract_nemotron_embeddings.py --num-gpus 8

# Extract only pretraining datasets (filter in config)
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-CC-Code-v1 \
            nvidia/Nemotron-CC-Math-v1 \
            nvidia/Nemotron-Pretraining-Code-v2
```

## Testing Strategy

1. **Start Small**: Test with `Nemotron-Pretraining-Dataset-sample` (27.7k samples)
2. **Medium Size**: Try `Nemotron-Pretraining-Specialized-v1` (60.7M)
3. **Large Scale**: Process CC datasets last (multi-billion samples)

## Performance Considerations

### Disk Space Requirements
- Sample: ~100MB
- Small datasets (60-200M): 5-20GB
- Medium datasets (300-900M): 30-100GB
- Large CC datasets (3.8B-8.79B): 500GB-2TB

### Processing Time (8 GPUs, batch_size=4)
- Sample (27.7k): ~5 minutes
- Small (60M): ~2-4 hours
- Medium (300M): ~8-12 hours
- Large (3.8B): ~100+ hours
- Very Large (8.79B): ~250+ hours

## Output Structure

Embeddings will be saved to `/raid/embeddings/` following this structure:

```
/raid/embeddings/
├── pretraining/
│   ├── sample/
│   │   └── train/
│   │       ├── data-00000-of-00003.npy
│   │       ├── data-00001-of-00003.npy
│   │       ├── data-00002-of-00003.npy
│   │       └── embedding_metadata.json
│   ├── cc-code-v1/
│   ├── cc-v2.1/
│   └── ...
└── nemotron-v3/
    └── ...
```

Each `.npy` file contains embeddings as `float32` arrays with shape `[num_samples, 4096]`.

## Next Steps

1. **Download**: Start with sample dataset to verify pipeline
2. **Explore**: Run exploration to understand dataset structures
3. **Extract**: Process embeddings starting with smaller datasets
4. **Monitor**: Watch disk space and GPU utilization
5. **Validate**: Check embedding quality on sample data

## Notes

- Pretraining datasets typically have simpler structure (just 'text' column)
- Post-training datasets have complex conversational formats
- All scripts now handle both types seamlessly
- Consider processing order based on dataset size and importance
