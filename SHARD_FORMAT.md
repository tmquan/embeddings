# Shard Format - Matching Input Dataset Structure

## Overview

The embedding extraction now saves outputs in **exactly the same format** as the input datasets, with matching shard counts and 5-digit zero-padding.

## Format Specification

### Input Format (Arrow files)
```
data-{shard_idx:05d}-of-{total_shards:05d}.arrow
```

Example:
```
data-00000-of-00183.arrow
data-00001-of-00183.arrow
...
data-00182-of-00183.arrow
```

### Output Format (NumPy embeddings)
```
data-{shard_idx:05d}-of-{total_shards:05d}.npy
```

Example:
```
data-00000-of-00183.npy
data-00001-of-00183.npy
...
data-00182-of-00183.npy
```

## Key Features

### 1. **Exact Shard Matching**
- Reads `dataset_info.json` from each split
- Uses the original `shard_lengths` to determine boundaries
- Each output `.npy` file corresponds 1:1 with an input `.arrow` file

### 2. **Preserved Sample Counts**
- Each shard contains embeddings for exactly the same samples as the input shard
- Shard boundaries align perfectly with input data

### 3. **5-Digit Zero-Padding**
- Consistent with HuggingFace datasets format
- Supports up to 99,999 shards per split

## Example: Code Split

```
Input Dataset: nvidia/Nemotron-Post-Training-Dataset-v1/code
Total samples: 1,896,395
Total shards: 175

Input structure:
  /raid/datasets/nvidia_Nemotron-Post-Training-Dataset-v1/code/
    ├── data-00000-of-00175.arrow  (13,363 samples)
    ├── data-00001-of-00175.arrow  (13,363 samples)
    ├── data-00002-of-00175.arrow  (12,363 samples)
    ...
    └── data-00174-of-00175.arrow  (10,550 samples)

Output structure:
  /raid/embeddings/nemotron-v1/code/
    ├── data-00000-of-00175.npy    (13,363 embeddings)
    ├── data-00001-of-00175.npy    (13,363 embeddings)
    ├── data-00002-of-00175.npy    (12,363 embeddings)
    ...
    └── data-00174-of-00175.npy    (10,550 embeddings)
```

## Metadata

Each split also gets an `embedding_metadata.json` file:

```json
{
  "dataset_name": "nvidia/Nemotron-Post-Training-Dataset-v1",
  "split_name": "code",
  "num_samples": 1896395,
  "num_shards": 175,
  "shards_completed": 175,
  "shard_lengths": [13363, 13363, 12363, ...],
  "embedding_dim": 8192,
  "model_id": "nvidia/llama-embed-nemotron-8b",
  "num_gpus": 8
}
```

## Implementation Details

### Shard Boundary Detection

```python
def get_shard_info(dataset_path: Path, split_name: str) -> tuple[List[int], int]:
    """Read shard lengths from dataset_info.json"""
    dataset_info_path = dataset_path / split_name / "dataset_info.json"
    
    with open(dataset_info_path, 'r') as f:
        info = json.load(f)
    
    shard_lengths = info["splits"][split_name]["shard_lengths"]
    return shard_lengths, len(shard_lengths)
```

### Processing Loop

```python
# Get original shard structure
shard_lengths, total_shards = get_shard_info(dataset_path, split_name)

# Process each shard
start_idx = 0
for shard_idx, shard_len in enumerate(shard_lengths):
    end_idx = start_idx + shard_len
    
    # Extract data for this shard
    shard_data = [dataset[i] for i in range(start_idx, end_idx)]
    
    # Process and save
    embeddings = extractor.process_chunk(shard_data, dataset_name)
    output_file = f"data-{shard_idx:05d}-of-{total_shards:05d}.npy"
    np.save(output_file, embeddings)
    
    start_idx = end_idx
```

## Benefits

1. **Consistency**: Output structure mirrors input structure
2. **Traceability**: Easy to map embeddings back to source data
3. **Compatibility**: Works with existing data loading pipelines
4. **Scalability**: Handles datasets with varying shard sizes
5. **Standards**: Follows HuggingFace dataset conventions

## Fallback Behavior

If `dataset_info.json` is not found or doesn't contain shard information, the system falls back to uniform chunk sizes:

```python
# Fallback: use uniform chunk_size
num_chunks = (num_samples + chunk_size - 1) // chunk_size
```

This ensures the pipeline works even with custom or modified datasets.

## Testing

Run the test script to verify format:

```bash
python test_shard_format.py
```

This will:
- Load the dataset and extract shard information
- Verify shard boundaries are valid
- Show expected output filenames
- Confirm format matches input structure

