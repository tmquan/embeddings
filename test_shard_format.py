#!/usr/bin/env python3
"""
Test script to verify the shard format matches the input dataset structure.
"""

import json
from pathlib import Path
from datasets import load_from_disk

# Test with the code split
dataset_path = Path("/raid/datasets/nvidia_Nemotron-Post-Training-Dataset-v1")
split_name = "code"

print("="*80)
print("TESTING SHARD FORMAT")
print("="*80)

# Load dataset
split_path = dataset_path / split_name
dataset = load_from_disk(str(split_path))

print(f"\nDataset: {dataset_path.name}")
print(f"Split: {split_name}")
print(f"Total samples: {len(dataset):,}")

# Get shard info
dataset_info_path = dataset_path / split_name / "dataset_info.json"

if dataset_info_path.exists():
    with open(dataset_info_path, 'r') as f:
        info = json.load(f)
    
    if "splits" in info and split_name in info["splits"]:
        split_info = info["splits"][split_name]
        shard_lengths = split_info.get("shard_lengths", [])
        total_shards = len(shard_lengths)
        
        print(f"\nShard Information:")
        print(f"  Total shards: {total_shards}")
        print(f"  Shard lengths: {shard_lengths[:10]}... (showing first 10)")
        print(f"  Sum of shard lengths: {sum(shard_lengths):,}")
        
        print(f"\nExpected output files:")
        for i in range(min(5, total_shards)):
            filename = f"data-{i:05d}-of-{total_shards:05d}.npy"
            print(f"  {filename} (samples: {shard_lengths[i]:,})")
        
        if total_shards > 5:
            print(f"  ... ({total_shards - 5} more)")
        
        # Verify boundaries
        print(f"\nVerifying shard boundaries:")
        start_idx = 0
        errors = []
        for shard_idx, shard_len in enumerate(shard_lengths[:5]):
            end_idx = start_idx + shard_len
            print(f"  Shard {shard_idx:05d}: samples {start_idx:,} to {end_idx:,} (length: {shard_len:,})")
            
            # Verify we can access these indices
            try:
                _ = dataset[start_idx]
                _ = dataset[end_idx - 1]
            except IndexError as e:
                errors.append(f"Shard {shard_idx}: {e}")
            
            start_idx = end_idx
        
        if errors:
            print("\n❌ Errors found:")
            for err in errors:
                print(f"  {err}")
        else:
            print("\n✓ All shard boundaries are valid!")
        
        print("\n" + "="*80)
        print("FORMAT SUMMARY")
        print("="*80)
        print(f"Input format:  data-{{shard_idx:05d}}-of-{{total:05d}}.arrow")
        print(f"Output format: data-{{shard_idx:05d}}-of-{{total:05d}}.npy")
        print(f"\nExample:")
        print(f"  Input:  data-00000-of-00{total_shards:03d}.arrow")
        print(f"  Output: data-00000-of-00{total_shards:03d}.npy")
        print("="*80)
    else:
        print(f"\n❌ Split '{split_name}' not found in dataset_info.json")
else:
    print(f"\n❌ dataset_info.json not found at {dataset_info_path}")

