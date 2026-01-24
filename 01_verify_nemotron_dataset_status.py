#!/usr/bin/env python3
"""
Quick utility to check which Nemotron datasets are downloaded and their sizes.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import humanize
except ImportError:
    print("Warning: 'humanize' package not installed. Using basic size formatting.")
    print("Install with: pip install humanize")
    humanize = None

from config import DATASET_CONFIGS, DEFAULT_DATASETS_DIR


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    if humanize:
        return humanize.naturalsize(size_bytes, binary=True)
    else:
        # Basic formatting if humanize not available
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"


def get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return total


def check_dataset_status(datasets_dir: Path) -> List[Dict]:
    """
    Check status of all configured datasets.
    Deduplicates datasets (e.g., Llama-Nemotron has both SFT and RL configs).
    
    Returns:
        List of dicts with dataset info
    """
    # Use dict to deduplicate by dataset name
    datasets_dict = {}
    
    for key, config in DATASET_CONFIGS.items():
        dataset_name = config["hf_name"]
        
        # Skip if already processed (deduplication)
        if dataset_name in datasets_dict:
            continue
            
        dataset_path = datasets_dir / dataset_name.replace("/", "_")
        
        status = {
            "key": key,
            "name": dataset_name,
            "subdir": config["subdir"],
            "exists": dataset_path.exists(),
            "path": str(dataset_path),
        }
        
        if status["exists"]:
            # Count splits
            splits = [d.name for d in dataset_path.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
            status["splits"] = splits
            status["num_splits"] = len(splits)
            
            # Calculate size
            size_bytes = get_directory_size(dataset_path)
            status["size_bytes"] = size_bytes
            status["size_human"] = format_size(size_bytes)
        else:
            status["splits"] = []
            status["num_splits"] = 0
            status["size_bytes"] = 0
            status["size_human"] = "N/A"
        
        datasets_dict[dataset_name] = status
    
    # Convert to list and return
    return list(datasets_dict.values())


def print_status_table(results: List[Dict]):
    """Print formatted status table."""
    # Separate by dataset type
    post_training = [r for r in results if not r["key"].startswith("pretrain")]
    pretraining = [r for r in results if r["key"].startswith("pretrain")]
    
    def print_section(datasets: List[Dict], title: str):
        print(f"\n{'='*100}")
        print(f"{title}")
        print(f"{'='*100}")
        print(f"{'Status':<10} {'Dataset':<50} {'Splits':<8} {'Size':<12}")
        print(f"{'-'*100}")
        
        total_size = 0
        downloaded = 0
        
        for ds in datasets:
            status_icon = "✓" if ds["exists"] else "✗"
            status_text = "Downloaded" if ds["exists"] else "Not Found"
            name = ds["name"].replace("nvidia/", "")
            
            print(f"{status_icon} {status_text:<8} {name:<50} "
                  f"{ds['num_splits']:<8} {ds['size_human']:<12}")
            
            if ds["exists"]:
                downloaded += 1
                total_size += ds["size_bytes"]
        
        print(f"{'-'*100}")
        print(f"Downloaded: {downloaded}/{len(datasets)} datasets")
        print(f"Total Size: {format_size(total_size)}")
    
    print_section(post_training, "POST-TRAINING DATASETS")
    print_section(pretraining, "PRETRAINING DATASETS")
    
    # Overall summary
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}")
    
    total_downloaded = sum(1 for r in results if r["exists"])
    total_size = sum(r["size_bytes"] for r in results if r["exists"])
    
    print(f"Total Unique Datasets: {len(results)}")
    print(f"Total Downloaded: {total_downloaded}")
    print(f"Total Disk Usage: {format_size(total_size)}")
    print(f"Datasets Directory: {DEFAULT_DATASETS_DIR}")
    
    # Show what's missing
    missing = [r["name"] for r in results if not r["exists"]]
    if missing:
        print(f"\nMissing Datasets ({len(missing)}):")
        for name in missing[:10]:  # Show first 10
            print(f"  - {name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    else:
        print("\n✓ All datasets downloaded!")


def main():
    """Main function."""
    datasets_dir = Path(DEFAULT_DATASETS_DIR)
    
    print(f"Checking datasets in: {datasets_dir}")
    print(f"(Duplicates like Llama-Nemotron SFT/RL are shown once)")
    
    if not datasets_dir.exists():
        print(f"❌ Datasets directory not found: {datasets_dir}")
        print(f"\nPlease create the directory or update DEFAULT_DATASETS_DIR in config.py")
        return
    
    results = check_dataset_status(datasets_dir)
    print_status_table(results)
    
    print(f"\n{'='*100}")
    print("Next Steps:")
    print(f"{'='*100}")
    print("1. Download missing datasets:")
    print("   python 00_download_nemotron_datasets.py")
    print("\n2. Explore downloaded datasets:")
    print("   python 01_explore_nemotron_datasets.py")
    print("\n3. Extract embeddings:")
    print("   python 02_extract_nemotron_embeddings.py --num-gpus 8")


if __name__ == "__main__":
    main()
