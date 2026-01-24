#!/usr/bin/env python3
"""
Download Nemotron datasets from HuggingFace to local storage.
Uses HuggingFace CLI for reliable, complete downloads.

REQUIREMENTS:
    pip install -U 'huggingface_hub[cli]'

USAGE:
    # Edit the DATASETS list below to choose which datasets to download
    # Then run:
    python 00_download_nemotron_datasets.py

NOTES:
    - Post-training datasets: ~4.3 GB total (recommended to start)
    - Pretraining datasets: ~24 TB total (uncomment as needed)
    - Datasets are downloaded to /raid/datasets/ by default
    - Already downloaded datasets are automatically skipped
"""

import os
import subprocess
from pathlib import Path
import logging
import sys

# Configuration
DATASETS_DIR = Path("/raid/datasets")
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of datasets to download
# NOTE: Comment out datasets you don't want to download to save time and disk space

# Post-Training Datasets (12 datasets, ~1.64M rows, ~4.3 GB total)
POST_TRAINING_DATASETS = [
    "nvidia/Llama-Nemotron-Post-Training-Dataset",  # ~400k rows, ~800 MB
    "nvidia/Nemotron-Post-Training-Dataset-v1",     # ~100k rows, ~200 MB
    "nvidia/Nemotron-Post-Training-Dataset-v2",     # ~200k rows, ~400 MB
    # From Nemotron Post-Training v3 collection:
    "nvidia/Nemotron-3-Nano-RL-Training-Blend",     # ~80k rows, ~160 MB
    "nvidia/Nemotron-Science-v1",                   # ~150k rows, ~450 MB
    "nvidia/Nemotron-Instruction-Following-Chat-v1",# ~200k rows, ~500 MB
    "nvidia/Nemotron-Math-Proofs-v1",               # ~50k rows, ~250 MB
    "nvidia/Nemotron-Agentic-v1",                   # ~100k rows, ~300 MB
    "nvidia/Nemotron-Competitive-Programming-v1",   # ~60k rows, ~240 MB
    "nvidia/Nemotron-Math-v2",                      # ~180k rows, ~540 MB
    "nvidia/Nemotron-SWE-v1",                       # ~120k rows, ~480 MB
]

# Pretraining Datasets (9 datasets, ~15.1B rows, ~24 TB total)
# WARNING: These are VERY LARGE! Start with the sample dataset first.
PRETRAINING_DATASETS = [
    "nvidia/Nemotron-Pretraining-Dataset-sample",   # 27.7k rows, ~50 MB (TEST FIRST!)
    # "nvidia/Nemotron-CC-Code-v1",                 # 216M rows, ~350 GB
    # "nvidia/Nemotron-CC-v2.1",                    # 3.8B rows, ~6 TB (VERY LARGE!)
    # "nvidia/Nemotron-Pretraining-Code-v2",        # 836M rows, ~1.3 TB
    # "nvidia/Nemotron-Pretraining-Specialized-v1", # 60.7M rows, ~100 GB
    # "nvidia/Nemotron-CC-Math-v1",                 # 190M rows, ~310 GB
    # "nvidia/Nemotron-CC-v2",                      # 8.79B rows, ~14 TB (LARGEST!)
    # "nvidia/Nemotron-Pretraining-SFT-v1",         # 299M rows, ~480 GB
    # "nvidia/Nemotron-Pretraining-Code-v1",        # 936M rows, ~1.5 TB
]

# Combine datasets to download
# Uncomment PRETRAINING_DATASETS if you want to download them (requires TB of space!)
DATASETS = POST_TRAINING_DATASETS + PRETRAINING_DATASETS


def check_huggingface_cli():
    """Check if hf (HuggingFace CLI) is installed."""
    try:
        # Try running 'hf --help' instead of '--version' (which may not be supported)
        result = subprocess.run(
            ["hf", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info(f"✓ HuggingFace CLI (hf) is available")
            return True
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        # Command exists but timed out - that's okay
        logger.info(f"✓ HuggingFace CLI (hf) is available")
        return True
    except Exception as e:
        logger.warning(f"Error checking hf CLI: {e}")
        # Try to proceed anyway
        return True
    
    logger.error("HuggingFace CLI (hf) not found!")
    logger.error("Please install it with: pip install -U 'huggingface_hub[cli]'")
    return False


def download_dataset(dataset_name: str, output_dir: Path):
    """
    Download a dataset from HuggingFace using the hf CLI.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace (e.g., "nvidia/dataset-name")
        output_dir: Directory where the dataset will be saved
    """
    dataset_path = output_dir / dataset_name.replace("/", "_")
    
    # Check if dataset already exists
    if dataset_path.exists():
        # Check if it has content
        try:
            subdirs = list(dataset_path.iterdir())
            if subdirs:
                logger.info(f"Dataset {dataset_name} already exists at {dataset_path}. Skipping...")
                return
        except Exception:
            pass
    
    logger.info(f"Downloading dataset: {dataset_name}")
    logger.info(f"Destination: {dataset_path}")
    
    try:
        # Create destination directory
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Use hf (HuggingFace CLI) download command
        cmd = [
            "hf",
            "download",
            dataset_name,
            "--repo-type", "dataset",
            "--local-dir", str(dataset_path)
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Successfully downloaded {dataset_name}")
        else:
            logger.error(f"✗ Failed to download {dataset_name} (exit code: {result.returncode})")
            raise RuntimeError(f"Download failed with exit code {result.returncode}")
        
    except Exception as e:
        logger.error(f"Failed to download {dataset_name}: {e}")
        raise


def main():
    """Main function to download all datasets."""
    logger.info("="*80)
    logger.info("Nemotron Datasets Downloader")
    logger.info("="*80)
    
    # Check if hf CLI is available
    if not check_huggingface_cli():
        logger.error("Cannot proceed without HuggingFace CLI (hf)")
        logger.info("Install with: pip install -U 'huggingface_hub[cli]'")
        sys.exit(1)
    
    logger.info(f"Starting download of {len(DATASETS)} datasets to {DATASETS_DIR}")
    logger.info(f"Datasets directory: {DATASETS_DIR.absolute()}")
    logger.info("")
    
    failed_datasets = []
    skipped_datasets = []
    
    for i, dataset_name in enumerate(DATASETS, 1):
        logger.info("\n" + "="*80)
        logger.info(f"[{i}/{len(DATASETS)}] {dataset_name}")
        logger.info("="*80)
        
        try:
            dataset_path = DATASETS_DIR / dataset_name.replace("/", "_")
            if dataset_path.exists() and list(dataset_path.iterdir()):
                logger.info(f"Already exists, skipping: {dataset_path}")
                skipped_datasets.append(dataset_name)
                continue
            
            download_dataset(dataset_name, DATASETS_DIR)
        except Exception as e:
            logger.error(f"✗ Error processing {dataset_name}: {e}")
            failed_datasets.append(dataset_name)
            continue
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Download Summary")
    logger.info("="*80)
    logger.info(f"Total datasets: {len(DATASETS)}")
    logger.info(f"Successfully downloaded: {len(DATASETS) - len(failed_datasets) - len(skipped_datasets)}")
    logger.info(f"Already existed (skipped): {len(skipped_datasets)}")
    logger.info(f"Failed: {len(failed_datasets)}")
    
    if skipped_datasets:
        logger.info("\nSkipped datasets (already downloaded):")
        for dataset in skipped_datasets:
            logger.info(f"  ✓ {dataset}")
    
    if failed_datasets:
        logger.error("\nFailed datasets:")
        for dataset in failed_datasets:
            logger.error(f"  ✗ {dataset}")
        sys.exit(1)
    else:
        logger.info("\n" + "="*80)
        logger.info("✓ All datasets processed successfully!")
        logger.info("="*80)


if __name__ == "__main__":
    main()

