#!/usr/bin/env python3
"""
Download Nemotron datasets from HuggingFace to local storage.
"""

import os
from pathlib import Path
from datasets import load_dataset
import logging

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
# Extracted from embeddings/datasets.txt
DATASETS = [
    "nvidia/Llama-Nemotron-Post-Training-Dataset",
    "nvidia/Nemotron-Post-Training-Dataset-v1",
    "nvidia/Nemotron-Post-Training-Dataset-v2",
    # From Nemotron Post-Training v3 collection:
    "nvidia/Nemotron-3-Nano-RL-Training-Blend",
    "nvidia/Nemotron-Science-v1",
    "nvidia/Nemotron-Instruction-Following-Chat-v1",
    "nvidia/Nemotron-Math-Proofs-v1",
    "nvidia/Nemotron-Agentic-v1",
    "nvidia/Nemotron-Competitive-Programming-v1",
    "nvidia/Nemotron-Math-v2",
]


def download_dataset(dataset_name: str, output_dir: Path):
    """
    Download a dataset from HuggingFace and save it locally.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace (e.g., "nvidia/dataset-name")
        output_dir: Directory where the dataset will be saved
    """
    dataset_path = output_dir / dataset_name.replace("/", "_")
    
    # Check if dataset already exists
    if dataset_path.exists():
        logger.info(f"Dataset {dataset_name} already exists at {dataset_path}. Skipping...")
        return
    
    logger.info(f"Downloading dataset: {dataset_name}")
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, cache_dir=str(output_dir / ".cache"))
        
        # Save to disk
        dataset.save_to_disk(str(dataset_path))
        logger.info(f"Successfully saved {dataset_name} to {dataset_path}")
        
    except Exception as e:
        logger.error(f"Failed to download {dataset_name}: {e}")
        raise


def main():
    """Main function to download all datasets."""
    logger.info(f"Starting download of {len(DATASETS)} datasets to {DATASETS_DIR}")
    logger.info(f"Datasets directory: {DATASETS_DIR.absolute()}")
    
    failed_datasets = []
    
    for i, dataset_name in enumerate(DATASETS, 1):
        logger.info(f"\n[{i}/{len(DATASETS)}] Processing: {dataset_name}")
        try:
            download_dataset(dataset_name, DATASETS_DIR)
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}")
            failed_datasets.append(dataset_name)
            continue
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Download Summary")
    logger.info("="*60)
    logger.info(f"Total datasets: {len(DATASETS)}")
    logger.info(f"Successfully downloaded: {len(DATASETS) - len(failed_datasets)}")
    logger.info(f"Failed: {len(failed_datasets)}")
    
    if failed_datasets:
        logger.error("\nFailed datasets:")
        for dataset in failed_datasets:
            logger.error(f"  - {dataset}")
    else:
        logger.info("\nAll datasets downloaded successfully!")


if __name__ == "__main__":
    main()

