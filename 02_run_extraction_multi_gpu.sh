#!/bin/bash
# Multi-GPU embedding extraction for downloaded Nemotron datasets
# Saves embeddings in exact same shard format as input datasets

set -e  # Exit on error

echo "==========================================="
echo "NeMo Curator Multi-GPU Embedding Extraction"
echo "==========================================="
echo ""

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"
echo ""

echo "Output format: data-{shard:05d}-of-{total:05d}.npy"
echo "Matches input: data-{shard:05d}-of-{total:05d}.arrow"
echo ""

# Activate embeddings environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate embeddings

# Check which datasets are downloaded
echo "Checking downloaded datasets..."
python 01_verify_nemotron_dataset_status.py | grep "✓" | head -20
echo ""

# Get list of downloaded POST-TRAINING datasets only
DOWNLOADED_DATASETS=""

# Post-training datasets to check
declare -a POST_TRAINING=(
    "nvidia/Llama-Nemotron-Post-Training-Dataset"
    "nvidia/Nemotron-Post-Training-Dataset-v1"
    "nvidia/Nemotron-Post-Training-Dataset-v2"
    "nvidia/Nemotron-3-Nano-RL-Training-Blend"
    "nvidia/Nemotron-Science-v1"
    "nvidia/Nemotron-Instruction-Following-Chat-v1"
    "nvidia/Nemotron-Math-Proofs-v1"
    "nvidia/Nemotron-Agentic-v1"
    "nvidia/Nemotron-Competitive-Programming-v1"
    "nvidia/Nemotron-Math-v2"
    "nvidia/Nemotron-SWE-v1"
)

# Check which ones exist
for dataset in "${POST_TRAINING[@]}"; do
    dataset_dir="/raid/datasets/${dataset//\//_}"
    if [ -d "$dataset_dir" ]; then
        DOWNLOADED_DATASETS="$DOWNLOADED_DATASETS $dataset"
        echo "✓ Found: $dataset"
    fi
done

echo ""
echo "Will process ${#POST_TRAINING[@]} post-training datasets"
echo ""

# Run extraction on downloaded datasets only
if [ -z "$DOWNLOADED_DATASETS" ]; then
    echo "❌ No datasets found in /raid/datasets/"
    echo "Please download datasets first:"
    echo "  python 00_download_nemotron_datasets.py"
    exit 1
fi

echo "Running extraction..."
echo ""

python 02_extract_nemotron_embeddings.py \
    --datasets $DOWNLOADED_DATASETS \
    --num-gpus $NUM_GPUS \
    --batch-size 2 \
    --dtype bfloat16

echo ""
echo "=========================================="
echo "✓ Extraction complete!"
echo "=========================================="
echo "Check embeddings at: /raid/embeddings/"
echo "Format: /raid/embeddings/{subdir}/{split}/data-{shard:05d}-of-{total:05d}.npy"


