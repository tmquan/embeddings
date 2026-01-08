#!/bin/bash
# Multi-GPU embedding extraction for all Nemotron datasets
# Saves embeddings in exact same shard format as input datasets

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

# Activate nemotron environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nemotron

# Run extraction on all datasets with all GPUs
# Note: chunk-size is ignored when dataset_info.json exists (uses original shards)
python 02_extract_nemotron_embeddings.py \
    --num-gpus $NUM_GPUS \
    --batch-size 8 \
    --chunk-size 50 \
    --device cuda \
    --dtype float16

echo ""
echo "Extraction complete! Check /raid/embeddings/"
echo "Format: /raid/embeddings/{subdir}/{split}/data-{shard:05d}-of-{total:05d}.npy"


