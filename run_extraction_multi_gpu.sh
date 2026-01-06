#!/bin/bash
# Multi-GPU embedding extraction for all Nemotron datasets

echo "==========================================="
echo "NeMo Curator Multi-GPU Embedding Extraction"
echo "==========================================="
echo ""

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"
echo ""

# Activate nemotron environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nemotron

# Run extraction on all datasets with all GPUs
python 02_extract_nemotron_embeddings.py \
    --num-gpus $NUM_GPUS \
    --batch-size 8 \
    --chunk-size 50 \
    --device cuda \
    --dtype float16

echo ""
echo "Extraction complete! Check /raid/embeddings/"

