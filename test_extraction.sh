#!/bin/bash
# Test embedding extraction on a small subset

echo "========================================="
echo "Testing Nemotron Embedding Extraction"
echo "========================================="
echo ""

# Activate nemotron environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate embeddings

# Test on a small dataset (Nemotron-Science-v1, smallest one)
python 02_extract_nemotron_embeddings.py \
    --datasets "nvidia/Nemotron-Science-v1" \
    --batch-size 4 \
    --chunk-size 100 \
    --device cuda \
    --dtype bfloat16

echo ""
echo "Test complete! Check /raid/embeddings/nemotron-v3/science/"

