#!/bin/bash
# Test multi-GPU extraction on small dataset

echo "==========================================="
echo "Testing Multi-GPU Embedding Extraction"
echo "==========================================="
echo ""

# Activate nemotron environment  
source ~/anaconda3/etc/profile.d/conda.sh
conda activate embeddings

# Test on smallest dataset with 2 GPUs
python 02_extract_nemotron_embeddings.py \
    --datasets "nvidia/Nemotron-Science-v1" \
    --splits "MCQ" \
    --num-gpus 2 \
    --batch-size 4 \
    --chunk-size 50 \
    --dtype float16

echo ""
echo "Test complete! Check /raid/embeddings/nemotron-v3/science/MCQ/"

