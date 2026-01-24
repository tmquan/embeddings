#!/bin/bash
# Extract embeddings from POST-TRAINING datasets only
# Only processes datasets that are actually downloaded

set -e  # Exit on error

echo "============================================================"
echo "Nemotron Post-Training Datasets - Embedding Extraction"
echo "============================================================"
echo ""

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "GPUs available: $NUM_GPUS"
echo ""

# Activate embeddings environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate embeddings

# List of post-training datasets
DATASETS_TO_PROCESS=(
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

# Check which datasets are downloaded
DOWNLOADED=""
NOT_FOUND=""

for dataset in "${DATASETS_TO_PROCESS[@]}"; do
    dataset_dir="/raid/datasets/${dataset//\//_}"
    if [ -d "$dataset_dir" ]; then
        DOWNLOADED="$DOWNLOADED $dataset"
        echo "✓ $dataset"
    else
        NOT_FOUND="$NOT_FOUND $dataset"
        echo "✗ $dataset (not downloaded)"
    fi
done

echo ""

# Exit if no datasets found
if [ -z "$DOWNLOADED" ]; then
    echo "❌ No post-training datasets found!"
    echo ""
    echo "Download datasets first:"
    echo "  python 00_download_nemotron_datasets.py"
    exit 1
fi

# Count
DOWNLOADED_COUNT=$(echo $DOWNLOADED | wc -w)
echo "Will process $DOWNLOADED_COUNT post-training datasets"
echo ""

# Run extraction
echo "============================================================"
echo "Starting extraction..."
echo "============================================================"
echo ""

python 02_extract_nemotron_embeddings.py \
    --datasets $DOWNLOADED \
    --num-gpus $NUM_GPUS \
    --batch-size 8 \
    --dtype bfloat16

echo ""
echo "============================================================"
echo "✓ Extraction Complete!"
echo "============================================================"
echo ""
echo "Embeddings saved to: /raid/embeddings/"
echo ""
echo "Check results:"
echo "  ls -lh /raid/embeddings/nemotron-*/"
echo "  ls -lh /raid/embeddings/llama-nemotron/"
echo ""
