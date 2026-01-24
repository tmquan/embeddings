#!/bin/bash
# Multi-GPU Embedding Extraction for Nemotron Datasets
# All 8 GPUs work together on each dataset, then move to the next
#
# Usage:
#   ./02_extract_post_training.sh           # Run all datasets
#   ./02_extract_post_training.sh --dry-run # Show what would be run
#   ./02_extract_post_training.sh v1        # Run only dataset v1

set -e

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate embeddings

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/02_extract_nemotron_embeddings.py"
DATASETS_DIR="/raid/datasets"
OUTPUT_DIR="/raid/embeddings"
MAX_LENGTH=32768
BATCH_SIZE=20
SHARD_SIZE=10000
DTYPE="bfloat16"
NUM_GPUS=8

# All datasets to process (in order of priority/size)
DATASETS=(
    "pretrain-sample"              # 28K rows
    "llama-sft"                    # 33M rows
    "v1"                           # 26M rows  
    "v2"                           # 6M rows
    "v3-math"                      # 7M rows
    "v3-competitive-programming"   # 4M rows
    "v3-math-proofs"               # 1.4M rows
    "v3-instruction-chat"          # 431K rows
    "v3-agentic"                   # 335K rows
    "v3-science"                   # 226K rows
    "v3-rl-blend"                  # 93K rows
    "v3-swe"                       # 51K rows
)

# Parse arguments
DRY_RUN=false
SINGLE_DATASET=""

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        *)
            SINGLE_DATASET="$arg"
            ;;
    esac
done

# If single dataset specified, only process that one
if [[ -n "$SINGLE_DATASET" ]]; then
    DATASETS=("$SINGLE_DATASET")
fi

echo "============================================================"
echo "Multi-GPU Embedding Extraction (8 GPUs per dataset)"
echo "============================================================"
echo "Datasets dir: ${DATASETS_DIR}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Max length:   ${MAX_LENGTH}"
echo "Batch size:   ${BATCH_SIZE}"
echo "Shard size:   ${SHARD_SIZE}"
echo "Dtype:        ${DTYPE}"
echo "Datasets:     ${#DATASETS[@]}"
echo "============================================================"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== DRY RUN MODE ==="
fi

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"

# Function to process a single dataset with all GPUs
process_dataset() {
    local dataset=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    echo ""
    echo "============================================================"
    echo "Processing dataset: ${dataset}"
    echo "Timestamp: ${timestamp}"
    echo "============================================================"
    
    # Launch all 8 GPUs in parallel
    local pids=()
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        local log_file="${OUTPUT_DIR}/logs/${dataset}_gpu${gpu_id}_${timestamp}.log"
        
        local cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python ${PYTHON_SCRIPT} \
            --datasets-dir ${DATASETS_DIR} \
            --output-dir ${OUTPUT_DIR} \
            --max-length ${MAX_LENGTH} \
            --batch-size ${BATCH_SIZE} \
            --shard-size ${SHARD_SIZE} \
            --dtype ${DTYPE} \
            --dataset ${dataset} \
            --gpu-id ${gpu_id} \
            --num-gpus ${NUM_GPUS}"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[GPU ${gpu_id}] Command: ${cmd}"
        else
            echo "[GPU ${gpu_id}] Starting... Log: ${log_file}"
            eval "${cmd}" > "${log_file}" 2>&1 &
            pids+=($!)
        fi
    done
    
    if [[ "$DRY_RUN" == "false" ]]; then
        echo ""
        echo "Waiting for all ${NUM_GPUS} GPUs to complete ${dataset}..."
        echo "Monitor: tail -f ${OUTPUT_DIR}/logs/${dataset}_gpu*_${timestamp}.log"
        
        # Wait for all processes and check exit codes
        local failed=0
        for pid in "${pids[@]}"; do
            if ! wait $pid; then
                echo "WARNING: Process $pid failed"
                ((failed++))
            fi
        done
        
        if [[ $failed -gt 0 ]]; then
            echo "WARNING: $failed GPU(s) failed for ${dataset}"
        else
            echo "SUCCESS: All GPUs completed ${dataset}"
        fi
    fi
}

# Process each dataset sequentially (all GPUs work together on each)
for dataset in "${DATASETS[@]}"; do
    process_dataset "$dataset"
done

echo ""
echo "============================================================"
echo "ALL DATASETS COMPLETE"
echo "============================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To merge GPU shards for each split, run:"
echo "  python -c \"import numpy as np; from pathlib import Path; ..."
echo ""

# Show summary of output files
if [[ "$DRY_RUN" == "false" ]]; then
    echo "Output summary:"
    find "${OUTPUT_DIR}" -name "*.npy" -type f | wc -l | xargs echo "  Total shard files:"
    du -sh "${OUTPUT_DIR}" | awk '{print "  Total size: " $1}'
fi
