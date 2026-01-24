# Nemotron Embeddings Pipeline - Quick Reference

## 🚀 Quick Commands

### Check What's Downloaded
```bash
python 01_verify_nemotron_dataset_status.py
```
Shows which datasets are available and their sizes.

### Download Datasets
```bash
# Download all datasets (WARNING: ~4TB total!)
python 00_download_nemotron_datasets.py

# Or download specific ones by editing the DATASETS list in the script
```

### Explore Dataset Structures
```bash
python 01_explore_nemotron_datasets.py
```
Analyzes datasets and generates extraction strategies.

### Extract Embeddings
```bash
# Start with sample dataset (recommended)
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-Dataset-sample \
  --num-gpus 8 \
  --batch-size 8

# Process specific datasets
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Science-v1 \
            nvidia/Nemotron-Math-v2 \
  --num-gpus 8 \
  --batch-size 8

# Process all datasets (WARNING: takes days!)
python 02_extract_nemotron_embeddings.py --num-gpus 8 --batch-size 8
```

## 📊 Dataset Overview

### 21 Total Datasets

**Post-Training (12)** - Instruction/chat format
- `nvidia/Nemotron-Post-Training-Dataset-v1`
- `nvidia/Nemotron-Post-Training-Dataset-v2`
- `nvidia/Llama-Nemotron-Post-Training-Dataset`
- `nvidia/Nemotron-Science-v1`
- `nvidia/Nemotron-Instruction-Following-Chat-v1`
- `nvidia/Nemotron-Math-Proofs-v1`
- `nvidia/Nemotron-3-Nano-RL-Training-Blend`
- `nvidia/Nemotron-Agentic-v1`
- `nvidia/Nemotron-Competitive-Programming-v1`
- `nvidia/Nemotron-Math-v2`
- `nvidia/Nemotron-SWE-v1`

**Pretraining (9)** - Raw text format
- `nvidia/Nemotron-Pretraining-Dataset-sample` ⭐ **START HERE** (27.7k)
- `nvidia/Nemotron-CC-Code-v1` (216M)
- `nvidia/Nemotron-CC-v2.1` (3.8B)
- `nvidia/Nemotron-Pretraining-Code-v2` (836M)
- `nvidia/Nemotron-Pretraining-Specialized-v1` (60.7M)
- `nvidia/Nemotron-CC-Math-v1` (190M)
- `nvidia/Nemotron-CC-v2` ⚠️ **LARGEST** (8.79B)
- `nvidia/Nemotron-Pretraining-SFT-v1` (299M)
- `nvidia/Nemotron-Pretraining-Code-v1` (936M)

## 🎯 Recommended Workflow

### 1️⃣ Initial Setup
```bash
# Check current status
python 01_verify_nemotron_dataset_status.py

# Download sample dataset for testing
# Edit 00_download_nemotron_datasets.py to only include:
# - nvidia/Nemotron-Pretraining-Dataset-sample
python 00_download_nemotron_datasets.py
```

### 2️⃣ Test the Pipeline
```bash
# Explore the sample dataset
python 01_explore_nemotron_datasets.py

# Extract embeddings from sample (should take ~5 min on 8 GPUs)
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-Dataset-sample \
  --num-gpus 8 \
  --batch-size 8

# Verify output
ls -lh /raid/embeddings/pretraining/sample/
```

### 3️⃣ Scale Up
```bash
# Download more datasets (start with smaller ones)
# Edit 00_download_nemotron_datasets.py to add:
# - nvidia/Nemotron-Pretraining-Specialized-v1 (60.7M)
# - nvidia/Nemotron-Science-v1 (~150k)
python 00_download_nemotron_datasets.py

# Process them
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-Specialized-v1 \
            nvidia/Nemotron-Science-v1 \
  --num-gpus 8 \
  --batch-size 8
```

### 4️⃣ Production Processing
```bash
# Download all datasets you need
# Process strategically based on priority and disk space
python 02_extract_nemotron_embeddings.py \
  --datasets <your-priority-datasets> \
  --num-gpus 8 \
  --batch-size 8 \
  --dtype bfloat16
```

## ⚙️ Common Options

### Extraction Script Options
```bash
--datasets          # Filter specific datasets
--splits            # Filter specific splits (train, test, etc.)
--num-gpus 8        # Number of GPUs (default: all available)
--batch-size 8      # Batch size per GPU (default: 8)
--chunk-size 10000  # Samples per shard
--dtype bfloat16    # Model precision (bfloat16/float16/float32)
--no-flash-attention # Disable Flash Attention 2
--datasets-dir /raid/datasets      # Input directory
--output-dir /raid/embeddings      # Output directory
```

## 📁 File Locations

### Configuration
- `config.py` - Central configuration (paths, model, batch size)

### Scripts
- `00_download_nemotron_datasets.py` - Download datasets
- `01_explore_nemotron_datasets.py` - Analyze structures
- `02_extract_nemotron_embeddings.py` - Extract embeddings
- `check_dataset_status.py` - Check download status

### Outputs
- `/raid/datasets/` - Downloaded datasets
- `/raid/embeddings/` - Extracted embeddings
- `dataset_exploration_summary.json` - Dataset overview
- `dataset_exploration_detailed.json` - Detailed analysis
- `embedding_extraction_functions.py` - Auto-generated functions

### Documentation
- `README.md` - Full documentation
- `QUICK_REFERENCE.md` - This file
- `PRETRAINING_DATASETS_UPDATE.md` - Recent changes
- `SHARD_FORMAT.md` - Shard format documentation

## 🔧 Troubleshooting

### "Out of memory" errors
```bash
# Reduce batch size
--batch-size 1

# Use lower precision
--dtype float16
```

### Slow processing
```bash
# Make sure Flash Attention 2 is enabled (default)
# Check GPU utilization: nvidia-smi

# Increase GPUs if available
--num-gpus 8
```

### Disk space issues
```bash
# Check disk usage
df -h /raid

# Process datasets selectively
python 02_extract_nemotron_embeddings.py \
  --datasets nvidia/Nemotron-Pretraining-Dataset-sample

# Clean up if needed
rm -rf /raid/embeddings/pretraining/cc-v2  # Example
```

### Import errors
```bash
# Install missing dependencies
pip install torch transformers datasets ray humanize
pip install flash-attn --no-build-isolation  # Optional, for Flash Attention 2
```

## 💡 Pro Tips

1. **Always start with the sample dataset** to validate your setup
2. **Monitor disk space** - large datasets can fill drives quickly
3. **Process overnight** - large datasets take many hours
4. **Check logs** - extraction script provides detailed progress
5. **Validate outputs** - check embedding_metadata.json files
6. **Save configs** - document which datasets you've processed

## 📊 Disk Space Planning

| Priority | Datasets | Size | Purpose |
|----------|----------|------|---------|
| High | Sample + Science + Math-v2 | ~5 GB | Testing & validation |
| Medium | All post-training (12 datasets) | ~50-100 GB | Quality data |
| Low | Small pretraining (60-200M) | ~50-200 GB | Additional coverage |
| Optional | Large CC datasets (3.8B, 8.79B) | ~2-3 TB | Massive scale |

## 🎓 Learning Resources

- **Nemotron Documentation**: https://huggingface.co/collections/nvidia
- **Embedding Model**: https://huggingface.co/nvidia/llama-embed-nemotron-8b
- **NeMo Curator**: https://github.com/NVIDIA-NeMo/Curator

## 🚨 Important Notes

⚠️ **Storage**: Full download requires ~4TB of disk space
⚠️ **Time**: Processing all datasets takes multiple days even with 8 GPUs
⚠️ **Testing**: Always test with sample dataset first
⚠️ **Monitoring**: Watch GPU memory, disk space, and processing logs

## ✅ Checklist

Before starting large-scale processing:

- [ ] Tested with sample dataset
- [ ] Verified embeddings output format
- [ ] Confirmed sufficient disk space (check with `df -h /raid`)
- [ ] Documented which datasets are priority
- [ ] Set up monitoring for long-running jobs
- [ ] Have cleanup plan for intermediate files

---

**Last Updated**: January 24, 2026  
**Pipeline Version**: 2.0 (with pretraining datasets support)
