# Environment Setup Guide

This guide helps you set up the environment for the Nemotron Embeddings Pipeline.

## Quick Setup (Recommended)

### Using Existing Deploy Environment (Recommended)

If you already have an `embeddings` environment:

```bash
# Activate your environment
conda activate embeddings

# Or manually:
pip install -r requirements.txt
```

### Manual Setup with New Conda Environment

```bash
# Create environment with Python 3.12 (recommended - avoid 3.14)
conda create -n embeddings python=3.11 -y
conda activate embeddings
# Install PyTorch with CUDA (if you have NVIDIA GPU)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# OR install CPU-only PyTorch (if no GPU)
conda install pytorch cpuonly -c pytorch -y

# Install remaining dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
hf --help
```

### Option 3: Using venv + pip

```bash
# Create virtual environment
python3.11 -m venv venv-nemotron
source venv-nemotron/bin/activate

# Install PyTorch (visit pytorch.org for specific command)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Option 4: Using uv (Fast Package Manager)

```bash
# Create environment with Python 3.11
uv venv --python 3.11 venv-nemotron
source venv-nemotron/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

---

## Python Version Compatibility

### ✅ Recommended Versions
- **Python 3.10** - Stable, widely tested
- **Python 3.11** - Recommended (good performance)
- **Python 3.12** - Supported

### ⚠️ Avoid
- **Python 3.14** - Too new, some packages (like Cython-based ones) may fail

---

## Dependency Overview

### Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| `datasets` | ≥2.14.0 | HuggingFace datasets library |
| `transformers` | ≥4.35.0 | Model loading and inference |
| `torch` | ≥2.0.0 | PyTorch for neural networks |
| `ray` | ≥2.8.0 | Distributed multi-GPU processing |
| `huggingface_hub[cli]` | ≥0.20.0 | HuggingFace CLI (`hf` command) |

### Supporting Libraries
| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24.0 | Array operations |
| `tqdm` | ≥4.65.0 | Progress bars |
| `pyarrow` | ≥14.0.0 | Dataset serialization |

### RAPIDS Packages (CUDA 13.x)
| Package | Version | Purpose |
|---------|---------|---------|
| `cudf-cu13` | ≥24.0.0 | GPU-accelerated DataFrame (like pandas) |
| `cuml-cu13` | ≥24.0.0 | GPU-accelerated machine learning |
| `cugraph-cu13` | ≥24.0.0 | GPU-accelerated graph analytics |
| `cupy-cuda13x` | ≥13.0.0 | GPU-accelerated array library (like numpy) |
| `cuvs-cu13` | ≥24.0.0 | GPU-accelerated vector search |

**Note**: RAPIDS packages should be installed via `conda` using `environment.yml`

### Optional (Performance)
| Package | When to Use |
|---------|-------------|
| `flash-attn` | NVIDIA GPU with compute ≥8.0 (A100, H100, RTX 3090+) |
| `accelerate` | Better model loading and distribution |
| `bitsandbytes` | Model quantization (8-bit, 4-bit) |

---

## Verification

After installation, verify everything works:

```bash
# Check installed versions
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import ray; print(f'Ray: {ray.__version__}')"

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Check HuggingFace CLI
hf --help

# Test the pipeline
cd /localhome/local-tranminhq/embeddings
python 01_verify_nemotron_dataset_status.py
```

---

## Troubleshooting

### Issue: `hf` command not found
```bash
# After installing huggingface_hub[cli], refresh shell hash
hash -r

# Or find where it's installed
which hf
```

### Issue: CUDA out of memory
```bash
# Reduce batch size in config.py
# DEFAULT_BATCH_SIZE = 2  # Try 1 or even smaller
```

### Issue: Import errors
```bash
# Reinstall in clean environment
conda create -n embeddings python=3.12 -y
conda activate embeddings
pip install -r requirements.txt
```

### Issue: TypeError with datasets library (dataclass error)
This happens when `datasets` library version is incompatible with downloaded datasets.

**Quick fix:**
```bash
# Run the version fix script
./check_and_fix_versions.sh

# Or manually downgrade
pip install --force-reinstall 'datasets==2.19.2' 'pyarrow==15.0.2'
```

### Issue: PyTorch not finding GPU
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch matching your CUDA version
# Visit: https://pytorch.org/get-started/locally/
```

---

## Disk Space Requirements

Before proceeding, ensure you have sufficient disk space:

| Component | Space Required |
|-----------|----------------|
| Environment + packages | ~5 GB |
| Post-training datasets | ~4.3 GB |
| Post-training embeddings | ~26 GB |
| Pretraining sample | ~50 MB |
| **Total (post-training)** | **~35 GB** |
| All pretraining datasets | ~24 TB (optional) |

---

## Next Steps

After setup:

1. **Verify datasets**: `python 01_verify_nemotron_dataset_status.py`
2. **Download datasets**: `python 00_download_nemotron_datasets.py`
3. **Explore datasets**: `python 01_explore_nemotron_datasets.py`
4. **Extract embeddings**: See [doc/POST_TRAINING_COMMANDS.md](doc/POST_TRAINING_COMMANDS.md)

---

## Additional Resources

- [Main README](README.md) - Full documentation
- [Quick Reference](doc/QUICK_REFERENCE.md) - Command cheat sheet
- [Post-Training Commands](doc/POST_TRAINING_COMMANDS.md) - Detailed guide for post-training datasets
- [Dataset Distribution](doc/README.md) - Size and statistics overview
