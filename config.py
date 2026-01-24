from typing import Dict, Any

# Mirrors llm-analysis/extract_embeddings_parallel_shards.py (66-79)
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Post-Training Datasets
    "v1": {"hf_name": "nvidia/Nemotron-Post-Training-Dataset-v1", "subdir": "nemotron-v1", "config": None},
    "v2": {"hf_name": "nvidia/Nemotron-Post-Training-Dataset-v2", "subdir": "nemotron-v2", "config": None},
    "llama-sft": {"hf_name": "nvidia/Llama-Nemotron-Post-Training-Dataset", "subdir": "llama-nemotron", "config": "SFT"},
    "llama-rl": {"hf_name": "nvidia/Llama-Nemotron-Post-Training-Dataset", "subdir": "llama-nemotron", "config": "RL"},
    "v3-science": {"hf_name": "nvidia/Nemotron-Science-v1", "subdir": "nemotron-v3/science", "config": None},
    "v3-instruction-chat": {"hf_name": "nvidia/Nemotron-Instruction-Following-Chat-v1", "subdir": "nemotron-v3/instruction-chat", "config": None},
    "v3-math-proofs": {"hf_name": "nvidia/Nemotron-Math-Proofs-v1", "subdir": "nemotron-v3/math-proofs", "config": None},
    "v3-rl-blend": {"hf_name": "nvidia/Nemotron-3-Nano-RL-Training-Blend", "subdir": "nemotron-v3/rl-blend", "config": None},
    "v3-agentic": {"hf_name": "nvidia/Nemotron-Agentic-v1", "subdir": "nemotron-v3/agentic", "config": None},
    "v3-competitive-programming": {"hf_name": "nvidia/Nemotron-Competitive-Programming-v1", "subdir": "nemotron-v3/competitive-programming", "config": None},
    "v3-math": {"hf_name": "nvidia/Nemotron-Math-v2", "subdir": "nemotron-v3/math-v2", "config": None},
    "v3-swe": {"hf_name": "nvidia/Nemotron-SWE-v1", "subdir": "nemotron-v3/swe", "config": None},
    
    # Pretraining Datasets
    "pretrain-sample": {"hf_name": "nvidia/Nemotron-Pretraining-Dataset-sample", "subdir": "pretraining/sample", "config": None},
    "pretrain-cc-code-v1": {"hf_name": "nvidia/Nemotron-CC-Code-v1", "subdir": "pretraining/cc-code-v1", "config": None},
    "pretrain-cc-v2.1": {"hf_name": "nvidia/Nemotron-CC-v2.1", "subdir": "pretraining/cc-v2.1", "config": None},
    "pretrain-code-v2": {"hf_name": "nvidia/Nemotron-Pretraining-Code-v2", "subdir": "pretraining/code-v2", "config": None},
    "pretrain-specialized-v1": {"hf_name": "nvidia/Nemotron-Pretraining-Specialized-v1", "subdir": "pretraining/specialized-v1", "config": None},
    "pretrain-cc-math-v1": {"hf_name": "nvidia/Nemotron-CC-Math-v1", "subdir": "pretraining/cc-math-v1", "config": None},
    "pretrain-cc-v2": {"hf_name": "nvidia/Nemotron-CC-v2", "subdir": "pretraining/cc-v2", "config": None},
    "pretrain-sft-v1": {"hf_name": "nvidia/Nemotron-Pretraining-SFT-v1", "subdir": "pretraining/sft-v1", "config": None},
    "pretrain-code-v1": {"hf_name": "nvidia/Nemotron-Pretraining-Code-v1", "subdir": "pretraining/code-v1", "config": None},
}


MODEL_ID_DEFAULT = "nvidia/llama-embed-nemotron-8b"
MODEL_EMBED_SIZE = 4096
MODEL_MAX_TOKENS = 32768  # per model card: max input sequence length is 32768 tokens

# Default runtime configuration (kept together for easy tuning)
DEFAULT_DATASETS_DIR = "/raid/datasets"
DEFAULT_EMBEDDINGS_DIR = "/raid/embeddings"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = MODEL_MAX_TOKENS
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_DTYPE = "bfloat16"  # "bfloat16" or "float16"
DEFAULT_NUM_WORKERS = 0
DEFAULT_PREFETCH_FACTOR = 8