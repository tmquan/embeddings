#!/usr/bin/env python3
"""
NVIDIA Nemotron Dataset Embedding Pipeline - Multi-GPU DataParallel Version

Processes all NVIDIA Nemotron datasets in /raid/datasets/ to generate embeddings
using nvidia/llama-embed-nemotron-8b model with PyTorch DataParallel across 8 GPUs.
Saves embeddings to /raid/embeddings/ with mirrored directory structure.

Processing strategy:
- Batch size = 1 (one row at a time for memory efficiency)
- Large files (>1GB): Read in chunks of 10,000 rows
- Small files (≤1GB): Load entire file into memory
- All inference distributed across 8 GPUs via DataParallel

Usage:
    python 02_extract_embeddings_dataparallel.py
    python 02_extract_embeddings_dataparallel.py --dataset nvidia_Nemotron-SWE-v1  # Process single dataset
    python 02_extract_embeddings_dataparallel.py --dry-run  # Preview without processing
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterator, Generator
from dataclasses import dataclass
import traceback
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TaskProgressColumn
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler

# Import configuration from existing config.py
from config import (
    MODEL_ID_DEFAULT,
    MODEL_EMBED_SIZE,
    MODEL_MAX_TOKENS,
    DEFAULT_DATASETS_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DTYPE,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = MODEL_ID_DEFAULT
DATASETS_DIR = Path(DEFAULT_DATASETS_DIR)
EMBEDDINGS_DIR = Path(DEFAULT_EMBEDDINGS_DIR)

BATCH_SIZE = 8  # Process 8 rows at a time to utilize all GPUs (DataParallel splits batch across GPUs)
PREFETCH_FACTOR = 4  # Prefetch 4 batches ahead while GPU is processing
CHUNK_SIZE = 10_000  # Read 10,000 rows at a time for large files
FILE_SIZE_THRESHOLD = 1_073_741_824  # 1GB in bytes

NUM_GPUS = 8  # Number of GPUs on single node

# Setup rich console
console = Console()

# Setup logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DatasetInfo:
    """Information about a dataset for processing."""
    name: str
    path: Path
    format: str  # "arrow", "jsonl", "parquet"
    strategy: str  # "concatenate_messages", "input_output", "math_proofs", "combine_columns", "direct_text"
    text_columns: List[str]
    splits: List[Path]  # List of split directories or files


# =============================================================================
# MULTI-GPU MODEL WRAPPER
# =============================================================================

class MultiGPUEmbeddingModel(nn.Module):
    """
    Wrapper for multi-GPU embedding generation.
    
    Uses mean pooling over the sequence (excluding padding tokens) to produce
    a fixed-size embedding vector from the model's last hidden states.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns mean-pooled embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Embeddings [batch_size, hidden_size]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling over sequence (excluding padding)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq, hidden]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        return sum_embeddings / sum_mask


# =============================================================================
# TEXT EXTRACTION STRATEGIES (EXPLICIT IF-ELSE)
# =============================================================================

def extract_text_concatenate_messages(row: Dict[str, Any]) -> str:
    """
    Extract text by concatenating messages from a conversation.
    
    Used by: v1, v2, v3-instruction-chat, v3-agentic, v3-science, 
             v3-competitive-programming, v3-math, v3-swe
    """
    messages = row.get('messages')
    if messages is None:
        return ""
    
    # Handle string-encoded JSON
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages
    
    if not isinstance(messages, list):
        return str(messages)
    
    text_parts = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            # Include reasoning_content if available (e.g., v3-science)
            reasoning = msg.get('reasoning_content')
            if reasoning:
                content = f"{reasoning}\n{content}"
            
            text_parts.append(f'{role}: {content}')
        else:
            text_parts.append(str(msg))
    
    return '\n'.join(text_parts)


def extract_text_input_output(row: Dict[str, Any]) -> str:
    """
    Extract text by combining input and output columns.
    
    Used by: nvidia/Llama-Nemotron-Post-Training-Dataset
    """
    input_val = row.get('input', '')
    output_val = row.get('output', '')
    
    # Handle input as list of messages
    if isinstance(input_val, list):
        input_parts = []
        for msg in input_val:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                input_parts.append(f'{role}: {content}')
            else:
                input_parts.append(str(msg))
        input_text = '\n'.join(input_parts)
    elif isinstance(input_val, str):
        input_text = f'user: {input_val}'
    else:
        input_text = str(input_val) if input_val else ''
    
    output_text = f'assistant: {output_val}' if output_val else ''
    
    return f"{input_text}\n{output_text}".strip()


def extract_text_math_proofs(row: Dict[str, Any]) -> str:
    """
    Extract text from math proofs dataset by combining multiple columns.
    
    Used by: nvidia/Nemotron-Math-Proofs-v1
    Columns: problem, formal_statement, lean_header, messages
    """
    parts = []
    
    if row.get('problem'):
        parts.append(f"Problem: {row['problem']}")
    
    if row.get('formal_statement'):
        parts.append(f"Formal Statement: {row['formal_statement']}")
    
    if row.get('lean_header'):
        parts.append(f"Lean Header: {row['lean_header']}")
    
    if row.get('messages'):
        messages_text = extract_text_concatenate_messages(row)
        if messages_text:
            parts.append(f"Messages:\n{messages_text}")
    
    return '\n\n'.join(parts)


def extract_text_combine_columns(row: Dict[str, Any], columns: List[str]) -> str:
    """
    Extract text by combining specified columns.
    
    Used by: nvidia/Nemotron-3-Nano-RL-Training-Blend
    """
    parts = []
    
    for col in columns:
        value = row.get(col)
        if value is not None:
            if isinstance(value, (dict, list)):
                try:
                    value = json.dumps(value, ensure_ascii=False)
                except:
                    value = str(value)
            parts.append(f"{col}: {value}")
    
    return '\n'.join(parts)


def extract_text_direct(row: Dict[str, Any], column: str = 'text') -> str:
    """
    Extract text directly from a single column.
    
    Used by: nvidia/Nemotron-Pretraining-Dataset-sample
    """
    value = row.get(column, '')
    return str(value) if value else ''


def extract_text_for_dataset(row: Dict[str, Any], dataset_name: str) -> str:
    """
    Extract text from a row based on dataset name using explicit if-else.
    
    This function implements explicit handling for each known dataset type
    for easy inspection and modification.
    
    Args:
        row: Dictionary containing row data
        dataset_name: Name of the dataset directory (e.g., "nvidia_Nemotron-Post-Training-Dataset-v1")
        
    Returns:
        Extracted text string for embedding
    """
    dataset_lower = dataset_name.lower()
    
    # =========================================================================
    # POST-TRAINING DATASETS (v1, v2) - Arrow format, messages column
    # =========================================================================
    if 'post-training-dataset-v1' in dataset_lower:
        return extract_text_concatenate_messages(row)
    
    elif 'post-training-dataset-v2' in dataset_lower:
        return extract_text_concatenate_messages(row)
    
    # =========================================================================
    # LLAMA-NEMOTRON - Arrow format, input/output columns
    # =========================================================================
    elif 'llama-nemotron' in dataset_lower or 'llama_nemotron' in dataset_lower:
        return extract_text_input_output(row)
    
    # =========================================================================
    # V3 SPECIALIZED DATASETS
    # =========================================================================
    elif 'instruction-following' in dataset_lower or 'instruction_following' in dataset_lower:
        # nvidia/Nemotron-Instruction-Following-Chat-v1 - Arrow, messages
        return extract_text_concatenate_messages(row)
    
    elif 'agentic' in dataset_lower:
        # nvidia/Nemotron-Agentic-v1 - JSONL, messages
        return extract_text_concatenate_messages(row)
    
    elif 'science' in dataset_lower:
        # nvidia/Nemotron-Science-v1 - Arrow, messages (with reasoning_content)
        return extract_text_concatenate_messages(row)
    
    elif 'math-proofs' in dataset_lower or 'math_proofs' in dataset_lower:
        # nvidia/Nemotron-Math-Proofs-v1 - Arrow, problem+formal_statement+lean_header+messages
        return extract_text_math_proofs(row)
    
    elif 'competitive-programming' in dataset_lower or 'competitive_programming' in dataset_lower:
        # nvidia/Nemotron-Competitive-Programming-v1 - JSONL, messages
        return extract_text_concatenate_messages(row)
    
    elif 'nemotron-math' in dataset_lower or 'nemotron_math' in dataset_lower:
        # nvidia/Nemotron-Math-v2 - JSONL, messages
        return extract_text_concatenate_messages(row)
    
    elif 'swe' in dataset_lower:
        # nvidia/Nemotron-SWE-v1 - Arrow, messages
        return extract_text_concatenate_messages(row)
    
    elif 'rl-training-blend' in dataset_lower or 'nano-rl' in dataset_lower or '3-nano' in dataset_lower:
        # nvidia/Nemotron-3-Nano-RL-Training-Blend - JSONL, responses_create_params
        return extract_text_combine_columns(row, ['responses_create_params'])
    
    # =========================================================================
    # PRETRAINING DATASETS - Parquet format, text column
    # =========================================================================
    elif 'pretraining' in dataset_lower or 'pretrain' in dataset_lower:
        return extract_text_direct(row, 'text')
    
    # =========================================================================
    # FALLBACK: Try common patterns
    # =========================================================================
    else:
        # Try messages first (most common)
        if 'messages' in row and row['messages']:
            return extract_text_concatenate_messages(row)
        
        # Try input/output
        if 'input' in row and 'output' in row:
            return extract_text_input_output(row)
        
        # Try direct text
        if 'text' in row and row['text']:
            return extract_text_direct(row)
        
        # Last resort: concatenate all string values
        parts = []
        for key, value in row.items():
            if isinstance(value, str) and len(value) > 20:
                parts.append(f"{key}: {value[:500]}")
        
        return '\n'.join(parts) if parts else str(row)


# =============================================================================
# FILE LOADING UTILITIES
# =============================================================================

def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return file_path.stat().st_size


def should_chunk_file(file_path: Path) -> bool:
    """Determine if file should be chunked based on size."""
    return get_file_size(file_path) > FILE_SIZE_THRESHOLD


def load_arrow_file(file_path: Path) -> pa.Table:
    """
    Load an Arrow IPC file (HuggingFace datasets format).
    
    HuggingFace uses Arrow IPC Stream format.
    """
    try:
        with pa.memory_map(str(file_path), 'r') as source:
            reader = ipc.open_stream(source)
            return reader.read_all()
    except Exception:
        # Fallback: try as IPC file format
        with pa.memory_map(str(file_path), 'r') as source:
            reader = ipc.open_file(source)
            return reader.read_all()


def iterate_arrow_rows(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Iterate over rows in an Arrow file."""
    table = load_arrow_file(file_path)
    columns = table.column_names
    
    for i in range(table.num_rows):
        row = {col: table.column(col)[i].as_py() for col in columns}
        yield row


def iterate_jsonl_rows(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Iterate over rows in a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iterate_parquet_rows(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Iterate over rows in a Parquet file."""
    table = pq.read_table(file_path)
    columns = table.column_names
    
    for i in range(table.num_rows):
        row = {col: table.column(col)[i].as_py() for col in columns}
        yield row


def iterate_parquet_rows_chunked(file_path: Path, chunk_size: int = CHUNK_SIZE) -> Generator[Tuple[int, List[Dict[str, Any]]], None, None]:
    """
    Iterate over rows in a Parquet file in chunks.
    
    Yields (chunk_index, list_of_rows) tuples.
    """
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    
    chunk_idx = 0
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        table = pa.Table.from_batches([batch])
        columns = table.column_names
        
        rows = []
        for i in range(table.num_rows):
            row = {col: table.column(col)[i].as_py() for col in columns}
            rows.append(row)
        
        yield chunk_idx, rows
        chunk_idx += 1


def count_rows_in_file(file_path: Path, file_format: str) -> int:
    """Count the number of rows in a file."""
    if file_format == "arrow":
        table = load_arrow_file(file_path)
        return table.num_rows
    elif file_format == "jsonl":
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    elif file_format == "parquet":
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows
    else:
        return 0


# =============================================================================
# DATASET DISCOVERY
# =============================================================================

def detect_dataset_format(dataset_path: Path) -> Tuple[str, List[Path]]:
    """
    Detect the format of a dataset and find all data files.
    
    Handles various directory structures:
    - Flat: dataset/*.arrow
    - One level: dataset/split/*.arrow
    - Nested: dataset/SFT/split/*.arrow (Llama-Nemotron style)
    
    Returns:
        Tuple of (format, list_of_files)
        format: "arrow", "jsonl", or "parquet"
    """
    files = []
    detected_format = None
    
    # Check for Arrow files recursively (handles nested structures like SFT/chat/*.arrow)
    arrow_files = list(dataset_path.rglob("*.arrow"))
    if arrow_files:
        return "arrow", sorted(arrow_files)
    
    # Check for JSONL files
    data_dir = dataset_path / "data"
    if data_dir.exists():
        jsonl_files = list(data_dir.glob("*.jsonl"))
        if jsonl_files:
            return "jsonl", sorted(jsonl_files)
    
    # Check recursively for JSONL
    jsonl_files = list(dataset_path.rglob("*.jsonl"))
    if jsonl_files:
        return "jsonl", sorted(jsonl_files)
    
    # Check for Parquet files recursively
    parquet_files = list(dataset_path.rglob("*.parquet"))
    if parquet_files:
        return "parquet", sorted(parquet_files)
    
    return None, []


def get_dataset_sort_priority(dataset_name: str) -> Tuple[int, str]:
    """
    Get sort priority for a dataset.
    
    Priority order:
    1. Llama-Nemotron (llama-sft)
    2. Post-Training v1
    3. Post-Training v2
    4. v3 datasets (alphabetically)
    5. Other datasets (alphabetically)
    
    Returns:
        Tuple of (priority_number, dataset_name) for sorting
    """
    name_lower = dataset_name.lower()
    
    # Priority 1: Llama-Nemotron (llama-sft)
    if 'llama-nemotron' in name_lower or 'llama_nemotron' in name_lower:
        return (1, dataset_name)
    
    # Priority 2: Post-Training v1
    if 'post-training-dataset-v1' in name_lower or 'post_training_dataset_v1' in name_lower:
        return (2, dataset_name)
    
    # Priority 3: Post-Training v2
    if 'post-training-dataset-v2' in name_lower or 'post_training_dataset_v2' in name_lower:
        return (3, dataset_name)
    
    # Priority 4: v3 datasets (v1 suffix means v3 generation for specialized datasets)
    v3_keywords = [
        'science', 'instruction-following', 'math-proofs', 'agentic',
        'competitive-programming', 'math-v2', 'swe', 'rl-training-blend', '3-nano'
    ]
    for keyword in v3_keywords:
        if keyword in name_lower:
            return (4, dataset_name)
    
    # Priority 5: Pretraining and other datasets
    return (5, dataset_name)


def discover_datasets(datasets_dir: Path, filter_dataset: Optional[str] = None) -> List[DatasetInfo]:
    """
    Discover all datasets in the datasets directory.
    
    Datasets are sorted in processing order:
    1. Llama-Nemotron (llama-sft)
    2. Post-Training v1
    3. Post-Training v2
    4. v3 specialized datasets
    5. Other datasets (pretraining, etc.)
    
    Args:
        datasets_dir: Path to datasets directory
        filter_dataset: Optional dataset name to process only that dataset
        
    Returns:
        List of DatasetInfo objects sorted by priority
    """
    datasets = []
    
    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return datasets
    
    for item in datasets_dir.iterdir():
        if not item.is_dir() or item.name.startswith('.'):
            continue
        
        # Filter by dataset name if specified
        if filter_dataset and filter_dataset not in item.name:
            continue
        
        # Detect format and find files
        file_format, files = detect_dataset_format(item)
        
        if not files:
            logger.warning(f"No data files found in {item.name}")
            continue
        
        # Determine extraction strategy based on dataset name
        dataset_lower = item.name.lower()
        
        if 'llama-nemotron' in dataset_lower or 'llama_nemotron' in dataset_lower:
            strategy = "input_output"
            text_columns = ["input", "output"]
        elif 'math-proofs' in dataset_lower or 'math_proofs' in dataset_lower:
            strategy = "math_proofs"
            text_columns = ["problem", "formal_statement", "lean_header", "messages"]
        elif 'rl-training-blend' in dataset_lower or 'nano-rl' in dataset_lower or '3-nano' in dataset_lower:
            strategy = "combine_columns"
            text_columns = ["responses_create_params"]
        elif 'pretraining' in dataset_lower or 'pretrain' in dataset_lower:
            strategy = "direct_text"
            text_columns = ["text"]
        else:
            strategy = "concatenate_messages"
            text_columns = ["messages"]
        
        datasets.append(DatasetInfo(
            name=item.name,
            path=item,
            format=file_format,
            strategy=strategy,
            text_columns=text_columns,
            splits=files
        ))
    
    # Sort datasets by priority: llama-sft -> v1 -> v2 -> v3 -> others
    datasets.sort(key=lambda ds: get_dataset_sort_priority(ds.name))
    
    return datasets


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_embedding_model(num_gpus: int = NUM_GPUS) -> Tuple[nn.Module, Any, int]:
    """
    Load the embedding model and tokenizer with multi-GPU DataParallel support.
    
    Returns:
        Tuple of (model, tokenizer, num_gpus_used)
    """
    console.print(Panel(f"[bold cyan]Loading Model: {MODEL_ID}[/bold cyan]"))
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPUs.")
    
    num_available_gpus = torch.cuda.device_count()
    console.print(f"[cyan]Detected {num_available_gpus} GPU(s)[/cyan]")
    
    if num_available_gpus < num_gpus:
        console.print(f"[yellow]Warning: Expected {num_gpus} GPUs but found {num_available_gpus}. "
                     f"Using {num_available_gpus} GPU(s).[/yellow]")
        num_gpus = num_available_gpus
    
    # Determine dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(DEFAULT_DTYPE, torch.bfloat16)
    
    # Load tokenizer
    console.print("[cyan]Loading tokenizer...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate attention implementation
    # This model only supports flash_attention_2 or eager (not sdpa)
    console.print("[cyan]Loading model weights...[/cyan]")
    
    # Try flash_attention_2 first (faster), fall back to eager
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        console.print("[cyan]Using flash_attention_2 implementation[/cyan]")
    except ImportError:
        attn_impl = "eager"
        console.print("[yellow]flash_attn not installed, using eager attention (slower)[/yellow]")
    
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl
    )
    
    model = model.cuda()
    model.eval()
    
    # Wrap model for embedding extraction
    embedding_model = MultiGPUEmbeddingModel(model)
    
    # Use DataParallel for multi-GPU inference
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        embedding_model = nn.DataParallel(embedding_model, device_ids=device_ids)
        console.print(f"[green]Model loaded with DataParallel across GPUs: {device_ids}[/green]")
    else:
        console.print(f"[green]Model loaded on single GPU[/green]")
    
    return embedding_model, tokenizer, num_gpus


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

@torch.no_grad()
def generate_embeddings_batch(
    texts: List[str],
    model: nn.Module,
    tokenizer: Any,
    max_length: int = MODEL_MAX_TOKENS
) -> List[np.ndarray]:
    """
    Generate embeddings for a batch of texts.
    
    This function processes multiple texts at once, which allows DataParallel
    to distribute the work across all GPUs (each GPU processes batch_size/num_gpus samples).
    
    Args:
        texts: List of input text strings
        model: Embedding model (with DataParallel wrapper)
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        List of embeddings as float32 numpy arrays [embed_dim]
    """
    if not texts:
        return []
    
    # Handle empty texts by tracking their indices
    valid_indices = []
    valid_texts = []
    results = [None] * len(texts)
    
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_indices.append(i)
            valid_texts.append(text)
        else:
            results[i] = np.zeros(MODEL_EMBED_SIZE, dtype=np.float32)
    
    if not valid_texts:
        return [np.zeros(MODEL_EMBED_SIZE, dtype=np.float32) for _ in texts]
    
    try:
        # Tokenize all valid texts together
        inputs = tokenizer(
            valid_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to GPU
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        
        # Generate embeddings (DataParallel distributes across GPUs)
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Convert to float32 numpy
        embeddings_np = embeddings.cpu().float().numpy().astype(np.float32)
        
        # Place results back in correct positions
        for idx, embedding in zip(valid_indices, embeddings_np):
            results[idx] = embedding
        
        return results
    
    except Exception as e:
        logger.warning(f"Error generating batch embeddings: {e}")
        # Return zeros for all texts on error
        return [np.zeros(MODEL_EMBED_SIZE, dtype=np.float32) for _ in texts]


@torch.no_grad()
def generate_embedding(
    text: str,
    model: nn.Module,
    tokenizer: Any,
    max_length: int = MODEL_MAX_TOKENS
) -> np.ndarray:
    """
    Generate embedding for a single text (wrapper for batch function).
    
    For better GPU utilization, prefer using generate_embeddings_batch() with
    batch_size >= num_gpus.
    """
    results = generate_embeddings_batch([text], model, tokenizer, max_length)
    return results[0]


# =============================================================================
# PREFETCHING BATCH ITERATOR
# =============================================================================

class PrefetchBatchIterator:
    """
    Iterator that prefetches and prepares batches in a background thread.
    
    While the GPU is processing the current batch, this iterator prepares
    the next N batches (where N = prefetch_factor) in advance, including
    text extraction and tokenization.
    
    This overlaps CPU work (data loading, text extraction, tokenization)
    with GPU work (inference), improving overall throughput.
    """
    
    def __init__(
        self,
        row_iterator: Iterator[Dict[str, Any]],
        dataset_name: str,
        tokenizer: Any,
        batch_size: int = BATCH_SIZE,
        prefetch_factor: int = PREFETCH_FACTOR,
        max_length: int = MODEL_MAX_TOKENS
    ):
        """
        Initialize the prefetch iterator.
        
        Args:
            row_iterator: Iterator yielding row dictionaries
            dataset_name: Name of dataset for text extraction
            tokenizer: Tokenizer for preparing inputs
            batch_size: Number of rows per batch
            prefetch_factor: Number of batches to prefetch ahead
            max_length: Maximum sequence length for tokenization
        """
        self.row_iterator = row_iterator
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.max_length = max_length
        
        # Queue to hold prefetched batches
        # Each item: (batch_texts, batch_counters, batch_inputs) or None for end
        self.queue = queue.Queue(maxsize=prefetch_factor)
        
        # Counter for file numbering
        self.file_counter = 1
        
        # Flag to signal worker to stop
        self.stop_event = threading.Event()
        
        # Start prefetch worker thread
        self.worker = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.worker.start()
    
    def _prefetch_worker(self):
        """Background worker that prepares batches ahead of time."""
        batch_texts = []
        batch_counters = []
        
        try:
            for row in self.row_iterator:
                if self.stop_event.is_set():
                    break
                
                # Extract text
                text = extract_text_for_dataset(row, self.dataset_name)
                batch_texts.append(text)
                batch_counters.append(self.file_counter)
                self.file_counter += 1
                
                # When batch is full, tokenize and queue it
                if len(batch_texts) >= self.batch_size:
                    # Tokenize the batch (CPU work done in background)
                    batch_inputs = self._tokenize_batch(batch_texts)
                    
                    # Put in queue (blocks if queue is full)
                    self.queue.put((batch_texts, batch_counters, batch_inputs))
                    
                    batch_texts = []
                    batch_counters = []
            
            # Handle remaining rows
            if batch_texts and not self.stop_event.is_set():
                batch_inputs = self._tokenize_batch(batch_texts)
                self.queue.put((batch_texts, batch_counters, batch_inputs))
        
        except Exception as e:
            logger.error(f"Prefetch worker error: {e}")
        
        finally:
            # Signal end of data
            self.queue.put(None)
    
    def _tokenize_batch(self, texts: List[str]) -> Optional[Dict[str, Any]]:
        """Tokenize a batch of texts."""
        # Filter empty texts
        valid_texts = [t if t and t.strip() else " " for t in texts]
        
        try:
            inputs = self.tokenizer(
                valid_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            return inputs
        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            return None
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[List[str], List[int], Optional[Dict[str, Any]]]:
        """
        Get next prefetched batch.
        
        Returns:
            Tuple of (batch_texts, batch_counters, tokenized_inputs)
        """
        item = self.queue.get()
        
        if item is None:
            raise StopIteration
        
        return item
    
    def stop(self):
        """Stop the prefetch worker."""
        self.stop_event.set()
        # Drain the queue to unblock worker
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass


@torch.no_grad()
def generate_embeddings_from_inputs(
    inputs: Dict[str, Any],
    model: nn.Module,
    num_texts: int
) -> List[np.ndarray]:
    """
    Generate embeddings from pre-tokenized inputs.
    
    Args:
        inputs: Tokenized inputs dict with input_ids and attention_mask
        model: Embedding model
        num_texts: Number of texts (for fallback on error)
        
    Returns:
        List of embeddings as float32 numpy arrays
    """
    if inputs is None:
        return [np.zeros(MODEL_EMBED_SIZE, dtype=np.float32) for _ in range(num_texts)]
    
    try:
        # Move to GPU
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        
        # Generate embeddings
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Convert to float32 numpy
        return list(embeddings.cpu().float().numpy().astype(np.float32))
    
    except Exception as e:
        logger.warning(f"Error generating embeddings from inputs: {e}")
        return [np.zeros(MODEL_EMBED_SIZE, dtype=np.float32) for _ in range(num_texts)]


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_dataset(
    dataset: DatasetInfo,
    model: nn.Module,
    tokenizer: Any,
    embeddings_dir: Path,
    progress: Progress,
    overall_task_id: int,
    split_task_id: int,
    row_task_id: int,
    save_task_id: int,
    batch_size: int = BATCH_SIZE,
    prefetch_factor: int = PREFETCH_FACTOR,
) -> Tuple[int, int, List[str]]:
    """
    Process a single dataset and generate embeddings using batched inference with prefetching.
    
    Output format depends on source file type:
    - Arrow/Parquet: 1 .npy file per source file (shape: [num_rows, 4096])
    - JSONL: Multiple .npy files chunked by CHUNK_SIZE (shape: [CHUNK_SIZE, 4096])
    
    Uses batch_size >= num_gpus to ensure all GPUs are utilized via DataParallel.
    Prefetches batches in background thread to overlap CPU and GPU work.
    
    Returns:
        Tuple of (total_processed, total_errors, list_of_error_messages)
    """
    total_processed = 0
    total_errors = 0
    error_messages = []
    
    console.print(f"\n[bold cyan]Processing: {dataset.name}[/bold cyan]")
    console.print(f"  Format: {dataset.format}")
    console.print(f"  Strategy: {dataset.strategy}")
    console.print(f"  Files: {len(dataset.splits)}")
    console.print(f"  Batch size: {batch_size} (distributes across GPUs)")
    console.print(f"  Prefetch: {prefetch_factor} batches ahead")
    
    if dataset.format == "jsonl":
        console.print(f"  Output: Chunked .npy files (shape: [{CHUNK_SIZE}, {MODEL_EMBED_SIZE}])")
    else:
        console.print(f"  Output: 1 .npy file per source file (shape: [num_rows, {MODEL_EMBED_SIZE}])")
    
    # Update split progress bar
    progress.update(split_task_id, total=len(dataset.splits), completed=0, 
                   description=f"[green]Files in {dataset.name[:30]}...")
    
    # Process each file in the dataset
    for file_idx, file_path in enumerate(dataset.splits):
        prefetch_iter = None
        try:
            # Determine output path (mirror structure)
            relative_path = file_path.relative_to(dataset.path)
            output_dir = embeddings_dir / dataset.name / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Count rows for progress bar
            num_rows = count_rows_in_file(file_path, dataset.format)
            progress.update(row_task_id, total=num_rows, completed=0,
                          description=f"[yellow]Embedding {file_path.name[:40]}...")
            
            # For JSONL: check if first chunk already exists to skip
            # For Arrow/Parquet: check if output file exists
            if dataset.format == "jsonl":
                first_chunk_file = output_dir / f"{file_path.stem}_000000.npy"
                num_expected_chunks = (num_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
                progress.update(save_task_id, total=num_expected_chunks, completed=0,
                              description=f"[magenta]Saving {file_path.stem}_*.npy...")
            else:
                output_file = output_dir / f"{file_path.stem}.npy"
                if output_file.exists():
                    console.print(f"  [yellow]Skipping {file_path.name} (already exists)[/yellow]")
                    progress.update(split_task_id, advance=1)
                    continue
                progress.update(save_task_id, total=1, completed=0,
                              description=f"[magenta]Saving {file_path.stem}.npy...")
            
            # Get row iterator based on format
            if dataset.format == "arrow":
                row_iterator = iterate_arrow_rows(file_path)
            elif dataset.format == "jsonl":
                row_iterator = iterate_jsonl_rows(file_path)
            elif dataset.format == "parquet":
                row_iterator = iterate_parquet_rows(file_path)
            else:
                logger.warning(f"Unknown format: {dataset.format}")
                continue
            
            # Create prefetching iterator - prepares batches in background
            prefetch_iter = PrefetchBatchIterator(
                row_iterator=row_iterator,
                dataset_name=dataset.name,
                tokenizer=tokenizer,
                batch_size=batch_size,
                prefetch_factor=prefetch_factor
            )
            
            # Accumulate embeddings
            all_embeddings = []
            chunk_idx = 0
            
            # Process prefetched batches
            for batch_texts, batch_counters, batch_inputs in prefetch_iter:
                try:
                    # Generate embeddings using pre-tokenized inputs (GPU work)
                    embeddings = generate_embeddings_from_inputs(
                        batch_inputs, model, len(batch_texts)
                    )
                    
                    # Accumulate embeddings
                    all_embeddings.extend(embeddings)
                    
                    progress.update(row_task_id, advance=len(batch_texts))
                    total_processed += len(batch_texts)
                    
                    # For JSONL: save in chunks of CHUNK_SIZE
                    if dataset.format == "jsonl":
                        while len(all_embeddings) >= CHUNK_SIZE:
                            # Extract chunk
                            chunk_embeddings = all_embeddings[:CHUNK_SIZE]
                            all_embeddings = all_embeddings[CHUNK_SIZE:]
                            
                            # Save chunk
                            chunk_file = output_dir / f"{file_path.stem}_{chunk_idx:06d}.npy"
                            embeddings_array = np.stack(chunk_embeddings, axis=0).astype(np.float32)
                            np.save(chunk_file, embeddings_array)
                            console.print(f"  [green]Saved {chunk_file.name}: shape {embeddings_array.shape}[/green]")
                            
                            progress.update(save_task_id, advance=1)
                            chunk_idx += 1
                    
                except Exception as e:
                    # On error, add zero embeddings to maintain row alignment
                    all_embeddings.extend([np.zeros(MODEL_EMBED_SIZE, dtype=np.float32) 
                                          for _ in range(len(batch_texts))])
                    total_errors += len(batch_texts)
                    if total_errors <= 10:
                        error_messages.append(f"Batch error in {file_path.name}: {str(e)[:100]}")
                    progress.update(row_task_id, advance=len(batch_texts))
            
            # Save remaining embeddings
            if all_embeddings:
                embeddings_array = np.stack(all_embeddings, axis=0).astype(np.float32)
                
                if dataset.format == "jsonl":
                    # Save final chunk (may be smaller than CHUNK_SIZE)
                    chunk_file = output_dir / f"{file_path.stem}_{chunk_idx:06d}.npy"
                    np.save(chunk_file, embeddings_array)
                    console.print(f"  [green]Saved {chunk_file.name}: shape {embeddings_array.shape}[/green]")
                else:
                    # Arrow/Parquet: save as single file
                    output_file = output_dir / f"{file_path.stem}.npy"
                    np.save(output_file, embeddings_array)
                    console.print(f"  [green]Saved {output_file.name}: shape {embeddings_array.shape}[/green]")
                
                progress.update(save_task_id, advance=1)
            
            progress.update(split_task_id, advance=1)
            
        except Exception as e:
            total_errors += 1
            error_msg = f"File error {file_path.name}: {str(e)[:200]}"
            error_messages.append(error_msg)
            logger.error(error_msg)
            progress.update(split_task_id, advance=1)
        
        finally:
            # Clean up prefetch iterator
            if prefetch_iter is not None:
                prefetch_iter.stop()
    
    return total_processed, total_errors, error_messages


def process_all_datasets(
    datasets_dir: Path = DATASETS_DIR,
    embeddings_dir: Path = EMBEDDINGS_DIR,
    filter_dataset: Optional[str] = None,
    dry_run: bool = False,
    num_gpus: int = NUM_GPUS,
    batch_size: int = BATCH_SIZE,
    prefetch_factor: int = PREFETCH_FACTOR
) -> None:
    """
    Main function to process all datasets with multi-GPU support.
    
    Args:
        datasets_dir: Path to input datasets
        embeddings_dir: Path to output embeddings
        filter_dataset: Optional dataset name filter
        dry_run: If True, only show what would be processed
        num_gpus: Number of GPUs to use
        batch_size: Batch size for inference (should be >= num_gpus)
        prefetch_factor: Number of batches to prefetch ahead
    """
    console.print(Panel.fit(
        "[bold green]NVIDIA Nemotron Embedding Pipeline[/bold green]\n"
        f"Input: {datasets_dir}\n"
        f"Output: {embeddings_dir}\n"
        f"GPUs: {num_gpus}\n"
        f"Batch size: {batch_size}\n"
        f"Prefetch: {prefetch_factor} batches",
        title="Configuration"
    ))
    
    # Discover datasets
    console.print("\n[cyan]Discovering datasets...[/cyan]")
    datasets = discover_datasets(datasets_dir, filter_dataset)
    
    if not datasets:
        console.print("[red]No datasets found to process.[/red]")
        return
    
    # Display dataset summary
    table = Table(title="Datasets to Process")
    table.add_column("Dataset", style="cyan")
    table.add_column("Format", style="green")
    table.add_column("Strategy", style="yellow")
    table.add_column("Files", justify="right")
    
    for ds in datasets:
        table.add_row(ds.name[:50], ds.format, ds.strategy, str(len(ds.splits)))
    
    console.print(table)
    
    if dry_run:
        console.print("\n[yellow]Dry run - no processing performed.[/yellow]")
        return
    
    # Load model
    model, tokenizer, actual_gpus = load_embedding_model(num_gpus)
    
    console.print(f"\n[cyan]Processing Configuration:[/cyan]")
    console.print(f"  Batch size: {batch_size} (distributes across {actual_gpus} GPUs)")
    console.print(f"  Prefetch factor: {prefetch_factor} batches ahead")
    console.print(f"  Chunk size: {CHUNK_SIZE:,} rows (for large files)")
    console.print(f"  File size threshold: {FILE_SIZE_THRESHOLD / 1e9:.1f} GB")
    console.print(f"  Embedding dimension: {MODEL_EMBED_SIZE}")
    console.print(f"  Max sequence length: {MODEL_MAX_TOKENS:,} tokens")
    
    if batch_size < actual_gpus:
        console.print(f"  [yellow]Warning: batch_size ({batch_size}) < num_gpus ({actual_gpus}). "
                     f"Some GPUs may be idle. Consider --batch-size {actual_gpus} or higher.[/yellow]")
    
    # Create embeddings directory
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Track overall statistics
    total_datasets_processed = 0
    total_rows_processed = 0
    total_errors = 0
    failed_datasets = []
    
    # Create progress bars
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2
    ) as progress:
        
        # Overall dataset progress
        overall_task = progress.add_task(
            "[cyan]Overall Progress",
            total=len(datasets)
        )
        
        # Split/file progress within dataset
        split_task = progress.add_task(
            "[green]Processing Files",
            total=0
        )
        
        # Row-level inference progress
        row_task = progress.add_task(
            "[yellow]Generating Embeddings",
            total=0
        )
        
        # Save progress
        save_task = progress.add_task(
            "[magenta]Saving Embeddings",
            total=0
        )
        
        # Process each dataset
        for dataset in datasets:
            progress.update(overall_task, 
                          description=f"[cyan]Dataset: {dataset.name[:40]}...")
            
            try:
                processed, errors, error_msgs = process_dataset(
                    dataset=dataset,
                    model=model,
                    tokenizer=tokenizer,
                    embeddings_dir=embeddings_dir,
                    progress=progress,
                    overall_task_id=overall_task,
                    split_task_id=split_task,
                    row_task_id=row_task,
                    save_task_id=save_task,
                    batch_size=batch_size,
                    prefetch_factor=prefetch_factor,
                )
                
                total_rows_processed += processed
                total_errors += errors
                total_datasets_processed += 1
                
                if errors > 0:
                    console.print(f"[yellow]  Completed with {errors} errors[/yellow]")
                    for msg in error_msgs[:5]:
                        console.print(f"[yellow]    - {msg}[/yellow]")
                else:
                    console.print(f"[green]  Completed: {processed:,} embeddings[/green]")
                
            except Exception as e:
                failed_datasets.append((dataset.name, str(e)))
                console.print(f"[red]  FAILED: {e}[/red]")
                logger.exception(f"Failed to process dataset {dataset.name}")
            
            progress.update(overall_task, advance=1)
    
    # Print summary
    console.print("\n" + "=" * 60)
    console.print(Panel.fit(
        f"[bold green]Processing Complete[/bold green]\n\n"
        f"Datasets processed: {total_datasets_processed}/{len(datasets)}\n"
        f"Total embeddings: {total_rows_processed:,}\n"
        f"Total errors: {total_errors:,}\n"
        f"Output directory: {embeddings_dir}",
        title="Summary"
    ))
    
    if failed_datasets:
        console.print("\n[bold red]Failed Datasets:[/bold red]")
        for name, error in failed_datasets:
            console.print(f"  [red]- {name}: {error[:100]}[/red]")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for NVIDIA Nemotron datasets using multi-GPU DataParallel",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DATASETS_DIR,
        help=f"Input datasets directory (default: {DATASETS_DIR})"
    )
    
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=EMBEDDINGS_DIR,
        help=f"Output embeddings directory (default: {EMBEDDINGS_DIR})"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process only datasets matching this name (partial match)"
    )
    
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=NUM_GPUS,
        help=f"Number of GPUs to use (default: {NUM_GPUS})"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for inference - should be >= num_gpus for full GPU utilization (default: {BATCH_SIZE})"
    )
    
    parser.add_argument(
        "--prefetch",
        type=int,
        default=PREFETCH_FACTOR,
        help=f"Number of batches to prefetch ahead while GPU is processing (default: {PREFETCH_FACTOR})"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview datasets without processing"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        process_all_datasets(
            datasets_dir=args.datasets_dir,
            embeddings_dir=args.embeddings_dir,
            filter_dataset=args.dataset,
            dry_run=args.dry_run,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
            prefetch_factor=args.prefetch
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
