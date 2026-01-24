#!/usr/bin/env python3
"""
Nemotron Embedding Extraction Pipeline

Scans /raid/datasets for downloaded Nemotron datasets, collects statistics,
and extracts embeddings using llama-embed-nemotron-8b with 32K max_length.

Supports three data formats:
- Arrow (HuggingFace datasets): load_from_disk
- JSONL: Iterative line-by-line processing
- Parquet: PyArrow iter_batches

Outputs:
- Per-dataset statistics JSON
- Embeddings as numpy float32 shards
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import gc

import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration
from config import (
    DATASET_CONFIGS,
    DEFAULT_DATASETS_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DTYPE,
    MODEL_ID_DEFAULT,
    MODEL_EMBED_SIZE,
)


# =============================================================================
# Data Structures
# =============================================================================

class DatasetFormat(Enum):
    """Supported dataset formats."""
    ARROW = "arrow"
    JSONL = "jsonl"
    PARQUET = "parquet"
    UNKNOWN = "unknown"


@dataclass
class SplitInfo:
    """Information about a dataset split."""
    name: str
    rows: int
    files: List[str] = field(default_factory=list)


@dataclass
class DatasetInfo:
    """Information about a discovered dataset."""
    name: str
    path: str
    format: DatasetFormat
    total_rows: int = 0
    splits: Dict[str, SplitInfo] = field(default_factory=dict)
    columns: List[str] = field(default_factory=list)
    text_columns: List[str] = field(default_factory=list)
    embedding_strategy: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "format": self.format.value,
            "total_rows": self.total_rows,
            "splits": {
                k: {"name": v.name, "rows": v.rows, "files": v.files}
                for k, v in self.splits.items()
            },
            "columns": self.columns,
            "text_columns": self.text_columns,
            "embedding_strategy": self.embedding_strategy,
        }


# =============================================================================
# Dataset Discovery
# =============================================================================

def detect_format(dataset_path: Path) -> DatasetFormat:
    """
    Detect the format of a dataset directory.
    
    Priority:
    1. Arrow files in subdirectories (HuggingFace format)
    2. JSONL files in data/ subdirectory or root directory
    3. Parquet files
    """
    # Check for HuggingFace Arrow format (subdirs with .arrow files)
    for subdir in dataset_path.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            arrow_files = list(subdir.glob("*.arrow"))
            if arrow_files:
                return DatasetFormat.ARROW
    
    # Check for JSONL in data/ subdirectory
    data_dir = dataset_path / "data"
    if data_dir.exists():
        jsonl_files = list(data_dir.glob("*.jsonl"))
        if jsonl_files:
            return DatasetFormat.JSONL
    
    # Check for JSONL files directly in root directory
    jsonl_files = list(dataset_path.glob("*.jsonl"))
    if jsonl_files:
        return DatasetFormat.JSONL
    
    # Check for Parquet files (including in subdirectories)
    parquet_files = list(dataset_path.rglob("*.parquet"))
    if parquet_files:
        return DatasetFormat.PARQUET
    
    return DatasetFormat.UNKNOWN


def discover_datasets(datasets_dir: str) -> List[DatasetInfo]:
    """
    Scan the datasets directory and discover all Nemotron datasets.
    
    Args:
        datasets_dir: Path to the datasets directory
        
    Returns:
        List of DatasetInfo objects for discovered datasets
    """
    datasets_path = Path(datasets_dir)
    discovered = []
    
    if not datasets_path.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return discovered
    
    # Look for nvidia_* directories
    for item in sorted(datasets_path.iterdir()):
        if item.is_dir() and item.name.startswith("nvidia_"):
            dataset_name = item.name.replace("nvidia_", "nvidia/")
            format_type = detect_format(item)
            
            if format_type != DatasetFormat.UNKNOWN:
                info = DatasetInfo(
                    name=dataset_name,
                    path=str(item),
                    format=format_type,
                )
                discovered.append(info)
                logger.info(f"Discovered: {dataset_name} ({format_type.value})")
            else:
                logger.warning(f"Unknown format for: {dataset_name}")
    
    return discovered


# =============================================================================
# Statistics Collection
# =============================================================================

def get_arrow_statistics(dataset_path: Path) -> Tuple[Dict[str, SplitInfo], List[str], int]:
    """
    Get statistics for an Arrow (HuggingFace) format dataset.
    
    Uses PyArrow directly to read arrow files for better compatibility.
    Falls back to datasets library if needed.
    
    Returns:
        Tuple of (splits_dict, columns_list, total_rows)
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    
    splits = {}
    columns = []
    total_rows = 0
    
    # Find all split directories
    for subdir in sorted(dataset_path.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            arrow_files = sorted(subdir.glob("*.arrow"))
            if arrow_files:
                try:
                    # Read arrow files directly with PyArrow for row count
                    split_rows = 0
                    for arrow_file in arrow_files:
                        try:
                            # Open as IPC stream reader
                            with pa.memory_map(str(arrow_file), 'r') as source:
                                reader = ipc.open_stream(source)
                                table = reader.read_all()
                                split_rows += table.num_rows
                                
                                # Get columns from first file
                                if not columns:
                                    columns = table.column_names
                        except Exception:
                            # Try as IPC file format instead of stream
                            try:
                                with pa.memory_map(str(arrow_file), 'r') as source:
                                    reader = ipc.open_file(source)
                                    table = reader.read_all()
                                    split_rows += table.num_rows
                                    if not columns:
                                        columns = table.column_names
                            except Exception as inner_e:
                                logger.debug(f"PyArrow error on {arrow_file.name}: {inner_e}")
                                continue
                    
                    if split_rows > 0:
                        splits[subdir.name] = SplitInfo(
                            name=subdir.name,
                            rows=split_rows,
                            files=[f.name for f in arrow_files]
                        )
                        total_rows += split_rows
                        
                except Exception as e:
                    logger.warning(f"Error loading split {subdir.name}: {e}")
    
    # Fallback: try using datasets library if PyArrow didn't work
    if not splits:
        try:
            from datasets import load_from_disk
            
            # Try loading as DatasetDict from parent
            try:
                dataset_dict = load_from_disk(str(dataset_path))
                if hasattr(dataset_dict, 'keys'):
                    for split_name in dataset_dict.keys():
                        split_data = dataset_dict[split_name]
                        num_rows = len(split_data)
                        splits[split_name] = SplitInfo(
                            name=split_name,
                            rows=num_rows,
                            files=[]
                        )
                        total_rows += num_rows
                        if not columns:
                            columns = split_data.column_names
            except Exception:
                pass
        except ImportError:
            pass
    
    return splits, columns, total_rows


def get_jsonl_statistics(dataset_path: Path) -> Tuple[Dict[str, SplitInfo], List[str], int]:
    """
    Get statistics for a JSONL format dataset.
    
    Checks both data/ subdirectory and root directory for JSONL files.
    Uses fast subprocess line counting for large files.
    
    Returns:
        Tuple of (splits_dict, columns_list, total_rows)
    """
    import subprocess
    
    splits = {}
    columns = []
    total_rows = 0
    
    # Collect JSONL files from data/ subdirectory and root directory
    jsonl_files = []
    
    data_dir = dataset_path / "data"
    if data_dir.exists():
        jsonl_files.extend(sorted(data_dir.glob("*.jsonl")))
    
    # Also check root directory for JSONL files
    root_jsonl_files = sorted(dataset_path.glob("*.jsonl"))
    jsonl_files.extend(root_jsonl_files)
    
    if not jsonl_files:
        return splits, columns, total_rows
    
    for jsonl_file in jsonl_files:
        # Get columns from first line only
        if not columns:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                try:
                    data = json.loads(first_line)
                    columns = list(data.keys())
                except json.JSONDecodeError:
                    pass
        
        # Fast line counting using wc -l
        try:
            result = subprocess.run(
                ['wc', '-l', str(jsonl_file)],
                capture_output=True,
                text=True,
                timeout=60
            )
            row_count = int(result.stdout.split()[0])
        except Exception:
            # Fallback to Python counting if wc fails
            row_count = 0
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for _ in f:
                    row_count += 1
        
        # Use filename (without extension) as split name
        split_name = jsonl_file.stem
        if split_name not in splits:
            splits[split_name] = SplitInfo(
                name=split_name,
                rows=row_count,
                files=[jsonl_file.name]
            )
        else:
            splits[split_name].rows += row_count
            splits[split_name].files.append(jsonl_file.name)
        
        total_rows += row_count
    
    return splits, columns, total_rows


def get_parquet_statistics(dataset_path: Path) -> Tuple[Dict[str, SplitInfo], List[str], int]:
    """
    Get statistics for a Parquet format dataset.
    
    Returns:
        Tuple of (splits_dict, columns_list, total_rows)
    """
    import pyarrow.parquet as pq
    
    splits = {}
    columns = []
    total_rows = 0
    
    # Find all parquet files (including in subdirectories)
    parquet_files = sorted(dataset_path.rglob("*.parquet"))
    
    for pq_file in parquet_files:
        try:
            # Get schema and row count from metadata
            pq_metadata = pq.read_metadata(pq_file)
            row_count = pq_metadata.num_rows
            
            # Get columns from schema
            if not columns:
                schema = pq.read_schema(pq_file)
                columns = schema.names
            
            # Use parent directory name as split name (or 'default')
            split_name = pq_file.parent.name if pq_file.parent != dataset_path else "default"
            
            if split_name not in splits:
                splits[split_name] = SplitInfo(
                    name=split_name,
                    rows=row_count,
                    files=[str(pq_file.relative_to(dataset_path))]
                )
            else:
                splits[split_name].rows += row_count
                splits[split_name].files.append(str(pq_file.relative_to(dataset_path)))
            
            total_rows += row_count
            
        except Exception as e:
            logger.warning(f"Error reading parquet {pq_file}: {e}")
    
    return splits, columns, total_rows


def identify_text_columns(columns: List[str], dataset_name: str) -> Tuple[List[str], str]:
    """
    Identify text columns and embedding strategy based on dataset name.
    
    Explicit mapping for each dataset for easy inspection and modification.
    
    Returns:
        Tuple of (text_columns, embedding_strategy)
    
    Dataset Reference:
    -----------------
    | Dataset                  | Key Text Columns                                    | Strategy            |
    |--------------------------|-----------------------------------------------------|---------------------|
    | pretrain-sample          | text                                                | direct_text         |
    | llama-sft                | input + output                                      | input_output        |
    | v1                       | messages                                            | concatenate_messages|
    | v2                       | messages                                            | concatenate_messages|
    | v3-instruction-chat      | messages                                            | concatenate_messages|
    | v3-agentic               | messages                                            | concatenate_messages|
    | v3-science               | messages                                            | concatenate_messages|
    | v3-math-proofs           | problem + formal_statement + lean_header + messages | math_proofs         |
    | v3-math                  | messages                                            | concatenate_messages|
    | v3-rl-blend              | responses_create_params                             | combine_columns     |
    | v3-competitive-prog      | messages                                            | concatenate_messages|
    | v3-swe                   | messages                                            | concatenate_messages|
    """
    dataset_lower = dataset_name.lower()
    
    # ==========================================================================
    # EXPLICIT DATASET MAPPINGS
    # ==========================================================================
    
    # -------------------------------------------------------------------------
    # pretrain-sample: nvidia/Nemotron-Pretraining-Dataset-sample
    # Columns: id, text
    # -------------------------------------------------------------------------
    if 'pretraining' in dataset_lower or 'pretrain' in dataset_lower:
        if 'text' in columns:
            return ['text'], 'direct_text'
    
    # -------------------------------------------------------------------------
    # llama-sft: nvidia/Llama-Nemotron-Post-Training-Dataset
    # Columns: input, output, category, license, reasoning, generator, 
    #          used_in_training, version, system_prompt
    # -------------------------------------------------------------------------
    elif 'llama-nemotron' in dataset_lower or 'llama_nemotron' in dataset_lower:
        if 'input' in columns and 'output' in columns:
            return ['input', 'output'], 'input_output'
    
    # -------------------------------------------------------------------------
    # v3-math-proofs: nvidia/Nemotron-Math-Proofs-v1
    # Columns: problem, source, formal_statement, lean_header, url, user_name,
    #          user_url, sft_line_number, messages, uuid, used_in, tools, license
    # -------------------------------------------------------------------------
    elif 'math-proofs' in dataset_lower or 'math_proofs' in dataset_lower:
        text_cols = []
        if 'problem' in columns:
            text_cols.append('problem')
        if 'formal_statement' in columns:
            text_cols.append('formal_statement')
        if 'lean_header' in columns:
            text_cols.append('lean_header')
        if 'messages' in columns:
            text_cols.append('messages')
        if text_cols:
            return text_cols, 'math_proofs'
    
    # -------------------------------------------------------------------------
    # v3-rl-blend: nvidia/Nemotron-3-Nano-RL-Training-Blend
    # Columns: id, responses_create_params, ground_truth, category, 
    #          environment_name, agent_ref, pass_rate, pass_rate_total, 
    #          pass_rate_passed, dataset
    # -------------------------------------------------------------------------
    elif 'rl-training-blend' in dataset_lower or 'rl_training_blend' in dataset_lower or 'nano-rl' in dataset_lower:
        if 'responses_create_params' in columns:
            return ['responses_create_params'], 'combine_columns'
    
    # -------------------------------------------------------------------------
    # v3-math: nvidia/Nemotron-Math-v2
    # Columns: expected_answer, problem, original_expected_answer, 
    #          changed_answer_to_majority, data_source, messages, tools, 
    #          used_in, metadata, license, uuid, url, user_url, user_name
    # Note: messages contains the full problem + solution, so we use messages
    # -------------------------------------------------------------------------
    elif 'nemotron-math' in dataset_lower or 'nemotron_math' in dataset_lower:
        if 'messages' in columns:
            return ['messages'], 'concatenate_messages'
    
    # -------------------------------------------------------------------------
    # v3-competitive-programming: nvidia/Nemotron-Competitive-Programming-v1
    # Columns: uuid, messages, license, used_in, tools, dataset, split, 
    #          index, source, difficulty, question_id
    # -------------------------------------------------------------------------
    elif 'competitive-programming' in dataset_lower or 'competitive_programming' in dataset_lower:
        if 'messages' in columns:
            return ['messages'], 'concatenate_messages'
    
    # -------------------------------------------------------------------------
    # v3-instruction-chat: nvidia/Nemotron-Instruction-Following-Chat-v1
    # Columns: uuid, messages, license, used_in, tools, reasoning, capability_target
    # -------------------------------------------------------------------------
    elif 'instruction-following' in dataset_lower or 'instruction_following' in dataset_lower:
        if 'messages' in columns:
            return ['messages'], 'concatenate_messages'
    
    # -------------------------------------------------------------------------
    # v3-agentic: nvidia/Nemotron-Agentic-v1
    # Columns: uuid, messages, license, used_in, tools, reasoning
    # -------------------------------------------------------------------------
    elif 'agentic' in dataset_lower:
        if 'messages' in columns:
            return ['messages'], 'concatenate_messages'
    
    # -------------------------------------------------------------------------
    # v3-science: nvidia/Nemotron-Science-v1
    # Columns: uuid, messages, license, used_in, tools
    # -------------------------------------------------------------------------
    elif 'science' in dataset_lower:
        if 'messages' in columns:
            return ['messages'], 'concatenate_messages'
    
    # -------------------------------------------------------------------------
    # v3-swe: nvidia/Nemotron-SWE-v1
    # Columns: uuid, messages, license, used_in, tools, dataset, repo
    # -------------------------------------------------------------------------
    elif 'swe' in dataset_lower:
        if 'messages' in columns:
            return ['messages'], 'concatenate_messages'
    
    # -------------------------------------------------------------------------
    # v1: nvidia/Nemotron-Post-Training-Dataset-v1
    # Columns: uuid, license, generator, version, category, reasoning, 
    #          messages, metadata
    # -------------------------------------------------------------------------
    elif 'post-training-dataset-v1' in dataset_lower or 'post_training_dataset_v1' in dataset_lower:
        if 'messages' in columns:
            return ['messages'], 'concatenate_messages'
    
    # -------------------------------------------------------------------------
    # v2: nvidia/Nemotron-Post-Training-Dataset-v2
    # Columns: uuid, license, generator, version, category, reasoning, messages
    # -------------------------------------------------------------------------
    elif 'post-training-dataset-v2' in dataset_lower or 'post_training_dataset_v2' in dataset_lower:
        if 'messages' in columns:
            return ['messages'], 'concatenate_messages'
    
    # ==========================================================================
    # FALLBACK: Generic detection based on column names
    # ==========================================================================
    
    # Input/output format (generic)
    if 'input' in columns and 'output' in columns:
        return ['input', 'output'], 'input_output'
    
    # Standard messages format (most common)
    if 'messages' in columns:
        return ['messages'], 'concatenate_messages'
    
    # Direct text column (pretraining datasets)
    if 'text' in columns:
        return ['text'], 'direct_text'
    
    # Math proofs format (generic)
    if 'problem' in columns and 'formal_statement' in columns:
        text_cols = ['problem', 'formal_statement']
        if 'lean_header' in columns:
            text_cols.append('lean_header')
        if 'messages' in columns:
            text_cols.append('messages')
        return text_cols, 'math_proofs'
    
    # Fallback: look for common text column names
    text_indicators = ['text', 'content', 'document', 'passage', 'prompt', 'response']
    text_cols = [c for c in columns if any(ind in c.lower() for ind in text_indicators)]
    
    if text_cols:
        return text_cols, 'combine_columns'
    
    return [], 'unknown'


def collect_statistics(dataset_info: DatasetInfo) -> DatasetInfo:
    """
    Collect detailed statistics for a dataset.
    
    Args:
        dataset_info: Basic dataset info with path and format
        
    Returns:
        Updated DatasetInfo with statistics
    """
    dataset_path = Path(dataset_info.path)
    
    if dataset_info.format == DatasetFormat.ARROW:
        splits, columns, total_rows = get_arrow_statistics(dataset_path)
    elif dataset_info.format == DatasetFormat.JSONL:
        splits, columns, total_rows = get_jsonl_statistics(dataset_path)
    elif dataset_info.format == DatasetFormat.PARQUET:
        splits, columns, total_rows = get_parquet_statistics(dataset_path)
    else:
        return dataset_info
    
    dataset_info.splits = splits
    dataset_info.columns = columns
    dataset_info.total_rows = total_rows
    
    # Identify text columns and strategy
    text_cols, strategy = identify_text_columns(columns, dataset_info.name)
    dataset_info.text_columns = text_cols
    dataset_info.embedding_strategy = strategy
    
    return dataset_info


# =============================================================================
# Data Iterators
# =============================================================================

def iter_arrow_dataset(
    dataset_path: Path,
    split_name: str,
    chunk_size: int = 100
) -> Iterator[List[Dict[str, Any]]]:
    """
    Iterate over an Arrow dataset split in chunks.
    
    Uses PyArrow directly for better compatibility across versions.
    Falls back to datasets library if PyArrow fails.
    
    Args:
        dataset_path: Path to the dataset
        split_name: Name of the split to iterate
        chunk_size: Number of examples per chunk
        
    Yields:
        List of examples (dicts)
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    
    split_path = dataset_path / split_name
    arrow_files = sorted(split_path.glob("*.arrow"))
    
    if not arrow_files:
        logger.warning(f"No arrow files found in {split_path}")
        return
    
    # Try PyArrow direct reading first
    pyarrow_success = False
    
    for arrow_file in arrow_files:
        try:
            # Try as IPC stream
            with pa.memory_map(str(arrow_file), 'r') as source:
                try:
                    reader = ipc.open_stream(source)
                except Exception:
                    # Try as IPC file
                    reader = ipc.open_file(source)
                
                table = reader.read_all()
                pyarrow_success = True
                
                # Convert to batches of dicts
                num_rows = table.num_rows
                for start_idx in range(0, num_rows, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_rows)
                    batch_table = table.slice(start_idx, end_idx - start_idx)
                    # Convert to list of dicts
                    batch = batch_table.to_pydict()
                    # Transform from column-oriented to row-oriented
                    num_batch_rows = end_idx - start_idx
                    batch_list = [
                        {col: batch[col][i] for col in batch.keys()}
                        for i in range(num_batch_rows)
                    ]
                    yield batch_list
                    
        except Exception as e:
            logger.debug(f"PyArrow error on {arrow_file}: {e}")
            continue
    
    # Fallback to datasets library if PyArrow didn't work
    if not pyarrow_success:
        try:
            from datasets import load_from_disk
            
            dataset = load_from_disk(str(split_path))
            
            for i in range(0, len(dataset), chunk_size):
                end_idx = min(i + chunk_size, len(dataset))
                # Convert to list of dicts
                batch = [dataset[j] for j in range(i, end_idx)]
                yield batch
        except Exception as e:
            logger.error(f"Failed to load {split_path} with both PyArrow and datasets: {e}")


def iter_jsonl_file(
    filepath: Path,
    chunk_size: int = 100
) -> Iterator[List[Dict[str, Any]]]:
    """
    Iterate over a JSONL file in chunks.
    
    Args:
        filepath: Path to the JSONL file
        chunk_size: Number of lines per chunk
        
    Yields:
        List of parsed JSON objects
    """
    batch = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                batch.append(data)
                
                if len(batch) >= chunk_size:
                    yield batch
                    batch = []
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                continue
    
    if batch:
        yield batch


def iter_parquet_file(
    filepath: Path,
    chunk_size: int = 100
) -> Iterator[List[Dict[str, Any]]]:
    """
    Iterate over a Parquet file in chunks.
    
    Args:
        filepath: Path to the Parquet file
        chunk_size: Number of rows per chunk
        
    Yields:
        List of rows as dicts
    """
    import pyarrow.parquet as pq
    
    parquet_file = pq.ParquetFile(filepath)
    
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        # Convert to pandas then to list of dicts
        df = batch.to_pandas()
        yield df.to_dict('records')


# =============================================================================
# Text Extraction
# =============================================================================

def extract_text_messages(example: Dict[str, Any]) -> str:
    """Extract text from messages format (concatenate role: content)."""
    messages = example.get('messages', [])
    if not messages:
        return ""
    
    text_parts = []
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        # Include reasoning_content if available
        if msg.get('reasoning_content'):
            content = msg['reasoning_content'] + '\n' + content
        
        text_parts.append(f'{role}: {content}')
    
    return '\n'.join(text_parts)


def extract_text_input_output(example: Dict[str, Any]) -> str:
    """Extract text from input/output format (Llama-Nemotron)."""
    parts = []
    
    # Handle input (can be list of messages or string)
    input_val = example.get('input', '')
    if isinstance(input_val, list):
        for msg in input_val:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                parts.append(f'{role}: {content}')
            else:
                parts.append(str(msg))
    else:
        parts.append(f'input: {input_val}')
    
    # Handle output
    output_val = example.get('output', '')
    if output_val:
        parts.append(f'assistant: {output_val}')
    
    return '\n'.join(parts)


def extract_text_direct(example: Dict[str, Any], column: str = 'text') -> str:
    """Extract text directly from a column."""
    return str(example.get(column, ''))


def extract_text_combine_columns(example: Dict[str, Any], columns: List[str]) -> str:
    """Combine multiple text columns."""
    parts = []
    for col in columns:
        val = example.get(col)
        if val:
            if isinstance(val, list):
                parts.append(f'{col}: {str(val)}')
            else:
                parts.append(f'{col}: {val}')
    return '\n'.join(parts)


def extract_text_math_proofs(example: Dict[str, Any]) -> str:
    """
    Extract text from math_proofs format.
    
    Combines:
    - problem: The math problem statement
    - formal_statement: The formal Lean statement
    - lean_header: Lean imports/setup
    - messages: Conversation if available (some rows have messages with proofs)
    """
    parts = []
    
    # Primary structured columns
    if example.get('problem'):
        parts.append(f"Problem: {example['problem']}")
    
    if example.get('formal_statement'):
        parts.append(f"Formal Statement:\n{example['formal_statement']}")
    
    if example.get('lean_header'):
        parts.append(f"Lean Header:\n{example['lean_header']}")
    
    # Also include messages if present (some rows have proof conversations)
    messages = example.get('messages')
    if messages and isinstance(messages, list) and len(messages) > 0:
        msg_parts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if msg.get('reasoning_content'):
                    content = msg['reasoning_content'] + '\n' + content
                if content:  # Only add if there's actual content
                    msg_parts.append(f'{role}: {content}')
        if msg_parts:
            parts.append("Messages:\n" + '\n'.join(msg_parts))
    
    return '\n\n'.join(parts)


def get_text_extractor(strategy: str, text_columns: List[str]):
    """
    Get the appropriate text extraction function.
    
    Args:
        strategy: Embedding strategy name
        text_columns: List of text column names
        
    Returns:
        Function that takes an example dict and returns text string
    """
    if strategy == 'concatenate_messages':
        return extract_text_messages
    elif strategy == 'input_output':
        return extract_text_input_output
    elif strategy == 'direct_text':
        col = text_columns[0] if text_columns else 'text'
        return lambda ex: extract_text_direct(ex, col)
    elif strategy == 'combine_columns':
        return lambda ex: extract_text_combine_columns(ex, text_columns)
    elif strategy == 'math_proofs':
        return extract_text_math_proofs
    else:
        # Default: try messages, then text, then combine all text columns
        def default_extractor(ex):
            if 'messages' in ex:
                return extract_text_messages(ex)
            if 'text' in ex:
                return extract_text_direct(ex, 'text')
            return extract_text_combine_columns(ex, text_columns)
        return default_extractor


# =============================================================================
# Model Loading and Embedding Extraction
# =============================================================================

def load_model_and_tokenizer(
    model_id: str = MODEL_ID_DEFAULT,
    max_length: int = DEFAULT_MAX_LENGTH,
    dtype: str = DEFAULT_DTYPE,
    device: str = "cuda"
):
    """
    Load the embedding model and tokenizer.
    
    Args:
        model_id: HuggingFace model ID
        max_length: Maximum sequence length
        dtype: Model dtype (bfloat16 or float16)
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    
    logger.info(f"Loading model: {model_id}")
    logger.info(f"Max length: {max_length}, dtype: {dtype}")
    
    # Determine torch dtype
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=max_length,
        trust_remote_code=True
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine attention implementation
    # llama-embed-nemotron-8b only supports flash_attention_2 or eager
    attn_implementation = "eager"  # Default to eager (works on all GPUs)
    
    # Try to use flash_attention_2 if available (faster, requires compatible GPU)
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
        logger.info("Using flash_attention_2 for faster inference")
    except ImportError:
        logger.info("flash_attn not available, using eager attention")
    
    # Load model
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device,
        attn_implementation=attn_implementation
    )
    model.eval()
    
    logger.info(f"Model loaded on {device}")
    return model, tokenizer


def extract_embeddings(
    texts: List[str],
    model,
    tokenizer,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """
    Extract embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model: Loaded model
        tokenizer: Loaded tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for inference
        
    Returns:
        Numpy array of embeddings (float32)
    """
    import torch
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Use last hidden state with mean pooling over non-padding tokens
            # For embedding models, [CLS] token or mean pooling is common
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        
        # Convert to numpy float32
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        all_embeddings.append(embeddings_np)
        
        # Clear CUDA cache periodically
        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
    
    return np.vstack(all_embeddings) if all_embeddings else np.array([])


# =============================================================================
# Shard Saving
# =============================================================================

def save_shard(
    embeddings: np.ndarray,
    output_dir: Path,
    shard_idx: int
) -> str:
    """
    Save embeddings shard as numpy file.
    
    Args:
        embeddings: Numpy array of embeddings
        output_dir: Output directory
        shard_idx: Shard index
        
    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"shard_{shard_idx:05d}.npy"
    filepath = output_dir / filename
    np.save(filepath, embeddings)
    return str(filepath)


def save_statistics(
    statistics: Dict[str, Any],
    output_path: Path
):
    """Save statistics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    logger.info(f"Statistics saved to: {output_path}")


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_arrow_dataset(
    dataset_info: DatasetInfo,
    output_dir: Path,
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
    chunk_size: int,
    shard_size: int,
    gpu_id: int = 0,
    num_gpus: int = 1
) -> Dict[str, int]:
    """
    Process an Arrow format dataset.
    - If #files >= #GPUs: each file becomes one shard (file index = shard index)
    - If #files < #GPUs: distribute chunks within files across GPUs (shard_size based)
    
    Returns:
        Dictionary of split_name -> embedding_count
    """
    import torch
    import pyarrow as pa
    import pyarrow.ipc as ipc
    
    dataset_path = Path(dataset_info.path)
    results = {}
    
    text_extractor = get_text_extractor(
        dataset_info.embedding_strategy,
        dataset_info.text_columns
    )
    
    for split_name, split_info in dataset_info.splits.items():
        split_path = dataset_path / split_name
        arrow_files = sorted(split_path.glob("*.arrow"))
        
        split_output_dir = output_dir / split_name
        total_embeddings = 0
        
        # Choose distribution strategy based on file count vs GPU count
        if len(arrow_files) >= num_gpus:
            # Strategy 1: Distribute files across GPUs (1 shard per file)
            my_files = [(idx, f) for idx, f in enumerate(arrow_files) if idx % num_gpus == gpu_id]
            logger.info(f"Processing split: {split_name} ({len(arrow_files)} files, {len(my_files)} for GPU {gpu_id}) [file-parallel]")
            
            for file_idx, arrow_file in tqdm(my_files, desc=f"  {split_name}"):
                try:
                    with pa.memory_map(str(arrow_file), 'r') as source:
                        try:
                            reader = ipc.open_stream(source)
                        except Exception:
                            reader = ipc.open_file(source)
                        table = reader.read_all()
                    
                    file_embeddings = []
                    num_rows = table.num_rows
                    
                    for start_idx in range(0, num_rows, chunk_size):
                        end_idx = min(start_idx + chunk_size, num_rows)
                        batch_dict = {col: table.column(col).slice(start_idx, end_idx - start_idx).to_pylist() 
                                      for col in table.column_names}
                        batch = [{col: batch_dict[col][i] for col in table.column_names}
                                 for i in range(end_idx - start_idx)]
                        
                        texts = [text_extractor(ex) for ex in batch]
                        texts = [t for t in texts if t.strip()]
                        
                        if texts:
                            embeddings = extract_embeddings(texts, model, tokenizer, max_length, batch_size)
                            file_embeddings.append(embeddings)
                    
                    if file_embeddings:
                        combined = np.vstack(file_embeddings)
                        save_shard(combined, split_output_dir, file_idx)
                        total_embeddings += combined.shape[0]
                    
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error processing {arrow_file}: {e}")
        else:
            # Strategy 2: All GPUs process all files, distribute chunks (shard_size based)
            logger.info(f"Processing split: {split_name} ({len(arrow_files)} files < {num_gpus} GPUs) [chunk-parallel]")
            
            shard_idx = gpu_id
            accumulated_embeddings = []
            chunk_idx = 0
            
            for arrow_file in arrow_files:
                try:
                    with pa.memory_map(str(arrow_file), 'r') as source:
                        try:
                            reader = ipc.open_stream(source)
                        except Exception:
                            reader = ipc.open_file(source)
                        table = reader.read_all()
                    
                    num_rows = table.num_rows
                    pbar = tqdm(total=num_rows, desc=f"  {arrow_file.name}")
                    
                    for start_idx in range(0, num_rows, chunk_size):
                        # Multi-GPU: only process chunks assigned to this GPU
                        if chunk_idx % num_gpus != gpu_id:
                            chunk_idx += 1
                            pbar.update(min(chunk_size, num_rows - start_idx))
                            continue
                        chunk_idx += 1
                        
                        end_idx = min(start_idx + chunk_size, num_rows)
                        batch_dict = {col: table.column(col).slice(start_idx, end_idx - start_idx).to_pylist() 
                                      for col in table.column_names}
                        batch = [{col: batch_dict[col][i] for col in table.column_names}
                                 for i in range(end_idx - start_idx)]
                        
                        texts = [text_extractor(ex) for ex in batch]
                        texts = [t for t in texts if t.strip()]
                        
                        if texts:
                            embeddings = extract_embeddings(texts, model, tokenizer, max_length, batch_size)
                            accumulated_embeddings.append(embeddings)
                            total_embeddings += len(embeddings)
                            
                            # Save shard if accumulated enough
                            total_accumulated = sum(e.shape[0] for e in accumulated_embeddings)
                            if total_accumulated >= shard_size:
                                combined = np.vstack(accumulated_embeddings)
                                save_shard(combined, split_output_dir, shard_idx)
                                shard_idx += num_gpus
                                accumulated_embeddings = []
                                torch.cuda.empty_cache()
                        
                        pbar.update(end_idx - start_idx)
                    
                    pbar.close()
                except Exception as e:
                    logger.error(f"Error processing {arrow_file}: {e}")
            
            # Save remaining embeddings
            if accumulated_embeddings:
                combined = np.vstack(accumulated_embeddings)
                save_shard(combined, split_output_dir, shard_idx)
        
        results[split_name] = total_embeddings
        logger.info(f"  {split_name}: {total_embeddings:,} embeddings saved (GPU {gpu_id})")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return results


def process_jsonl_dataset(
    dataset_info: DatasetInfo,
    output_dir: Path,
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
    chunk_size: int,
    shard_size: int,
    gpu_id: int = 0,
    num_gpus: int = 1
) -> Dict[str, int]:
    """
    Process a JSONL format dataset.
    
    Returns:
        Dictionary of split_name -> embedding_count
    """
    import torch
    
    dataset_path = Path(dataset_info.path)
    data_dir = dataset_path / "data"
    results = {}
    
    text_extractor = get_text_extractor(
        dataset_info.embedding_strategy,
        dataset_info.text_columns
    )
    
    for split_name, split_info in dataset_info.splits.items():
        gpu_rows = split_info.rows // num_gpus + (1 if gpu_id < split_info.rows % num_gpus else 0)
        logger.info(f"Processing split: {split_name} ({split_info.rows:,} total rows, ~{gpu_rows:,} for GPU {gpu_id})")
        
        # All GPUs write to same split directory with interleaved shard numbers
        split_output_dir = output_dir / split_name
        # Each GPU starts at its gpu_id and increments by num_gpus to avoid conflicts
        shard_idx = gpu_id
        total_embeddings = 0
        accumulated_embeddings = []
        
        # Process each JSONL file for this split
        chunk_idx = 0
        for filename in split_info.files:
            # Check data/ subdirectory first, then root directory
            filepath = data_dir / filename
            if not filepath.exists():
                filepath = dataset_path / filename
            
            pbar = tqdm(desc=f"  {filename}")
            
            for batch in iter_jsonl_file(filepath, chunk_size):
                # Multi-GPU: only process chunks assigned to this GPU
                if num_gpus > 1 and chunk_idx % num_gpus != gpu_id:
                    chunk_idx += 1
                    continue
                chunk_idx += 1
                
                # Extract texts
                texts = [text_extractor(ex) for ex in batch]
                texts = [t for t in texts if t.strip()]
                
                if not texts:
                    pbar.update(len(batch))
                    continue
                
                # Extract embeddings
                embeddings = extract_embeddings(texts, model, tokenizer, max_length, batch_size)
                accumulated_embeddings.append(embeddings)
                total_embeddings += len(embeddings)
                
                # Save shard if accumulated enough
                total_accumulated = sum(e.shape[0] for e in accumulated_embeddings)
                if total_accumulated >= shard_size:
                    combined = np.vstack(accumulated_embeddings)
                    save_shard(combined, split_output_dir, shard_idx)
                    shard_idx += num_gpus  # Increment by num_gpus to avoid conflicts
                    accumulated_embeddings = []
                    torch.cuda.empty_cache()
                
                pbar.update(len(batch))
            
            pbar.close()
        
        # Save remaining embeddings
        if accumulated_embeddings:
            combined = np.vstack(accumulated_embeddings)
            save_shard(combined, split_output_dir, shard_idx)
        
        results[split_name] = total_embeddings
        logger.info(f"  {split_name}: {total_embeddings:,} embeddings saved (GPU {gpu_id})")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return results


def process_parquet_dataset(
    dataset_info: DatasetInfo,
    output_dir: Path,
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
    chunk_size: int,
    shard_size: int,
    gpu_id: int = 0,
    num_gpus: int = 1
) -> Dict[str, int]:
    """
    Process a Parquet format dataset.
    - If #files >= #GPUs: each file becomes one shard (file index = shard index)
    - If #files < #GPUs: distribute chunks within files across GPUs (shard_size based)
    
    Returns:
        Dictionary of split_name -> embedding_count
    """
    import torch
    import pyarrow.parquet as pq
    
    dataset_path = Path(dataset_info.path)
    results = {}
    
    text_extractor = get_text_extractor(
        dataset_info.embedding_strategy,
        dataset_info.text_columns
    )
    
    for split_name, split_info in dataset_info.splits.items():
        parquet_files = [(idx, dataset_path / f) for idx, f in enumerate(split_info.files)]
        
        split_output_dir = output_dir / split_name
        total_embeddings = 0
        
        # Choose distribution strategy based on file count vs GPU count
        if len(parquet_files) >= num_gpus:
            # Strategy 1: Distribute files across GPUs (1 shard per file)
            my_files = [(idx, f) for idx, f in parquet_files if idx % num_gpus == gpu_id]
            logger.info(f"Processing split: {split_name} ({len(parquet_files)} files, {len(my_files)} for GPU {gpu_id}) [file-parallel]")
            
            for file_idx, parquet_file in tqdm(my_files, desc=f"  {split_name}"):
                try:
                    table = pq.read_table(parquet_file)
                    file_embeddings = []
                    num_rows = table.num_rows
                    
                    for start_idx in range(0, num_rows, chunk_size):
                        end_idx = min(start_idx + chunk_size, num_rows)
                        batch_table = table.slice(start_idx, end_idx - start_idx)
                        batch = batch_table.to_pandas().to_dict('records')
                        
                        texts = [text_extractor(ex) for ex in batch]
                        texts = [t for t in texts if t.strip()]
                        
                        if texts:
                            embeddings = extract_embeddings(texts, model, tokenizer, max_length, batch_size)
                            file_embeddings.append(embeddings)
                    
                    if file_embeddings:
                        combined = np.vstack(file_embeddings)
                        save_shard(combined, split_output_dir, file_idx)
                        total_embeddings += combined.shape[0]
                    
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error processing {parquet_file}: {e}")
        else:
            # Strategy 2: All GPUs process all files, distribute chunks (shard_size based)
            logger.info(f"Processing split: {split_name} ({len(parquet_files)} files < {num_gpus} GPUs) [chunk-parallel]")
            
            shard_idx = gpu_id
            accumulated_embeddings = []
            chunk_idx = 0
            
            for _, parquet_file in parquet_files:
                try:
                    table = pq.read_table(parquet_file)
                    num_rows = table.num_rows
                    pbar = tqdm(total=num_rows, desc=f"  {parquet_file.name}")
                    
                    for start_idx in range(0, num_rows, chunk_size):
                        # Multi-GPU: only process chunks assigned to this GPU
                        if chunk_idx % num_gpus != gpu_id:
                            chunk_idx += 1
                            pbar.update(min(chunk_size, num_rows - start_idx))
                            continue
                        chunk_idx += 1
                        
                        end_idx = min(start_idx + chunk_size, num_rows)
                        batch_table = table.slice(start_idx, end_idx - start_idx)
                        batch = batch_table.to_pandas().to_dict('records')
                        
                        texts = [text_extractor(ex) for ex in batch]
                        texts = [t for t in texts if t.strip()]
                        
                        if texts:
                            embeddings = extract_embeddings(texts, model, tokenizer, max_length, batch_size)
                            accumulated_embeddings.append(embeddings)
                            total_embeddings += len(embeddings)
                            
                            # Save shard if accumulated enough
                            total_accumulated = sum(e.shape[0] for e in accumulated_embeddings)
                            if total_accumulated >= shard_size:
                                combined = np.vstack(accumulated_embeddings)
                                save_shard(combined, split_output_dir, shard_idx)
                                shard_idx += num_gpus
                                accumulated_embeddings = []
                                torch.cuda.empty_cache()
                        
                        pbar.update(end_idx - start_idx)
                    
                    pbar.close()
                except Exception as e:
                    logger.error(f"Error processing {parquet_file}: {e}")
            
            # Save remaining embeddings
            if accumulated_embeddings:
                combined = np.vstack(accumulated_embeddings)
                save_shard(combined, split_output_dir, shard_idx)
        
        results[split_name] = total_embeddings
        logger.info(f"  {split_name}: {total_embeddings:,} embeddings saved (GPU {gpu_id})")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return results


def process_dataset(
    dataset_info: DatasetInfo,
    output_base_dir: Path,
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
    chunk_size: int,
    shard_size: int,
    gpu_id: int = 0,
    num_gpus: int = 1
) -> Dict[str, int]:
    """
    Process a single dataset and extract embeddings.
    
    Args:
        dataset_info: Dataset information
        output_base_dir: Base output directory
        model: Loaded model
        tokenizer: Loaded tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for inference
        chunk_size: Chunk size for iteration
        shard_size: Number of embeddings per shard
        gpu_id: GPU ID for data parallelism (0 to num_gpus-1)
        num_gpus: Total number of GPUs for data parallelism
        
    Returns:
        Dictionary of split_name -> embedding_count
    """
    # Create output directory based on dataset name
    dataset_subdir = dataset_info.name.replace("nvidia/", "").replace("-", "_").lower()
    output_dir = output_base_dir / dataset_subdir
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {dataset_info.name}")
    logger.info(f"Format: {dataset_info.format.value}")
    logger.info(f"Total rows: {dataset_info.total_rows:,}")
    logger.info(f"Output: {output_dir}")
    if num_gpus > 1:
        logger.info(f"GPU {gpu_id}/{num_gpus} (processing rows {gpu_id}, {gpu_id + num_gpus}, {gpu_id + 2*num_gpus}, ...)")
    logger.info(f"{'='*60}")
    
    if dataset_info.format == DatasetFormat.ARROW:
        return process_arrow_dataset(
            dataset_info, output_dir, model, tokenizer,
            max_length, batch_size, chunk_size, shard_size,
            gpu_id, num_gpus
        )
    elif dataset_info.format == DatasetFormat.JSONL:
        return process_jsonl_dataset(
            dataset_info, output_dir, model, tokenizer,
            max_length, batch_size, chunk_size, shard_size,
            gpu_id, num_gpus
        )
    elif dataset_info.format == DatasetFormat.PARQUET:
        return process_parquet_dataset(
            dataset_info, output_dir, model, tokenizer,
            max_length, batch_size, chunk_size, shard_size,
            gpu_id, num_gpus
        )
    else:
        logger.warning(f"Unsupported format: {dataset_info.format}")
        return {}


# =============================================================================
# CLI and Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from Nemotron datasets using llama-embed-nemotron-8b"
    )
    
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=DEFAULT_DATASETS_DIR,
        help=f"Path to datasets directory (default: {DEFAULT_DATASETS_DIR})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_EMBEDDINGS_DIR,
        help=f"Path to output embeddings directory (default: {DEFAULT_EMBEDDINGS_DIR})"
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID_DEFAULT,
        help=f"HuggingFace model ID (default: {MODEL_ID_DEFAULT})"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum sequence length (default: {DEFAULT_MAX_LENGTH})"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for inference (default: {DEFAULT_BATCH_SIZE})"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for data iteration (default: {DEFAULT_CHUNK_SIZE})"
    )
    
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of embeddings per shard file (default: 10000)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["bfloat16", "float16"],
        help=f"Model dtype (default: {DEFAULT_DTYPE})"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process only this dataset (by config key, e.g., 'v1', 'llama-sft')"
    )
    
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only collect and save statistics, don't extract embeddings"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run model on (default: cuda)"
    )
    
    # Multi-GPU data parallelism arguments
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID for data parallelism (0 to num-gpus-1)"
    )
    
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Total number of GPUs for data parallelism"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Nemotron Embedding Extraction Pipeline")
    logger.info("="*80)
    logger.info(f"Datasets directory: {args.datasets_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Shard size: {args.shard_size}")
    
    # Step 1: Discover datasets
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Discovering datasets...")
    logger.info("="*60)
    
    discovered_datasets = discover_datasets(args.datasets_dir)
    
    if not discovered_datasets:
        logger.error("No datasets found!")
        return 1
    
    logger.info(f"Found {len(discovered_datasets)} datasets")
    
    # Filter to specific dataset if requested
    if args.dataset:
        if args.dataset in DATASET_CONFIGS:
            target_name = DATASET_CONFIGS[args.dataset]["hf_name"]
            discovered_datasets = [d for d in discovered_datasets if d.name == target_name]
            if not discovered_datasets:
                logger.error(f"Dataset '{args.dataset}' not found in downloaded datasets")
                return 1
        else:
            logger.error(f"Unknown dataset key: {args.dataset}")
            logger.info(f"Available keys: {list(DATASET_CONFIGS.keys())}")
            return 1
    
    # Step 2: Collect statistics
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Collecting statistics...")
    logger.info("="*60)
    
    all_statistics = {}
    
    for dataset_info in discovered_datasets:
        logger.info(f"\nCollecting stats for: {dataset_info.name}")
        dataset_info = collect_statistics(dataset_info)
        all_statistics[dataset_info.name] = dataset_info.to_dict()
        
        logger.info(f"  Format: {dataset_info.format.value}")
        logger.info(f"  Total rows: {dataset_info.total_rows:,}")
        logger.info(f"  Splits: {list(dataset_info.splits.keys())}")
        logger.info(f"  Columns: {dataset_info.columns}")
        logger.info(f"  Text columns: {dataset_info.text_columns}")
        logger.info(f"  Strategy: {dataset_info.embedding_strategy}")
    
    # Save statistics
    output_path = Path(args.output_dir)
    stats_file = output_path / "statistics.json"
    save_statistics(all_statistics, stats_file)
    
    if args.stats_only:
        logger.info("\n" + "="*60)
        logger.info("Statistics collection complete (--stats-only mode)")
        logger.info("="*60)
        return 0
    
    # Step 3: Load model
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Loading model...")
    logger.info("="*60)
    
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        max_length=args.max_length,
        dtype=args.dtype,
        device=args.device
    )
    
    # Step 4: Process datasets
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Extracting embeddings...")
    if args.num_gpus > 1:
        logger.info(f"Multi-GPU mode: GPU {args.gpu_id} of {args.num_gpus}")
    logger.info("="*60)
    
    total_embeddings = 0
    results_summary = {}
    
    for dataset_info in discovered_datasets:
        try:
            results = process_dataset(
                dataset_info=dataset_info,
                output_base_dir=output_path,
                model=model,
                tokenizer=tokenizer,
                max_length=args.max_length,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                shard_size=args.shard_size,
                gpu_id=args.gpu_id,
                num_gpus=args.num_gpus
            )
            
            dataset_total = sum(results.values())
            total_embeddings += dataset_total
            results_summary[dataset_info.name] = {
                "splits": results,
                "total": dataset_total,
                "gpu_id": args.gpu_id
            }
            
        except Exception as e:
            logger.error(f"Error processing {dataset_info.name}: {e}", exc_info=True)
            results_summary[dataset_info.name] = {"error": str(e)}
    
    # Save results summary (include GPU ID in filename for multi-GPU)
    if args.num_gpus > 1:
        results_file = output_path / f"extraction_results_gpu{args.gpu_id}.json"
    else:
        results_file = output_path / "extraction_results.json"
    save_statistics(results_summary, results_file)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info(f"EXTRACTION COMPLETE (GPU {args.gpu_id})")
    logger.info("="*80)
    logger.info(f"Total embeddings extracted: {total_embeddings:,}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Statistics saved to: {stats_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
