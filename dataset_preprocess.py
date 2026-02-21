"""
Shared text extraction, data I/O, and preprocessing helpers for the Nemotron embedding pipeline.

Imported by:
    - 01_explore_nemotron_datasets.py
    - 02_embedding_extraction_huggingface.py
    - 02_embedding_extraction_nemocurator.py
"""

from __future__ import annotations

import glob as glob_module
import json
import os
from typing import Dict, Generator, List, Tuple

import pyarrow.parquet as pq

# Rough tokenization estimate (for progress reporting only)
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 chars per token)."""
    return len(text) // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def flatten_messages(messages: list) -> str:
    """
    Flatten a list of chat messages [{role, content}, ...] into a single string.

    Text concatenation strategy for chat/instruction-tuning data:
    - Preserve role labels to maintain conversational structure
    - Join turns with double newlines for readability
    - Include system messages at the top when present
    """
    if not messages or not isinstance(messages, list):
        return ""
    parts: List[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            if content:
                parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def extract_text_from_record(record: dict, config: dict) -> str:
    """
    Extract and concatenate text from a single record based on the dataset's
    text_strategy configuration.

    Returns the concatenated text suitable for embedding.
    """
    strategy = config.get("text_strategy", {})
    template = strategy.get("template", "messages_list")

    if template == "raw_text":
        return str(record.get("text", ""))

    if template == "messages_list":
        return flatten_messages(record.get("messages", []))

    if template == "messages_concat":
        text_parts: List[str] = []
        input_msgs = record.get("input", [])
        if isinstance(input_msgs, list) and input_msgs:
            text_parts.append(flatten_messages(input_msgs))
        output = record.get("output", "")
        if output:
            text_parts.append(f"Assistant: {output}")
        if not text_parts:
            return flatten_messages(record.get("messages", []))
        return "\n\n".join(text_parts)

    if template == "rl_blend":
        params = record.get("responses_create_params", {})
        input_msgs = params.get("input", []) if isinstance(params, dict) else []
        text = flatten_messages(input_msgs)
        gt = record.get("ground_truth", [])
        if gt:
            text += f"\n\nGround Truth: {json.dumps(gt, ensure_ascii=False)}"
        return text

    if template == "math_proof":
        parts: List[str] = []
        problem = record.get("problem", "")
        if problem:
            parts.append(f"Problem: {problem}")
        lean_header = record.get("lean_header", "")
        formal = record.get("formal_statement", "")
        if lean_header or formal:
            parts.append(f"Formal Statement (Lean 4):\n{lean_header}\n{formal}")
        messages = record.get("messages", [])
        if messages:
            parts.append(flatten_messages(messages))
        return "\n\n".join(parts)

    if template == "math_v2":
        messages = record.get("messages", [])
        if messages:
            return flatten_messages(messages)
        return str(record.get("problem", ""))

    if template == "agentic":
        parts: List[str] = []
        tools = record.get("tools", [])
        if tools and isinstance(tools, list):
            tool_descs: List[str] = []
            for t in tools[:10]:
                if isinstance(t, dict):
                    func = t.get("function", t)
                    name = func.get("name", "unknown")
                    desc = func.get("description", "")
                    if desc and len(desc) > 200:
                        desc = desc[:200] + "..."
                    tool_descs.append(f"Tool: {name} - {desc}")
            if tool_descs:
                parts.append("Available Tools:\n" + "\n".join(tool_descs))
        messages = record.get("messages", [])
        if messages:
            parts.append(flatten_messages(messages))
        return "\n\n".join(parts)

    # Fallback: concatenate all string values
    return " ".join(str(v) for v in record.values() if isinstance(v, str))


# ---------------------------------------------------------------------------
# Data I/O – counting
# ---------------------------------------------------------------------------


def count_jsonl_rows_fast(filepath: str) -> int:
    """Count newlines in a JSONL file (fast byte-level scan)."""
    count = 0
    try:
        with open(filepath, "rb") as fh:
            buf_size = 4 * 1024 * 1024
            while True:
                buf = fh.read(buf_size)
                if not buf:
                    break
                count += buf.count(b"\n")
    except OSError:
        pass
    return count


def estimate_jsonl_rows(filepath: str, sample_bytes: int = 10 * 1024 * 1024) -> int:
    """Estimate row count by sampling the first *sample_bytes* of a JSONL file."""
    try:
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            return 0
        if file_size <= sample_bytes:
            return count_jsonl_rows_fast(filepath)
        with open(filepath, "rb") as fh:
            sample = fh.read(sample_bytes)
        newlines_in_sample = sample.count(b"\n")
        if newlines_in_sample == 0:
            return 1
        return int(newlines_in_sample * (file_size / len(sample)))
    except OSError:
        return 0


def count_parquet_rows(filepath: str) -> int:
    """Read row count from parquet metadata (no data loaded)."""
    try:
        pf = pq.ParquetFile(filepath)
        return pf.metadata.num_rows
    except Exception:
        return 0


def estimate_record_count(filepath: str, file_format: str) -> int:
    """Estimate number of records in a file (fast, for progress reporting only)."""
    if file_format == "parquet":
        return count_parquet_rows(filepath)
    elif file_format == "jsonl":
        return estimate_jsonl_rows(filepath)
    return 0


# ---------------------------------------------------------------------------
# Data I/O – streaming
# ---------------------------------------------------------------------------


def stream_jsonl(filepath: str) -> Generator[dict, None, None]:
    """Yield one parsed JSON record at a time from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def stream_parquet(filepath: str) -> Generator[dict, None, None]:
    """Yield one record dict at a time from a parquet file.

    Uses pyarrow's ``to_pydict()`` instead of pandas to ensure nested
    list/struct columns are returned as native Python types (not numpy arrays).
    """
    pf = pq.ParquetFile(filepath)
    for batch in pf.iter_batches(batch_size=1024):
        cols = batch.to_pydict()
        n_rows = batch.num_rows
        keys = list(cols.keys())
        for i in range(n_rows):
            yield {k: cols[k][i] for k in keys}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def resolve_glob(base_dir: str, pattern: str) -> List[str]:
    """Resolve a glob pattern relative to *base_dir*, returning sorted file paths."""
    return sorted(glob_module.glob(os.path.join(base_dir, pattern)))
