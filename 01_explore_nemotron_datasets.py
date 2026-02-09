#!/usr/bin/env python3
"""
Explore Nemotron datasets and generate a summary document for downstream embedding extraction.

Loads each dataset from DATASET_CONFIGS, inspects structure, designs text
concatenation strategies, and outputs a markdown report to doc/dataset_exploration_summary.md.

USAGE:
    python 01_explore_nemotron_datasets.py

REQUIREMENTS:
    pip install datasets pyarrow pandas
"""

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATASETS_DIR = "/raid/datasets"
DEFAULT_CHUNK_SIZE = 100_000
MODEL_MAX_TOKENS = 32768

# Rough tokenization: ~4 characters per token (GPT-style approximation)
CHARS_PER_TOKEN = 4

# Maximum rows to sample for lightweight inspection
SAMPLE_SIZE = 5
# Maximum rows to scan for text-length statistics
STATS_SAMPLE_SIZE = 1000

# Output directory for the markdown report
DOC_DIR = Path(__file__).parent / "doc"
OUTPUT_FILE = DOC_DIR / "dataset_exploration_summary.md"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DATASET_CONFIGS
# ---------------------------------------------------------------------------
# Each entry maps a short name to its configuration:
#   hf_name   – HuggingFace repo id (used for the local directory name)
#   category  – "post-training" or "pretraining"
#   sub_paths – list of (label, relative_glob_or_path) within the downloaded repo
#               for datasets stored as raw files (JSONL / parquet shards)
#   format    – "jsonl" or "parquet"
#   text_strategy – dict describing how to concatenate fields into embedding text
#       fields      : list of field names or special extractors
#       template    : Python format-string using those field names
#       description : human-readable explanation
#       preprocessing : any preprocessing notes
# ---------------------------------------------------------------------------

DATASET_CONFIGS: Dict[str, dict] = {
    # ===== POST-TRAINING DATASETS =====

    # "Llama-Nemotron-Post-Training-Dataset": {
    #     "hf_name": "nvidia/Llama-Nemotron-Post-Training-Dataset",
    #     "category": "post-training",
    #     "format": "jsonl",
    #     "sub_paths": [
    #         ("SFT/chat", "SFT/chat/chat.jsonl"),
    #         ("SFT/code_v1", "SFT/code/code_v1.jsonl"),
    #         ("SFT/code_v1.1", "SFT/code/code_v1.1.jsonl"),
    #         ("SFT/math_v1", "SFT/math/math_v1.jsonl"),
    #         ("SFT/math_v1.1", "SFT/math/math_v1.1.jsonl"),
    #         ("SFT/safety", "SFT/safety/safety.jsonl"),
    #         ("SFT/science", "SFT/science/science.jsonl"),
    #         ("RL/instruction_following", "RL/instruction_following/instruction_following.jsonl"),
    #         ("train/when2call_sft", "train/when2call_train_sft.jsonl"),
    #         ("train/when2call_pref", "train/when2call_train_pref.jsonl"),
    #     ],
    #     "text_strategy": {
    #         # SFT files: input (list of messages) + output (string)
    #         # Concatenate all message contents with role prefixes, then append output
    #         "fields": ["input", "output"],
    #         "template": "messages_concat",
    #         "description": (
    #             "Flatten the 'input' message list (role: content pairs) and append "
    #             "the 'output' field. Format: 'User: ... \\nAssistant: ...' preserving "
    #             "multi-turn structure. For RL/when2call subsets, flatten the 'messages' "
    #             "list instead."
    #         ),
    #         "preprocessing": (
    #             "Strip system_prompt if it duplicates the first system message. "
    #             "Remove <think> tags for non-reasoning variants if needed."
    #         ),
    #     },
    # },

    # "Nemotron-Post-Training-Dataset-v1": {
    #     "hf_name": "nvidia/Nemotron-Post-Training-Dataset-v1",
    #     "category": "post-training",
    #     "format": "parquet",
    #     "sub_paths": [
    #         # Sharded parquet files under data/
    #         ("code", "data/code-*.parquet"),
    #         ("math", "data/math-*.parquet"),
    #         ("stem", "data/stem-*.parquet"),
    #         ("tool", "data/tool-*.parquet"),
    #         ("chat", "data/chat-*.parquet"),
    #     ],
    #     "text_strategy": {
    #         "fields": ["messages"],
    #         "template": "messages_list",
    #         "description": (
    #             "Flatten the 'messages' column (list of {role, content} dicts) into a "
    #             "single text. Format: 'User: ...\\nAssistant: ...' Multi-turn "
    #             "conversations are concatenated in order."
    #         ),
    #         "preprocessing": (
    #             "Handle nested tool_calls structs—serialize tool call names/arguments "
    #             "as text if present. Skip metadata column (JSON string)."
    #         ),
    #     },
    # },

    # "Nemotron-Post-Training-Dataset-v2": {
    #     "hf_name": "nvidia/Nemotron-Post-Training-Dataset-v2",
    #     "category": "post-training",
    #     "format": "parquet",
    #     "sub_paths": [
    #         ("chat", "data/chat-*.parquet"),
    #         ("code", "data/code-*.parquet"),
    #         ("math", "data/math-*.parquet"),
    #         ("stem", "data/stem-*.parquet"),
    #         ("multilingual", "data/multilingual-*.parquet"),
    #         ("multilingual_de", "data/multilingual_de-*.parquet"),
    #         ("multilingual_es", "data/multilingual_es-*.parquet"),
    #         ("multilingual_fr", "data/multilingual_fr-*.parquet"),
    #         ("multilingual_it", "data/multilingual_it-*.parquet"),
    #         ("multilingual_ja", "data/multilingual_ja-*.parquet"),
    #     ],
    #     "text_strategy": {
    #         "fields": ["messages"],
    #         "template": "messages_list",
    #         "description": (
    #             "Same as v1: flatten 'messages' list into role-prefixed text. "
    #             "Includes multilingual subsets (DE, ES, FR, IT, JA)—embeddings will "
    #             "capture cross-lingual semantics."
    #         ),
    #         "preprocessing": (
    #             "No tool_calls in v2 schema. Some shards have 'metadata' column "
    #             "(string)—ignore for embedding. Handle non-Latin scripts (JA) carefully."
    #         ),
    #     },
    # },

    "Nemotron-3-Nano-RL-Training-Blend": {
        "hf_name": "nvidia/Nemotron-3-Nano-RL-Training-Blend",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("train", "train.jsonl"),
        ],
        "text_strategy": {
            "fields": ["responses_create_params", "ground_truth"],
            "template": "rl_blend",
            "description": (
                "Extract 'input' messages from responses_create_params dict, flatten "
                "into role-prefixed text. Optionally append ground_truth tool calls "
                "as structured text."
            ),
            "preprocessing": (
                "Deep-extract nested dict: responses_create_params['input'] contains "
                "the message list. Ground truth is a list of tool-call dicts."
            ),
        },
    },

    "Nemotron-Science-v1": {
        "hf_name": "nvidia/Nemotron-Science-v1",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("MCQ", "data/MCQ.jsonl"),
            ("RQA", "data/RQA.jsonl"),
        ],
        "text_strategy": {
            "fields": ["messages"],
            "template": "messages_list",
            "description": (
                "Flatten 'messages' list (user question + assistant answer). "
                "MCQ: multiple-choice science questions. RQA: research Q&A with "
                "boxed answers."
            ),
            "preprocessing": "None needed—clean chat format.",
        },
    },

    "Nemotron-Instruction-Following-Chat-v1": {
        "hf_name": "nvidia/Nemotron-Instruction-Following-Chat-v1",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("chat_if", "data/chat_if.jsonl"),
            ("structured_outputs", "data/structured_outputs.jsonl"),
        ],
        "text_strategy": {
            "fields": ["messages"],
            "template": "messages_list",
            "description": (
                "Flatten 'messages' into text. Includes system prompts with specific "
                "instruction-following constraints and structured output schemas."
            ),
            "preprocessing": (
                "System messages may contain XML/JSON schemas for structured outputs. "
                "Keep them as-is for semantic richness."
            ),
        },
    },

    "Nemotron-Math-Proofs-v1": {
        "hf_name": "nvidia/Nemotron-Math-Proofs-v1",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("lean", "data/lean.jsonl"),
        ],
        "text_strategy": {
            "fields": ["problem", "formal_statement", "lean_header"],
            "template": "math_proof",
            "description": (
                "Concatenate: 'Problem: {problem}\\n\\nFormal Statement (Lean 4):\\n"
                "{lean_header}\\n{formal_statement}'. Combines natural-language math "
                "with formal verification code."
            ),
            "preprocessing": (
                "Many fields (url, user_name, sft_line_number) are None—skip them. "
                "The 'messages' field is often empty; use problem + formal_statement."
            ),
        },
    },

    "Nemotron-Agentic-v1": {
        "hf_name": "nvidia/Nemotron-Agentic-v1",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("tool_calling", "data/tool_calling.jsonl"),
            ("interactive_agent", "data/interactive_agent.jsonl"),
        ],
        "text_strategy": {
            "fields": ["messages", "tools"],
            "template": "agentic",
            "description": (
                "Flatten 'messages' into text. Prepend serialized 'tools' definitions "
                "(function name + description) as context. This captures the agentic "
                "task structure: available tools + conversation."
            ),
            "preprocessing": (
                "Serialize tool definitions to compact text: "
                "'Tool: {name} - {description}'. Skip detailed parameter schemas."
            ),
        },
    },

    "Nemotron-Competitive-Programming-v1": {
        "hf_name": "nvidia/Nemotron-Competitive-Programming-v1",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("cpp_part0", "data/competitive_coding_cpp.part_00.jsonl"),
            ("cpp_part1", "data/competitive_coding_cpp.part_01.jsonl"),
            ("python_part0", "data/competitive_coding_python.part_00.jsonl"),
            ("python_part1", "data/competitive_coding_python.part_01.jsonl"),
            ("infinibyte_part0", "data/infinibyte.part_00.jsonl"),
            ("infinibyte_part1", "data/infinibyte.part_01.jsonl"),
        ],
        "text_strategy": {
            "fields": ["messages"],
            "template": "messages_list",
            "description": (
                "Flatten 'messages' (user problem + assistant solution with code). "
                "Competitive programming: problem statement + reasoning + code solution."
            ),
            "preprocessing": (
                "Solutions contain code blocks—preserve formatting. "
                "Some messages have '-' as user content (placeholder); may need filtering."
            ),
        },
    },

    "Nemotron-Math-v2": {
        "hf_name": "nvidia/Nemotron-Math-v2",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("low", "data/low.jsonl"),
            ("medium", "data/medium.jsonl"),
            ("high_part0", "data/high.part_00.jsonl"),
            ("high_part1", "data/high.part_01.jsonl"),
            ("high_part2", "data/high.part_02.jsonl"),
        ],
        "text_strategy": {
            "fields": ["problem", "messages"],
            "template": "math_v2",
            "description": (
                "Primary text: flatten 'messages' (user prompt with \\\\boxed{} instruction "
                "+ assistant solution). Fallback: use 'problem' field directly. "
                "Difficulty tiers: low, medium, high."
            ),
            "preprocessing": (
                "Remove 'Solve the following math problem...' boilerplate prefix from "
                "user messages if desired. Keep \\\\boxed{answer} in text."
            ),
        },
    },

    "Nemotron-SWE-v1": {
        "hf_name": "nvidia/Nemotron-SWE-v1",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("r2e_gym", "data/r2e_gym.jsonl"),
        ],
        "text_strategy": {
            "fields": ["messages", "tools"],
            "template": "agentic",
            "description": (
                "Software engineering agent data: flatten long multi-turn 'messages' "
                "(system prompt with agent instructions + user bug report + assistant "
                "tool calls + tool outputs). Very long conversations."
            ),
            "preprocessing": (
                "WARNING: Messages can be extremely long (>100k chars). May need "
                "truncation to MODEL_MAX_TOKENS. System prompt is ~2k chars of "
                "agent instructions—consider summarizing or truncating."
            ),
        },
    },

    # ===== PRETRAINING DATASETS =====

    "Nemotron-Pretraining-Dataset-sample": {
        "hf_name": "nvidia/Nemotron-Pretraining-Dataset-sample",
        "category": "pretraining",
        "format": "parquet",
        "sub_paths": [
            ("CC-High-Quality", "Nemotron-CC-High-Quality/part_*.parquet"),
            ("CC-High-Quality-Synthetic", "Nemotron-CC-High-Quality-Synthetic/part_*.parquet"),
            ("CC-Diverse-QA", "Nemotron-CC-Diverse-QA/part_*.parquet"),
            ("CC-Translated-Diverse-QA", "Nemotron-CC-Translated-Diverse-QA/part_*.parquet"),
            ("CC-MATH", "Nemotron-CC-MATH/part_*.parquet"),
            ("Code-Metadata", "Nemotron-Code-Metadata/part_*.parquet"),
            ("SFT-Code", "Nemotron-SFT-Code/part_*.parquet"),
            ("SFT-General", "Nemotron-SFT-General/part_*.parquet"),
            ("SFT-MATH", "Nemotron-SFT-MATH/part_*.parquet"),
            ("Synthetic-Code", "Nemotron-Synthetic-Code/part_*.parquet"),
        ],
        "text_strategy": {
            "fields": ["text"],
            "template": "raw_text",
            "description": (
                "Use the 'text' column directly—already contains the full document. "
                "For CC subsets: web-crawled, cleaned text. For SFT subsets: "
                "instruction-response pairs pre-formatted as text."
            ),
            "preprocessing": (
                "Code-Metadata subset has NO 'text' column (only repo, commit_id, "
                "rel_path)—skip or use as metadata only. "
                "CC-Translated-Diverse-QA has a 'language' column for filtering."
            ),
        },
    },
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 chars per token)."""
    return len(text) // CHARS_PER_TOKEN


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
    parts = []
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
        # Pretraining data: use 'text' column directly
        return str(record.get("text", ""))

    elif template == "messages_list":
        # Most post-training datasets: flatten messages list
        messages = record.get("messages", [])
        return flatten_messages(messages)

    elif template == "messages_concat":
        # Llama-Nemotron: input (message list) + output (string)
        text_parts = []
        input_msgs = record.get("input", [])
        if isinstance(input_msgs, list):
            text_parts.append(flatten_messages(input_msgs))
        output = record.get("output", "")
        if output:
            text_parts.append(f"Assistant: {output}")
        # Fallback to messages if input/output not present (when2call subsets)
        if not text_parts:
            messages = record.get("messages", [])
            return flatten_messages(messages)
        return "\n\n".join(text_parts)

    elif template == "rl_blend":
        # Nemotron-3-Nano-RL: nested dict extraction
        params = record.get("responses_create_params", {})
        input_msgs = params.get("input", []) if isinstance(params, dict) else []
        text = flatten_messages(input_msgs)
        gt = record.get("ground_truth", [])
        if gt:
            text += f"\n\nGround Truth: {json.dumps(gt, ensure_ascii=False)}"
        return text

    elif template == "math_proof":
        # Math proofs: problem + formal statement
        parts = []
        problem = record.get("problem", "")
        if problem:
            parts.append(f"Problem: {problem}")
        lean_header = record.get("lean_header", "")
        formal = record.get("formal_statement", "")
        if lean_header or formal:
            parts.append(f"Formal Statement (Lean 4):\n{lean_header}\n{formal}")
        # Fallback to messages if available
        messages = record.get("messages", [])
        if messages:
            parts.append(flatten_messages(messages))
        return "\n\n".join(parts)

    elif template == "math_v2":
        # Math v2: prefer messages, fallback to problem
        messages = record.get("messages", [])
        if messages:
            return flatten_messages(messages)
        return str(record.get("problem", ""))

    elif template == "agentic":
        # Agentic/SWE: tools context + messages
        parts = []
        tools = record.get("tools", [])
        if tools and isinstance(tools, list):
            tool_descs = []
            for t in tools[:10]:  # Cap tool descriptions to avoid huge text
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


def load_jsonl_sample(filepath: str, n: int = SAMPLE_SIZE) -> List[dict]:
    """Load the first n records from a JSONL file."""
    records = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        logger.warning(f"  Error reading {filepath}: {e}")
    return records


def count_jsonl_rows(filepath: str) -> int:
    """Count total rows in a JSONL file efficiently using raw byte counting."""
    count = 0
    try:
        with open(filepath, "rb") as f:
            # Read in large chunks for speed (~100x faster than line iteration)
            buf_size = 1024 * 1024  # 1 MB buffer
            while True:
                buf = f.read(buf_size)
                if not buf:
                    break
                count += buf.count(b"\n")
    except Exception as e:
        logger.warning(f"  Error counting {filepath}: {e}")
    return count


def get_jsonl_schema(records: List[dict]) -> Dict[str, str]:
    """Infer schema (field name -> type) from sample records."""
    schema = {}
    for rec in records:
        for key, val in rec.items():
            if key not in schema:
                schema[key] = type(val).__name__
    return schema


def load_parquet_sample(filepath: str, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Load the first n rows from a parquet file."""
    try:
        pf = pq.ParquetFile(filepath)
        table = pf.read_row_groups([0])
        df = table.to_pandas()
        return df.head(n)
    except Exception:
        pass
    # Fallback: use pandas read_parquet with pyarrow (handles more edge cases)
    try:
        df = pd.read_parquet(filepath, engine="pyarrow")
        return df.head(n)
    except Exception as e:
        logger.warning(f"  Error reading parquet {filepath}: {e}")
        return pd.DataFrame()


def get_parquet_metadata(filepath: str) -> Tuple[dict, int]:
    """Get schema and row count from a parquet file without reading data."""
    try:
        pf = pq.ParquetFile(filepath)
        schema = {}
        for i in range(len(pf.schema_arrow)):
            field = pf.schema_arrow.field(i)
            schema[field.name] = str(field.type)
        return schema, pf.metadata.num_rows
    except Exception:
        pass
    # Fallback: read file with pandas to get schema and count
    try:
        import pyarrow.parquet as pq2
        metadata = pq2.read_metadata(filepath)
        schema_arrow = pq2.read_schema(filepath)
        schema = {}
        for i in range(len(schema_arrow)):
            field = schema_arrow.field(i)
            schema[field.name] = str(field.type)
        return schema, metadata.num_rows
    except Exception as e:
        logger.warning(f"  Error reading parquet metadata {filepath}: {e}")
        return {}, 0


def resolve_glob(base_dir: str, pattern: str) -> List[str]:
    """Resolve a glob pattern relative to base_dir, returning sorted file paths."""
    import glob
    full_pattern = os.path.join(base_dir, pattern)
    files = sorted(glob.glob(full_pattern))
    return files


# ---------------------------------------------------------------------------
# Per-dataset exploration
# ---------------------------------------------------------------------------


def explore_sub_path(
    base_dir: str,
    label: str,
    rel_path: str,
    file_format: str,
    config: dict,
) -> Optional[dict]:
    """
    Explore a single sub-path (file or glob) within a dataset.

    Returns a dict with:
        label, files_found, total_rows, schema, sample_records,
        text_length_stats, quality_notes
    """
    result = {
        "label": label,
        "files_found": 0,
        "total_rows": 0,
        "schema": {},
        "sample_texts": [],
        "text_lengths": [],
        "token_counts": [],
        "quality_notes": [],
    }

    # Resolve files
    files = resolve_glob(base_dir, rel_path)
    if not files:
        result["quality_notes"].append(f"No files found matching {rel_path}")
        logger.warning(f"    No files found: {rel_path}")
        return result

    result["files_found"] = len(files)
    logger.info(f"    [{label}] Found {len(files)} file(s)")

    if file_format == "jsonl":
        # Get schema from first file's first record
        sample_records = load_jsonl_sample(files[0], SAMPLE_SIZE)
        if sample_records:
            result["schema"] = get_jsonl_schema(sample_records)

        # Count total rows across all files
        total_rows = 0
        for fpath in files:
            total_rows += count_jsonl_rows(fpath)
        result["total_rows"] = total_rows

        # Compute text lengths from sample
        stats_records = load_jsonl_sample(files[0], STATS_SAMPLE_SIZE)
        for rec in stats_records:
            text = extract_text_from_record(rec, config)
            tlen = len(text)
            result["text_lengths"].append(tlen)
            result["token_counts"].append(estimate_tokens(text))

        # Sample texts (first 3)
        for rec in sample_records[:3]:
            text = extract_text_from_record(rec, config)
            preview = text[:300] + "..." if len(text) > 300 else text
            result["sample_texts"].append(preview)

        # Quality checks
        if sample_records:
            for key in result["schema"]:
                none_count = sum(1 for r in stats_records if r.get(key) is None)
                if none_count > 0:
                    pct = none_count / len(stats_records) * 100
                    if pct > 5:
                        result["quality_notes"].append(
                            f"Field '{key}': {pct:.0f}% null values"
                        )

    elif file_format == "parquet":
        # Get schema from first file
        schema, first_file_rows = get_parquet_metadata(files[0])
        result["schema"] = schema

        # Count total rows across all shards
        total_rows = 0
        for fpath in files:
            try:
                pf = pq.ParquetFile(fpath)
                total_rows += pf.metadata.num_rows
            except Exception:
                pass
        result["total_rows"] = total_rows

        # Sample data and compute text lengths
        try:
            sample_df = load_parquet_sample(files[0], STATS_SAMPLE_SIZE)
            if not sample_df.empty:
                for _, row in sample_df.iterrows():
                    rec = row.to_dict()
                    text = extract_text_from_record(rec, config)
                    tlen = len(text)
                    result["text_lengths"].append(tlen)
                    result["token_counts"].append(estimate_tokens(text))

                # Sample texts (first 3)
                for _, row in sample_df.head(3).iterrows():
                    rec = row.to_dict()
                    text = extract_text_from_record(rec, config)
                    preview = text[:300] + "..." if len(text) > 300 else text
                    result["sample_texts"].append(preview)

                # Quality: check for nulls
                for col in sample_df.columns:
                    null_pct = sample_df[col].isna().mean() * 100
                    if null_pct > 5:
                        result["quality_notes"].append(
                            f"Field '{col}': {null_pct:.0f}% null values"
                        )
        except Exception as e:
            result["quality_notes"].append(f"Error sampling data: {e}")

    return result


def explore_dataset(name: str, config: dict) -> dict:
    """
    Explore a single dataset: iterate over all sub_paths, collect metadata.

    Returns a summary dict for this dataset.
    """
    hf_name = config["hf_name"]
    local_dir_name = hf_name.replace("/", "_")
    base_dir = os.path.join(DEFAULT_DATASETS_DIR, local_dir_name)

    logger.info(f"\n{'='*70}")
    logger.info(f"Exploring: {name}")
    logger.info(f"  Local dir: {base_dir}")
    logger.info(f"  Format: {config['format']}")
    logger.info(f"  Category: {config['category']}")

    dataset_result = {
        "name": name,
        "hf_name": hf_name,
        "category": config["category"],
        "format": config["format"],
        "local_dir": base_dir,
        "exists": os.path.isdir(base_dir),
        "sub_results": [],
        "total_rows": 0,
        "text_strategy": config.get("text_strategy", {}),
        "error": None,
    }

    if not dataset_result["exists"]:
        msg = f"Dataset directory not found: {base_dir}"
        logger.warning(f"  {msg}")
        dataset_result["error"] = msg
        return dataset_result

    # Explore each sub-path
    for label, rel_path in config.get("sub_paths", []):
        try:
            sub = explore_sub_path(
                base_dir, label, rel_path, config["format"], config
            )
            if sub:
                dataset_result["sub_results"].append(sub)
                dataset_result["total_rows"] += sub["total_rows"]
        except Exception as e:
            logger.error(f"    Error exploring {label}: {e}")
            dataset_result["sub_results"].append({
                "label": label,
                "error": str(e),
                "files_found": 0,
                "total_rows": 0,
                "schema": {},
                "text_lengths": [],
                "token_counts": [],
                "sample_texts": [],
                "quality_notes": [f"Error: {e}"],
            })

    logger.info(f"  Total rows: {dataset_result['total_rows']:,}")
    return dataset_result


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def compute_aggregate_stats(sub_results: list) -> dict:
    """Compute aggregate text length / token stats across all sub-results."""
    all_lengths = []
    all_tokens = []
    for sub in sub_results:
        all_lengths.extend(sub.get("text_lengths", []))
        all_tokens.extend(sub.get("token_counts", []))

    if not all_lengths:
        return {
            "min_chars": 0, "max_chars": 0, "mean_chars": 0, "median_chars": 0,
            "min_tokens": 0, "max_tokens": 0, "mean_tokens": 0, "median_tokens": 0,
            "pct_over_max": 0,
        }

    all_lengths.sort()
    all_tokens.sort()
    n = len(all_lengths)

    over_max = sum(1 for t in all_tokens if t > MODEL_MAX_TOKENS)

    return {
        "min_chars": all_lengths[0],
        "max_chars": all_lengths[-1],
        "mean_chars": int(sum(all_lengths) / n),
        "median_chars": all_lengths[n // 2],
        "min_tokens": all_tokens[0],
        "max_tokens": all_tokens[-1],
        "mean_tokens": int(sum(all_tokens) / n),
        "median_tokens": all_tokens[n // 2],
        "pct_over_max": round(over_max / n * 100, 1),
    }


def format_number(n: int) -> str:
    """Format a number with commas."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def generate_markdown(results: List[dict]) -> str:
    """Generate the full markdown report from exploration results."""
    lines = []
    lines.append("# Nemotron Dataset Exploration Summary")
    lines.append("")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Datasets directory**: `{DEFAULT_DATASETS_DIR}`")
    lines.append(f"- **Default chunk size**: {DEFAULT_CHUNK_SIZE:,} rows")
    lines.append(f"- **Model max tokens**: {MODEL_MAX_TOKENS:,}")
    lines.append(f"- **Token estimation**: ~{CHARS_PER_TOKEN} chars/token")
    lines.append("")

    # ---- Summary table ----
    lines.append("## Dataset Summary Table")
    lines.append("")
    lines.append(
        "| # | Dataset | Category | Format | Row Count | "
        "Text Strategy | Median Tokens | % Over Max |"
    )
    lines.append(
        "|---|---------|----------|--------|-----------|"
        "---------------|---------------|------------|"
    )

    for i, res in enumerate(results, 1):
        stats = compute_aggregate_stats(res.get("sub_results", []))
        strategy_desc = res.get("text_strategy", {}).get("template", "N/A")
        status = "" if res["exists"] else " (NOT FOUND)"
        lines.append(
            f"| {i} | {res['name']}{status} | {res['category']} | "
            f"{res['format']} | {format_number(res['total_rows'])} | "
            f"`{strategy_desc}` | "
            f"{format_number(stats['median_tokens'])} | "
            f"{stats['pct_over_max']}% |"
        )

    lines.append("")

    # ---- Detailed findings per dataset ----
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Findings")
    lines.append("")

    # Group by category
    post_training = [r for r in results if r["category"] == "post-training"]
    pretraining = [r for r in results if r["category"] == "pretraining"]

    for category_label, category_results in [
        ("Post-Training Datasets", post_training),
        ("Pretraining Datasets", pretraining),
    ]:
        lines.append(f"### {category_label}")
        lines.append("")

        for res in category_results:
            lines.append(f"#### {res['name']}")
            lines.append("")
            lines.append(f"- **HuggingFace**: `{res['hf_name']}`")
            lines.append(f"- **Format**: {res['format']}")
            lines.append(f"- **Total rows**: {res['total_rows']:,}")

            if not res["exists"]:
                lines.append(f"- **Status**: NOT FOUND on disk")
                lines.append("")
                continue

            if res.get("error"):
                lines.append(f"- **Error**: {res['error']}")
                lines.append("")
                continue

            # Text strategy
            strategy = res.get("text_strategy", {})
            lines.append(f"- **Text concatenation strategy**: {strategy.get('description', 'N/A')}")
            if strategy.get("preprocessing"):
                lines.append(f"- **Preprocessing notes**: {strategy['preprocessing']}")

            # Aggregate stats
            stats = compute_aggregate_stats(res.get("sub_results", []))
            if stats["mean_tokens"] > 0:
                lines.append(
                    f"- **Text length stats** (sampled): "
                    f"min={format_number(stats['min_tokens'])} / "
                    f"median={format_number(stats['median_tokens'])} / "
                    f"mean={format_number(stats['mean_tokens'])} / "
                    f"max={format_number(stats['max_tokens'])} tokens"
                )
                if stats["pct_over_max"] > 0:
                    lines.append(
                        f"- **Texts exceeding MODEL_MAX_TOKENS ({MODEL_MAX_TOKENS:,})**: "
                        f"{stats['pct_over_max']}% of sampled records"
                    )

            # Sub-path breakdown
            lines.append("")
            lines.append("**Sub-path breakdown:**")
            lines.append("")
            lines.append("| Sub-path | Files | Rows | Schema Fields |")
            lines.append("|----------|-------|------|---------------|")
            for sub in res.get("sub_results", []):
                schema_fields = ", ".join(sub.get("schema", {}).keys())
                if len(schema_fields) > 60:
                    schema_fields = schema_fields[:57] + "..."
                lines.append(
                    f"| {sub['label']} | {sub['files_found']} | "
                    f"{format_number(sub['total_rows'])} | {schema_fields} |"
                )
            lines.append("")

            # Quality notes
            all_notes = []
            for sub in res.get("sub_results", []):
                for note in sub.get("quality_notes", []):
                    all_notes.append(f"[{sub['label']}] {note}")
            if all_notes:
                lines.append("**Data quality observations:**")
                lines.append("")
                for note in all_notes[:8]:  # Cap at 8 notes
                    lines.append(f"- {note}")
                lines.append("")

            # Batch processing recommendation
            num_chunks = max(1, res["total_rows"] // DEFAULT_CHUNK_SIZE)
            lines.append("**Processing recommendation:**")
            lines.append("")
            if res["total_rows"] <= DEFAULT_CHUNK_SIZE:
                lines.append(
                    f"- Single batch processing (total rows {res['total_rows']:,} "
                    f"< chunk size {DEFAULT_CHUNK_SIZE:,})"
                )
            else:
                lines.append(
                    f"- Chunked processing: ~{num_chunks} chunks of "
                    f"{DEFAULT_CHUNK_SIZE:,} rows each"
                )
            if stats["pct_over_max"] > 10:
                lines.append(
                    f"- **WARNING**: {stats['pct_over_max']}% of texts exceed "
                    f"MODEL_MAX_TOKENS—truncation required"
                )
            if stats["max_tokens"] > MODEL_MAX_TOKENS * 2:
                lines.append(
                    f"- Consider chunking long texts into overlapping windows "
                    f"(max observed: {format_number(stats['max_tokens'])} tokens)"
                )
            lines.append("")

    # ---- Processing recommendations by category ----
    lines.append("---")
    lines.append("")
    lines.append("## Processing Recommendations by Category")
    lines.append("")

    lines.append("### Post-Training Datasets")
    lines.append("")
    lines.append(
        "Post-training datasets use **chat/instruction format** with `messages` lists. "
        "The primary text concatenation strategy is to flatten multi-turn conversations "
        "into role-prefixed text blocks."
    )
    lines.append("")
    lines.append("**Common approach:**")
    lines.append("1. Extract `messages` list from each record")
    lines.append("2. Format as `Role: content` with double-newline separators")
    lines.append("3. For datasets with `input`/`output` fields (Llama-Nemotron), combine both")
    lines.append("4. For agentic datasets, prepend tool definitions as context")
    lines.append("5. Truncate to MODEL_MAX_TOKENS if text exceeds limit")
    lines.append("")
    lines.append("**Special cases:**")
    lines.append("- **Math-Proofs-v1**: Use `problem` + `formal_statement` (Lean 4 code) instead of empty `messages`")
    lines.append("- **Math-v2**: Use `messages` with `problem` as fallback")
    lines.append("- **SWE-v1**: Very long agent conversations—truncation essential")
    lines.append("- **Competitive-Programming**: Contains code blocks—preserve formatting")
    lines.append("- **Nano-RL-Blend**: Nested dict structure requires deep extraction")
    lines.append("")

    lines.append("### Pretraining Datasets")
    lines.append("")
    lines.append(
        "Pretraining datasets use **raw text format** with a simple `text` column. "
        "No concatenation needed—use the text column directly."
    )
    lines.append("")
    lines.append("**Common approach:**")
    lines.append("1. Read `text` column directly from parquet files")
    lines.append("2. Skip `Code-Metadata` subset (no text column—metadata only)")
    lines.append("3. For `CC-Translated-Diverse-QA`, note the `language` column for filtering")
    lines.append("4. The sample dataset is small (~27K rows total); full pretraining sets are TB-scale")
    lines.append("")

    # ---- Appendix: schema reference ----
    lines.append("---")
    lines.append("")
    lines.append("## Appendix: Schema Reference")
    lines.append("")

    for res in results:
        if not res["exists"] or not res.get("sub_results"):
            continue
        lines.append(f"### {res['name']}")
        lines.append("")
        for sub in res.get("sub_results", []):
            if sub.get("schema"):
                lines.append(f"**{sub['label']}:**")
                lines.append("```")
                for field_name, field_type in sub["schema"].items():
                    lines.append(f"  {field_name}: {field_type}")
                lines.append("```")
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 70)
    logger.info("Nemotron Dataset Explorer")
    logger.info("=" * 70)
    logger.info(f"Datasets dir: {DEFAULT_DATASETS_DIR}")
    logger.info(f"Output file:  {OUTPUT_FILE}")
    logger.info(f"Datasets to explore: {len(DATASET_CONFIGS)}")

    all_results = []

    for i, (name, config) in enumerate(DATASET_CONFIGS.items(), 1):
        logger.info(f"\n[{i}/{len(DATASET_CONFIGS)}] Processing {name}...")
        try:
            result = explore_dataset(name, config)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to explore {name}: {e}")
            all_results.append({
                "name": name,
                "hf_name": config["hf_name"],
                "category": config["category"],
                "format": config["format"],
                "local_dir": "",
                "exists": False,
                "sub_results": [],
                "total_rows": 0,
                "text_strategy": config.get("text_strategy", {}),
                "error": str(e),
            })

    # Generate markdown report
    logger.info("\n" + "=" * 70)
    logger.info("Generating markdown report...")

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    markdown = generate_markdown(all_results)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(markdown)

    logger.info(f"Report written to: {OUTPUT_FILE}")
    logger.info(f"Report size: {len(markdown):,} characters")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Exploration Summary")
    logger.info("=" * 70)
    total_rows = sum(r["total_rows"] for r in all_results)
    found = sum(1 for r in all_results if r["exists"])
    logger.info(f"Datasets found: {found}/{len(all_results)}")
    logger.info(f"Total rows across all datasets: {total_rows:,}")
    logger.info(f"Report: {OUTPUT_FILE}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
