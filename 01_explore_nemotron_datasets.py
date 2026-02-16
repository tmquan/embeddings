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
# DATASET_CONFIGS and preprocessing – imported from shared modules
# ---------------------------------------------------------------------------

from dataset_configs import DATASET_CONFIGS  # noqa: E402
from dataset_preprocess import (  # noqa: E402
    estimate_tokens,
    flatten_messages,
    extract_text_from_record,
    count_jsonl_rows_fast as count_jsonl_rows,
    count_parquet_rows,
    resolve_glob,
)

# ---------------------------------------------------------------------------
# Helper functions (explore-specific)
# ---------------------------------------------------------------------------


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
