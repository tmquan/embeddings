#!/usr/bin/env python3
"""
02_embedding_verification_progress.py – Check embedding extraction progress

Scans the embeddings output directory and cross-references with DATASET_CONFIGS
to report which datasets/sub-paths are fully complete, partially complete,
or not yet started.  Also verifies source data: schema, row counts, text
extraction strategies, combined concatenation preview, and sample rows.

Usage
-----
    python 02_embedding_verification_progress.py
    python 02_embedding_verification_progress.py --embeddings-dir /raid/embeddings_curator
    python 02_embedding_verification_progress.py --json
    python 02_embedding_verification_progress.py --skip-source
    python 02_embedding_verification_progress.py --markdown report.md
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import os
import re
import sys
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

from dataset_configs import DATASET_CONFIGS

DEFAULT_EMBEDDINGS_DIR = "/raid/embeddings_curator"
DEFAULT_DATASETS_DIR = "/raid/datasets"
MAX_EXAMPLE_VALUE_LEN = 120
MAX_COMBINED_TEXT_LEN = 500
MAX_COMBINED_TEXT_LEN_MD = 1000
NUM_EXAMPLES = 3

# Terminal colours
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ─── Number formatting ───────────────────────────────────────────────────────


def format_approx(n: int) -> str:
    """Compact approximate format: 1.23M, 45.7K, etc."""
    if n == 0:
        return "-"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_count(n: int) -> str:
    """Dual format: '1,234,567 (~1.23M)' or just '0' for zero."""
    if n == 0:
        return "0"
    exact = f"{n:,}"
    if n >= 1_000:
        return f"{exact} (~{format_approx(n)})"
    return exact


# ─── File & source helpers ────────────────────────────────────────────────────


def count_parquet_files(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    return sum(
        1 for f in os.listdir(directory)
        if f.endswith(".parquet") and not f.startswith(".")
    )


def count_parquet_rows(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    total = 0
    for f in sorted(os.listdir(directory)):
        if f.endswith(".parquet") and not f.startswith("."):
            try:
                total += pq.ParquetFile(os.path.join(directory, f)).metadata.num_rows
            except Exception:
                pass
    return total


def load_metadata(meta_path: str) -> Optional[Dict[str, Any]]:
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def check_source_available(datasets_dir: str, hf_name: str) -> bool:
    local_dir = os.path.join(datasets_dir, hf_name.replace("/", "_"))
    return os.path.isdir(local_dir)


def resolve_source_files(datasets_dir: str, hf_name: str, rel_pattern: str) -> List[str]:
    local_dir = os.path.join(datasets_dir, hf_name.replace("/", "_"))
    pattern = os.path.join(local_dir, rel_pattern)
    return sorted(glob_mod.glob(pattern))


def count_file_lines_fast(filepath: str) -> int:
    count = 0
    with open(filepath, "rb") as f:
        while True:
            buf = f.read(1024 * 1024)
            if not buf:
                break
            count += buf.count(b"\n")
    return count


def hf_link(hf_name: str) -> str:
    return f"https://huggingface.co/datasets/{hf_name}"


def truncate_value(val: Any, max_len: int = MAX_EXAMPLE_VALUE_LEN) -> str:
    s = str(val)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def get_source_info(
    datasets_dir: str,
    hf_name: str,
    rel_pattern: str,
    fmt: str,
) -> Dict[str, Any]:
    """Get schema, row count, and top N examples from source files."""
    files = resolve_source_files(datasets_dir, hf_name, rel_pattern)
    info: Dict[str, Any] = {
        "num_files": len(files),
        "total_rows": 0,
        "columns": [],
        "top_examples": [],
        "error": None,
    }
    if not files:
        info["error"] = "no source files found"
        return info

    try:
        if fmt == "parquet":
            pf = pq.ParquetFile(files[0])
            schema = pf.schema_arrow
            info["columns"] = [(field.name, str(field.type)) for field in schema]

            for fpath in files:
                try:
                    info["total_rows"] += pq.ParquetFile(fpath).metadata.num_rows
                except Exception:
                    pass

            table = pq.read_table(files[0]).slice(0, NUM_EXAMPLES)
            d = table.to_pydict()
            keys = list(d.keys())
            for i in range(table.num_rows):
                info["top_examples"].append({k: d[k][i] for k in keys})

        elif fmt == "jsonl":
            examples: List[dict] = []
            col_types: Dict[str, str] = {}
            with open(files[0], "r") as fh:
                for i, line in enumerate(fh):
                    if i >= NUM_EXAMPLES:
                        break
                    row = json.loads(line.strip())
                    examples.append(row)
                    for k, v in row.items():
                        if k not in col_types:
                            col_types[k] = type(v).__name__
            info["top_examples"] = examples
            info["columns"] = list(col_types.items())

            for fpath in files:
                try:
                    info["total_rows"] += count_file_lines_fast(fpath)
                except Exception:
                    pass
    except Exception as e:
        info["error"] = str(e)

    return info


# ─── Text concatenation strategies ───────────────────────────────────────────


def _flatten_messages(messages: Any) -> str:
    """Flatten a list of {role, content} dicts into role-prefixed text."""
    if not isinstance(messages, list):
        return str(messages)
    parts: List[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            if content is None:
                content = ""
            parts.append(f"{role}: {content}")
        else:
            parts.append(str(msg))
    return "\n".join(parts)


def apply_text_strategy(row: dict, strategy: dict) -> str:
    """Apply the configured text extraction strategy to a single row,
    returning the combined text that would be fed to the embedding model."""
    template = strategy.get("template", "")
    fields = strategy.get("fields", [])

    if template == "messages_list":
        return _flatten_messages(row.get("messages", []))

    if template == "messages_concat":
        input_msgs = row.get("input", row.get("messages", []))
        output_val = row.get("output", "")
        parts: List[str] = []
        if isinstance(input_msgs, list):
            parts.append(_flatten_messages(input_msgs))
        elif input_msgs:
            parts.append(str(input_msgs))
        if output_val:
            parts.append(
                f"Assistant: {output_val}"
                if isinstance(output_val, str)
                else str(output_val)
            )
        return "\n".join(parts)

    if template == "rl_blend":
        params = row.get("responses_create_params", {})
        parts = []
        if isinstance(params, dict):
            parts.append(_flatten_messages(params.get("input", [])))
        else:
            parts.append(str(params))
        gt = row.get("ground_truth")
        if gt:
            parts.append(f"Ground Truth: {truncate_value(gt, 300)}")
        return "\n".join(parts)

    if template == "math_proof":
        problem = row.get("problem", "")
        formal = row.get("formal_statement", "")
        header = row.get("lean_header", "")
        return (
            f"Problem: {problem}\n\n"
            f"Formal Statement (Lean 4):\n{header}\n{formal}"
        )

    if template == "agentic":
        tools = row.get("tools", [])
        messages = row.get("messages", [])
        parts = []
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    fn = tool.get("function", tool)
                    name = fn.get("name", "")
                    desc = fn.get("description", "")
                    parts.append(f"Tool: {name} - {desc}")
        parts.append(_flatten_messages(messages))
        return "\n".join(parts)

    if template == "math_v2":
        messages = row.get("messages", [])
        if isinstance(messages, list) and messages:
            return _flatten_messages(messages)
        return str(row.get("problem", ""))

    if template == "raw_text":
        return str(row.get("text", ""))

    # Fallback: concatenate all specified fields
    parts = []
    for field in fields:
        val = row.get(field)
        if val is not None:
            parts.append(str(val))
    return "\n".join(parts) if parts else str(row)


# ─── Embedding analysis ──────────────────────────────────────────────────────


def analyse_sub_path(
    embeddings_dir: str,
    ds_key: str,
    sub_label: str,
) -> Dict[str, Any]:
    base = os.path.join(embeddings_dir, ds_key, sub_label)
    preprocess_dir = os.path.join(base, "preprocessed")
    emb_dir = os.path.join(base, "embeddings")
    meta_path = os.path.join(base, "metadata.json")

    pre_files = count_parquet_files(preprocess_dir)
    emb_files = count_parquet_files(emb_dir)
    pre_rows = count_parquet_rows(preprocess_dir) if pre_files > 0 else 0
    emb_rows = count_parquet_rows(emb_dir) if emb_files > 0 else 0
    metadata = load_metadata(meta_path)

    has_preprocess_dir = os.path.isdir(preprocess_dir)
    has_emb_dir = os.path.isdir(emb_dir)

    if emb_files > 0 and pre_files > 0 and emb_files == pre_files:
        status = "complete"
    elif emb_files > 0 and pre_files > 0 and emb_files < pre_files:
        status = "partial_embeddings"
    elif emb_files > 0 and pre_files == 0:
        status = "complete"
    elif pre_files > 0 and emb_files == 0:
        status = "preprocessed_only"
    elif has_preprocess_dir or has_emb_dir:
        status = "started_empty"
    else:
        status = "not_started"

    result: Dict[str, Any] = {
        "dataset_key": ds_key,
        "sub_label": sub_label,
        "status": status,
        "preprocessed_files": pre_files,
        "embedding_files": emb_files,
        "preprocessed_rows": pre_rows,
        "embedding_rows": emb_rows,
        "has_metadata": metadata is not None,
        "row_mismatch": (pre_rows != emb_rows)
        if (pre_files > 0 and emb_files > 0)
        else False,
    }

    if metadata:
        result["meta_docs_processed"] = metadata.get("num_documents_processed")
        result["meta_time_s"] = metadata.get("time_taken_s")
        result["meta_throughput"] = metadata.get("throughput_docs_per_sec")
        result["meta_timestamp"] = metadata.get("timestamp")

    return result


# ─── Display helpers ──────────────────────────────────────────────────────────


def status_icon(status: str) -> str:
    icons = {
        "complete": f"{GREEN}COMPLETE{RESET}",
        "partial_embeddings": f"{YELLOW}PARTIAL (embeddings incomplete){RESET}",
        "preprocessed_only": f"{YELLOW}PARTIAL (preprocessed only){RESET}",
        "started_empty": f"{YELLOW}STARTED (empty dirs){RESET}",
        "not_started": f"{RED}NOT STARTED{RESET}",
    }
    return icons.get(status, status)


STATUS_LABELS = {
    "complete": "COMPLETE",
    "partial_embeddings": "PARTIAL (embeddings incomplete)",
    "preprocessed_only": "PARTIAL (preprocessed only)",
    "started_empty": "STARTED (empty dirs)",
    "not_started": "NOT STARTED",
}


def dataset_status(sub_results: List[Dict[str, Any]]) -> str:
    statuses = {r["status"] for r in sub_results}
    if statuses == {"complete"}:
        return "complete"
    if "complete" in statuses or "partial_embeddings" in statuses or "preprocessed_only" in statuses:
        return "partial"
    if statuses <= {"started_empty", "not_started"}:
        if "started_empty" in statuses:
            return "partial"
        return "not_started"
    return "partial"


def print_wrapped(text: str, indent: int = 6, width: int = 86) -> None:
    prefix = " " * indent
    for line in textwrap.wrap(text, width=width - indent):
        print(f"{prefix}{line}")


def print_examples(
    examples: List[dict],
    columns: List[Tuple[str, str]],
    strategy: Optional[dict] = None,
) -> None:
    if not examples:
        print(f"      {DIM}(no examples available){RESET}")
        return
    col_names = [c[0] for c in columns] if columns else list(examples[0].keys())
    for i, row in enumerate(examples):
        print(f"      {DIM}── Example {i + 1} ──{RESET}")
        for col in col_names:
            val = row.get(col, "N/A")
            truncated = truncate_value(val)
            print(f"        {CYAN}{col}{RESET}: {truncated}")
        if strategy:
            combined = apply_text_strategy(row, strategy)
            combined_trunc = truncate_value(combined, MAX_COMBINED_TEXT_LEN)
            print(
                f"        {MAGENTA}{BOLD}Combined text "
                f"({strategy.get('template', '?')}):{RESET}"
            )
            for cline in combined_trunc.split("\n"):
                print(f"          {DIM}│{RESET} {cline}")


# ─── Markdown report writer ──────────────────────────────────────────────────


def _md_status_emoji(status: str) -> str:
    return {
        "complete": "✅",
        "partial_embeddings": "🟡",
        "preprocessed_only": "🟡",
        "started_empty": "🟡",
        "not_started": "❌",
    }.get(status, "❓")


def _md_ds_status_emoji(status: str) -> str:
    return {"complete": "✅", "partial": "🟡", "not_started": "❌"}.get(status, "❓")


def write_markdown_report(
    filepath: str,
    args: argparse.Namespace,
    all_results: List[Dict[str, Any]],
    by_dataset: Dict[str, List[Dict[str, Any]]],
    source_info: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """Write the full verification report as a Markdown file."""
    lines: List[str] = []
    w = lines.append

    w("# Embedding Extraction Progress Report\n")
    w(f"| Key | Value |")
    w(f"|-----|-------|")
    w(f"| Embeddings dir | `{args.embeddings_dir}` |")
    w(f"| Datasets dir | `{args.datasets_dir}` |")
    w(f"| Total datasets | {len(by_dataset)} |")

    total_subs = len(all_results)
    complete_subs = sum(1 for r in all_results if r["status"] == "complete")
    partial_subs = sum(
        1
        for r in all_results
        if r["status"] in ("partial_embeddings", "preprocessed_only", "started_empty")
    )
    not_started_subs = sum(1 for r in all_results if r["status"] == "not_started")
    w(
        f"| Total sub-paths | {total_subs} "
        f"(✅ {complete_subs} complete, 🟡 {partial_subs} partial, "
        f"❌ {not_started_subs} not started) |"
    )

    total_emb_rows = sum(r["embedding_rows"] for r in all_results)
    total_pre_rows = sum(r["preprocessed_rows"] for r in all_results)
    w(f"| Total embedded | {format_count(total_emb_rows)} rows |")
    w(f"| Total preprocessed | {format_count(total_pre_rows)} rows |")
    w("")

    # ── Dataset-level summary ──
    complete_ds = [k for k, v in by_dataset.items() if dataset_status(v) == "complete"]
    partial_ds = [k for k, v in by_dataset.items() if dataset_status(v) == "partial"]
    not_started_ds = [
        k for k, v in by_dataset.items() if dataset_status(v) == "not_started"
    ]

    w("## Dataset-Level Summary\n")

    if complete_ds:
        w(f"### ✅ Fully Complete ({len(complete_ds)} datasets)\n")
        for ds in complete_ds:
            subs = by_dataset[ds]
            total_rows = sum(r["embedding_rows"] for r in subs)
            w(f"- **{ds}** — {len(subs)} sub-paths, {format_count(total_rows)} rows")
        w("")

    if partial_ds:
        w(f"### 🟡 Partially Complete ({len(partial_ds)} datasets)\n")
        for ds in partial_ds:
            subs = by_dataset[ds]
            done = sum(1 for r in subs if r["status"] == "complete")
            w(f"- **{ds}** — {done}/{len(subs)} sub-paths complete")
        w("")

    if not_started_ds:
        w(f"### ❌ Not Yet Started ({len(not_started_ds)} datasets)\n")
        for ds in not_started_ds:
            subs = by_dataset[ds]
            src = (
                "source downloaded"
                if subs[0].get("source_available")
                else "**source NOT found**"
            )
            w(f"- **{ds}** — {len(subs)} sub-paths, {src}")
        w("")

    # ── Detailed breakdown ──
    w("## Detailed Breakdown by Sub-path\n")

    for ds_key, subs in by_dataset.items():
        ds_stat = dataset_status(subs)
        w(f"### {_md_ds_status_emoji(ds_stat)} {ds_key} [{ds_stat.upper()}]\n")
        w(
            "| Sub-path | Status | Pre Files | Emb Files "
            "| Pre Rows | Emb Rows | Meta |"
        )
        w("|----------|--------|-----------|-----------|----------|----------|------|")
        for r in subs:
            meta_mark = "✅" if r["has_metadata"] else "-"
            mismatch = " ⚠️" if r.get("row_mismatch") else ""
            label = STATUS_LABELS.get(r["status"], r["status"])
            w(
                f"| {r['sub_label']} "
                f"| {_md_status_emoji(r['status'])} {label} "
                f"| {r['preprocessed_files']} "
                f"| {r['embedding_files']} "
                f"| {format_approx(r['preprocessed_rows'])} "
                f"| {format_approx(r['embedding_rows'])} "
                f"| {meta_mark}{mismatch} |"
            )
        w("")

    # ── Source data verification ──
    if source_info:
        w("## Source Data Verification\n")

        for ds_key, ds_cfg in DATASET_CONFIGS.items():
            if ds_key not in by_dataset:
                continue
            ds_stat = dataset_status(by_dataset[ds_key])

            w(f"### {_md_ds_status_emoji(ds_stat)} {ds_key}\n")
            w(
                f"**HuggingFace:** [{ds_cfg['hf_name']}]"
                f"({hf_link(ds_cfg['hf_name'])})"
            )
            w(
                f"**Format:** {ds_cfg['format']} | "
                f"**Category:** {ds_cfg.get('category', 'unknown')}\n"
            )

            # Text extraction strategy
            ts = ds_cfg.get("text_strategy", {})
            if ts:
                w("#### Text Extraction Strategy\n")
                w(f"- **Template:** `{ts.get('template', 'N/A')}`")
                w(f"- **Fields:** `{', '.join(ts.get('fields', []))}`")
                if ts.get("description"):
                    w(f"- **Strategy:** {ts['description']}")
                if ts.get("preprocessing"):
                    w(f"- **Preprocessing:** {ts['preprocessing']}")
                w("")

            # Aggregate totals
            ds_source_total = sum(
                source_info.get(ds_key, {}).get(sl, {}).get("total_rows", 0)
                for sl, _ in ds_cfg.get("sub_paths", [])
            )
            ds_emb_total = sum(r["embedding_rows"] for r in by_dataset[ds_key])
            w(f"**Total source rows:** {format_count(ds_source_total)}  ")
            w(f"**Total embed rows:** {format_count(ds_emb_total)}\n")

            # Per sub-path
            for sub_label, rel_pattern in ds_cfg.get("sub_paths", []):
                si = source_info.get(ds_key, {}).get(sub_label, {})
                if not si:
                    continue
                emb_result = next(
                    (r for r in by_dataset[ds_key] if r["sub_label"] == sub_label),
                    None,
                )

                w(f"#### `{sub_label}` — `{rel_pattern}`\n")

                if si.get("error"):
                    w(f"> ⚠️ **Error:** {si['error']}\n")
                    continue

                src_rows = si.get("total_rows", 0)
                w(f"- **Source files:** {si.get('num_files', 0)}")
                w(f"- **Source rows:** {format_count(src_rows)}")
                if emb_result:
                    w(
                        f"- **Embed rows:** "
                        f"{format_count(emb_result['embedding_rows'])}"
                    )
                    w(
                        f"- **Preproc rows:** "
                        f"{format_count(emb_result['preprocessed_rows'])}"
                    )
                w("")

                # Columns
                columns = si.get("columns", [])
                if columns:
                    w(f"**Columns ({len(columns)}):**\n")
                    w("| Column | Type |")
                    w("|--------|------|")
                    for col_name, col_dtype in columns:
                        w(f"| `{col_name}` | `{col_dtype}` |")
                    w("")

                # Examples + combined text
                examples = si.get("top_examples", [])
                if examples:
                    col_names = (
                        [c[0] for c in columns]
                        if columns
                        else list(examples[0].keys())
                    )
                    w(f"**Top {len(examples)} Examples:**\n")
                    for idx, row in enumerate(examples):
                        w(f"<details>")
                        w(f"<summary>Example {idx + 1}</summary>\n")
                        for col in col_names:
                            val = row.get(col, "N/A")
                            w(f"- **{col}:** `{truncate_value(val)}`")
                        if ts:
                            combined = apply_text_strategy(row, ts)
                            combined_trunc = truncate_value(
                                combined, MAX_COMBINED_TEXT_LEN_MD
                            )
                            w(
                                f"\n**Combined text** "
                                f"(`{ts.get('template', '?')}`):\n"
                            )
                            w("```")
                            w(combined_trunc)
                            w("```")
                        w("\n</details>\n")

    # ── Next steps ──
    pending = [r for r in all_results if r["status"] != "complete"]
    if pending:
        w("## Next Steps\n")

        partial_emb = [r for r in pending if r["status"] == "partial_embeddings"]
        preproc_only = [r for r in pending if r["status"] == "preprocessed_only"]
        started_empty = [r for r in pending if r["status"] == "started_empty"]
        not_started = [r for r in pending if r["status"] == "not_started"]

        if partial_emb:
            w("### Incomplete embeddings (need re-run)\n")
            for r in partial_emb:
                w(
                    f"- `{r['dataset_key']}/{r['sub_label']}` — "
                    f"{r['embedding_files']}/{r['preprocessed_files']} files embedded"
                )
            w("")

        if preproc_only:
            w("### Preprocessed but no embeddings\n")
            for r in preproc_only:
                w(
                    f"- `{r['dataset_key']}/{r['sub_label']}` — "
                    f"{r['preprocessed_files']} preprocessed files, "
                    f"{format_count(r['preprocessed_rows'])} rows"
                )
            w("")

        if started_empty:
            w("### Started but empty (preprocessing may have failed)\n")
            for r in started_empty:
                w(f"- `{r['dataset_key']}/{r['sub_label']}`")
            w("")

        if not_started:
            ds_groups: Dict[str, List[str]] = defaultdict(list)
            for r in not_started:
                ds_groups[r["dataset_key"]].append(r["sub_label"])
            w("### Not started (run extraction)\n")
            for ds, labels in ds_groups.items():
                src_avail = any(
                    r.get("source_available")
                    for r in not_started
                    if r["dataset_key"] == ds
                )
                warning = "" if src_avail else " ⚠️ **source data not downloaded!**"
                w(f"- **{ds}:** {', '.join(f'`{l}`' for l in labels)}{warning}")
            w("")

        remaining_rows = sum(
            r["preprocessed_rows"] for r in pending if r["preprocessed_rows"] > 0
        )
        w(
            f"**Remaining preprocessed rows needing embedding:** "
            f"{format_count(remaining_rows)}"
        )
    else:
        w("## ✅ All datasets fully complete!")

    w("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check embedding extraction progress across all datasets."
    )
    parser.add_argument(
        "--embeddings-dir",
        default=DEFAULT_EMBEDDINGS_DIR,
        help=f"Root embeddings directory (default: {DEFAULT_EMBEDDINGS_DIR})",
    )
    parser.add_argument(
        "--datasets-dir",
        default=DEFAULT_DATASETS_DIR,
        help=f"Root datasets directory (default: {DEFAULT_DATASETS_DIR})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON instead of formatted text.",
    )
    parser.add_argument(
        "--skip-source",
        action="store_true",
        help="Skip source data verification (schema, row counts, examples).",
    )
    parser.add_argument(
        "--markdown",
        metavar="FILE",
        default=None,
        help="Write report as Markdown to FILE (e.g. --markdown report.md).",
    )
    args = parser.parse_args()

    all_results: List[Dict[str, Any]] = []
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for ds_key, ds_cfg in DATASET_CONFIGS.items():
        for sub_label, _rel_pattern in ds_cfg.get("sub_paths", []):
            result = analyse_sub_path(args.embeddings_dir, ds_key, sub_label)
            result["source_available"] = check_source_available(
                args.datasets_dir, ds_cfg["hf_name"]
            )
            result["category"] = ds_cfg.get("category", "unknown")
            all_results.append(result)
            by_dataset[ds_key].append(result)

    # ── Collect source info if not skipped ──
    source_info: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not args.skip_source:
        for ds_key, ds_cfg in DATASET_CONFIGS.items():
            source_info[ds_key] = {}
            for sub_label, rel_pattern in ds_cfg.get("sub_paths", []):
                source_info[ds_key][sub_label] = get_source_info(
                    args.datasets_dir,
                    ds_cfg["hf_name"],
                    rel_pattern,
                    ds_cfg["format"],
                )

    # ── JSON output ──
    if args.json:
        summary: Dict[str, Any] = {
            "embeddings_dir": args.embeddings_dir,
            "datasets_dir": args.datasets_dir,
            "datasets": {},
        }
        for ds_key, subs in by_dataset.items():
            ds_cfg = DATASET_CONFIGS[ds_key]
            ds_entry: Dict[str, Any] = {
                "hf_link": hf_link(ds_cfg["hf_name"]),
                "status": dataset_status(subs),
                "sub_paths": [],
            }
            for r in subs:
                sub_entry = dict(r)
                if ds_key in source_info and r["sub_label"] in source_info[ds_key]:
                    si = source_info[ds_key][r["sub_label"]]
                    sub_entry["source_files"] = si["num_files"]
                    sub_entry["source_rows"] = si["total_rows"]
                    sub_entry["source_columns"] = si["columns"]
                    sub_entry["source_examples"] = si["top_examples"]
                    ts = ds_cfg.get("text_strategy")
                    if ts and si["top_examples"]:
                        sub_entry["combined_text_preview"] = [
                            truncate_value(
                                apply_text_strategy(ex, ts),
                                MAX_COMBINED_TEXT_LEN_MD,
                            )
                            for ex in si["top_examples"]
                        ]
                sub_entry["text_strategy"] = ds_cfg.get("text_strategy")
                ds_entry["sub_paths"].append(sub_entry)
            summary["datasets"][ds_key] = ds_entry
        json.dump(summary, sys.stdout, indent=2, default=str)
        print()
        return

    # ── Markdown output ──
    if args.markdown:
        write_markdown_report(
            args.markdown, args, all_results, by_dataset, source_info
        )
        print(f"Markdown report written to {args.markdown}")
        return

    # ── Terminal: Header ──
    print(f"\n{BOLD}{'=' * 90}")
    print("  EMBEDDING EXTRACTION PROGRESS REPORT")
    print(f"{'=' * 90}{RESET}")
    print(f"  Embeddings dir : {args.embeddings_dir}")
    print(f"  Datasets dir   : {args.datasets_dir}")
    print(f"  Total datasets : {len(by_dataset)}")
    total_subs = len(all_results)
    complete_subs = sum(1 for r in all_results if r["status"] == "complete")
    partial_subs = sum(
        1
        for r in all_results
        if r["status"] in ("partial_embeddings", "preprocessed_only", "started_empty")
    )
    not_started_subs = sum(1 for r in all_results if r["status"] == "not_started")
    print(
        f"  Total sub-paths: {total_subs}  ({GREEN}{complete_subs} complete{RESET}, "
        f"{YELLOW}{partial_subs} partial{RESET}, {RED}{not_started_subs} not started{RESET})"
    )

    total_emb_rows = sum(r["embedding_rows"] for r in all_results)
    total_pre_rows = sum(r["preprocessed_rows"] for r in all_results)
    print(f"  Total embedded : {format_count(total_emb_rows)} rows")
    print(f"  Total preproc  : {format_count(total_pre_rows)} rows")

    # ── Dataset-level summary ──
    complete_ds = [k for k, v in by_dataset.items() if dataset_status(v) == "complete"]
    partial_ds = [k for k, v in by_dataset.items() if dataset_status(v) == "partial"]
    not_started_ds = [
        k for k, v in by_dataset.items() if dataset_status(v) == "not_started"
    ]

    print(f"\n{BOLD}{'─' * 90}")
    print(f"  DATASET-LEVEL SUMMARY")
    print(f"{'─' * 90}{RESET}")

    if complete_ds:
        print(f"\n  {GREEN}{BOLD}FULLY COMPLETE ({len(complete_ds)} datasets):{RESET}")
        for ds in complete_ds:
            subs = by_dataset[ds]
            total_rows = sum(r["embedding_rows"] for r in subs)
            sub_count = len(subs)
            print(
                f"    {GREEN}[OK]{RESET}  {ds}  "
                f"({sub_count} sub-paths, {format_count(total_rows)} rows)"
            )

    if partial_ds:
        print(
            f"\n  {YELLOW}{BOLD}PARTIALLY COMPLETE ({len(partial_ds)} datasets):{RESET}"
        )
        for ds in partial_ds:
            subs = by_dataset[ds]
            done = sum(1 for r in subs if r["status"] == "complete")
            print(
                f"    {YELLOW}[..]{RESET}  {ds}  ({done}/{len(subs)} sub-paths complete)"
            )

    if not_started_ds:
        print(
            f"\n  {RED}{BOLD}NOT YET STARTED ({len(not_started_ds)} datasets):{RESET}"
        )
        for ds in not_started_ds:
            subs = by_dataset[ds]
            src = (
                "source downloaded"
                if subs[0].get("source_available")
                else "source NOT found"
            )
            print(f"    {RED}[  ]{RESET}  {ds}  ({len(subs)} sub-paths, {src})")

    # ── Detailed breakdown ──
    print(f"\n{BOLD}{'─' * 90}")
    print(f"  DETAILED BREAKDOWN BY SUB-PATH")
    print(f"{'─' * 90}{RESET}")

    for ds_key, subs in by_dataset.items():
        ds_stat = dataset_status(subs)
        ds_colour = (
            GREEN
            if ds_stat == "complete"
            else (YELLOW if ds_stat == "partial" else RED)
        )
        print(f"\n  {BOLD}{ds_colour}{ds_key}{RESET}  [{ds_stat.upper()}]")

        header = (
            f"    {'Sub-path':<30} {'Status':<38} "
            f"{'Pre Files':>10} {'Emb Files':>10} "
            f"{'Pre Rows':>10} {'Emb Rows':>10} {'Meta':>5}"
        )
        print(f"{DIM}{header}{RESET}")

        for r in subs:
            meta_mark = f"{GREEN}Y{RESET}" if r["has_metadata"] else f"{DIM}-{RESET}"
            mismatch = (
                f" {YELLOW}!row mismatch{RESET}" if r.get("row_mismatch") else ""
            )
            print(
                f"    {r['sub_label']:<30} "
                f"{status_icon(r['status']):<50} "
                f"{r['preprocessed_files']:>10} "
                f"{r['embedding_files']:>10} "
                f"{format_approx(r['preprocessed_rows']):>10} "
                f"{format_approx(r['embedding_rows']):>10} "
                f"{meta_mark:>5}"
                f"{mismatch}"
            )

    # ── Source data verification ──
    if not args.skip_source and source_info:
        print(f"\n{BOLD}{'=' * 90}")
        print("  SOURCE DATA VERIFICATION")
        print(f"{'=' * 90}{RESET}")

        for ds_key, ds_cfg in DATASET_CONFIGS.items():
            ds_stat = dataset_status(by_dataset[ds_key])
            ds_colour = (
                GREEN
                if ds_stat == "complete"
                else (YELLOW if ds_stat == "partial" else RED)
            )

            print(f"\n  {BOLD}{ds_colour}{'━' * 86}{RESET}")
            print(f"  {BOLD}{ds_key}{RESET}")
            print(f"  {BLUE}HuggingFace:{RESET} {hf_link(ds_cfg['hf_name'])}")
            print(
                f"  {DIM}Format: {ds_cfg['format']} | "
                f"Category: {ds_cfg.get('category', 'unknown')}{RESET}"
            )

            # Text extraction strategy
            ts = ds_cfg.get("text_strategy", {})
            if ts:
                print(f"\n    {MAGENTA}{BOLD}Text Extraction Strategy:{RESET}")
                print(f"      Template : {ts.get('template', 'N/A')}")
                print(f"      Fields   : {', '.join(ts.get('fields', []))}")
                if ts.get("description"):
                    print(f"      Strategy :")
                    print_wrapped(ts["description"], indent=17, width=90)
                if ts.get("preprocessing"):
                    print(f"      Preproc  :")
                    print_wrapped(str(ts["preprocessing"]), indent=17, width=90)

            # Aggregate source rows across all sub-paths for this dataset
            ds_source_total = sum(
                source_info.get(ds_key, {}).get(sl, {}).get("total_rows", 0)
                for sl, _ in ds_cfg.get("sub_paths", [])
            )
            ds_emb_total = sum(r["embedding_rows"] for r in by_dataset[ds_key])
            print(
                f"\n    {BOLD}Total source rows:{RESET} {format_count(ds_source_total)}"
            )
            print(f"    {BOLD}Total embed rows :{RESET} {format_count(ds_emb_total)}")

            # Per sub-path details
            for sub_label, rel_pattern in ds_cfg.get("sub_paths", []):
                si = source_info.get(ds_key, {}).get(sub_label, {})
                if not si:
                    continue

                emb_result = next(
                    (r for r in by_dataset[ds_key] if r["sub_label"] == sub_label),
                    None,
                )

                print(
                    f"\n    {CYAN}{BOLD}▸ {sub_label}{RESET}  {DIM}({rel_pattern}){RESET}"
                )

                if si.get("error"):
                    print(f"      {RED}Error: {si['error']}{RESET}")
                    continue

                src_rows = si.get("total_rows", 0)
                num_files = si.get("num_files", 0)
                print(f"      Source files : {num_files}")
                print(f"      Source rows  : {format_count(src_rows)}")
                if emb_result:
                    print(
                        f"      Embed rows   : "
                        f"{format_count(emb_result['embedding_rows'])}"
                    )
                    print(
                        f"      Preproc rows : "
                        f"{format_count(emb_result['preprocessed_rows'])}"
                    )

                # Columns with datatypes
                columns = si.get("columns", [])
                if columns:
                    print(f"      {BOLD}Columns ({len(columns)}):{RESET}")
                    for col_name, col_dtype in columns:
                        print(
                            f"        {CYAN}{col_name:<30}{RESET} "
                            f"{DIM}{col_dtype}{RESET}"
                        )

                # Top N examples + combined concatenation preview
                examples = si.get("top_examples", [])
                if examples:
                    print(f"      {BOLD}Top {len(examples)} Examples:{RESET}")
                    print_examples(examples, columns, strategy=ts if ts else None)

    # ── Actionable next steps ──
    pending = [r for r in all_results if r["status"] != "complete"]
    if pending:
        print(f"\n{BOLD}{'─' * 90}")
        print(f"  NEXT STEPS")
        print(f"{'─' * 90}{RESET}")

        partial_emb = [r for r in pending if r["status"] == "partial_embeddings"]
        preproc_only = [r for r in pending if r["status"] == "preprocessed_only"]
        started_empty = [r for r in pending if r["status"] == "started_empty"]
        not_started = [r for r in pending if r["status"] == "not_started"]

        if partial_emb:
            print(f"\n  {YELLOW}Incomplete embeddings (need re-run):{RESET}")
            for r in partial_emb:
                print(
                    f"    - {r['dataset_key']}/{r['sub_label']}  "
                    f"({r['embedding_files']}/{r['preprocessed_files']} files embedded)"
                )

        if preproc_only:
            print(
                f"\n  {YELLOW}Preprocessed but no embeddings (need embedding run):{RESET}"
            )
            for r in preproc_only:
                print(
                    f"    - {r['dataset_key']}/{r['sub_label']}  "
                    f"({r['preprocessed_files']} preprocessed files, "
                    f"{format_count(r['preprocessed_rows'])} rows)"
                )

        if started_empty:
            print(
                f"\n  {YELLOW}Started but empty (preprocessing may have failed):{RESET}"
            )
            for r in started_empty:
                print(f"    - {r['dataset_key']}/{r['sub_label']}")

        if not_started:
            ds_groups: Dict[str, List[str]] = defaultdict(list)
            for r in not_started:
                ds_groups[r["dataset_key"]].append(r["sub_label"])
            print(f"\n  {RED}Not started (run extraction):{RESET}")
            for ds, labels in ds_groups.items():
                print(f"    - {ds}: {', '.join(labels)}")
                src_avail = any(
                    r.get("source_available")
                    for r in not_started
                    if r["dataset_key"] == ds
                )
                if not src_avail:
                    print(f"      {RED}WARNING: source data not downloaded!{RESET}")

        remaining_rows = sum(
            r["preprocessed_rows"] for r in pending if r["preprocessed_rows"] > 0
        )
        print(
            f"\n  Remaining preprocessed rows needing embedding: "
            f"{format_count(remaining_rows)}"
        )
    else:
        print(f"\n  {GREEN}{BOLD}All datasets fully complete!{RESET}")

    print(f"\n{'=' * 90}\n")


if __name__ == "__main__":
    main()
