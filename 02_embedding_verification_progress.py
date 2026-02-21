#!/usr/bin/env python3
"""
02_embedding_verification_progress.py – Check embedding extraction progress

Scans the embeddings output directory and cross-references with DATASET_CONFIGS
to report which datasets/sub-paths are fully complete, partially complete,
or not yet started.

Usage
-----
    python 02_embedding_verification_progress.py
    python 02_embedding_verification_progress.py --embeddings-dir /raid/embeddings_curator
    python 02_embedding_verification_progress.py --json   # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

from dataset_configs import DATASET_CONFIGS

DEFAULT_EMBEDDINGS_DIR = "/raid/embeddings_curator"
DEFAULT_DATASETS_DIR = "/raid/datasets"

# Terminal colours
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


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
        status = "complete"  # preprocessed may have been cleaned up
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
        "row_mismatch": (pre_rows != emb_rows) if (pre_files > 0 and emb_files > 0) else False,
    }

    if metadata:
        result["meta_docs_processed"] = metadata.get("num_documents_processed")
        result["meta_time_s"] = metadata.get("time_taken_s")
        result["meta_throughput"] = metadata.get("throughput_docs_per_sec")
        result["meta_timestamp"] = metadata.get("timestamp")

    return result


def format_rows(n: int) -> str:
    if n == 0:
        return "-"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def status_icon(status: str) -> str:
    icons = {
        "complete": f"{GREEN}COMPLETE{RESET}",
        "partial_embeddings": f"{YELLOW}PARTIAL (embeddings incomplete){RESET}",
        "preprocessed_only": f"{YELLOW}PARTIAL (preprocessed only){RESET}",
        "started_empty": f"{YELLOW}STARTED (empty dirs){RESET}",
        "not_started": f"{RED}NOT STARTED{RESET}",
    }
    return icons.get(status, status)


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

    if args.json:
        summary = {
            "embeddings_dir": args.embeddings_dir,
            "datasets_dir": args.datasets_dir,
            "datasets": {},
        }
        for ds_key, subs in by_dataset.items():
            summary["datasets"][ds_key] = {
                "status": dataset_status(subs),
                "sub_paths": subs,
            }
        json.dump(summary, sys.stdout, indent=2)
        print()
        return

    # ── Header ──
    print(f"\n{BOLD}{'=' * 90}")
    print("  EMBEDDING EXTRACTION PROGRESS REPORT")
    print(f"{'=' * 90}{RESET}")
    print(f"  Embeddings dir : {args.embeddings_dir}")
    print(f"  Datasets dir   : {args.datasets_dir}")
    print(f"  Total datasets : {len(by_dataset)}")
    total_subs = len(all_results)
    complete_subs = sum(1 for r in all_results if r["status"] == "complete")
    partial_subs = sum(1 for r in all_results if r["status"] in ("partial_embeddings", "preprocessed_only", "started_empty"))
    not_started_subs = sum(1 for r in all_results if r["status"] == "not_started")
    print(f"  Total sub-paths: {total_subs}  ({GREEN}{complete_subs} complete{RESET}, "
          f"{YELLOW}{partial_subs} partial{RESET}, {RED}{not_started_subs} not started{RESET})")

    total_emb_rows = sum(r["embedding_rows"] for r in all_results)
    total_pre_rows = sum(r["preprocessed_rows"] for r in all_results)
    print(f"  Total embedded : {format_rows(total_emb_rows)} rows")
    print(f"  Total preproc  : {format_rows(total_pre_rows)} rows")

    # ── Dataset-level summary ──
    complete_ds = [k for k, v in by_dataset.items() if dataset_status(v) == "complete"]
    partial_ds = [k for k, v in by_dataset.items() if dataset_status(v) == "partial"]
    not_started_ds = [k for k, v in by_dataset.items() if dataset_status(v) == "not_started"]

    print(f"\n{BOLD}{'─' * 90}")
    print(f"  DATASET-LEVEL SUMMARY")
    print(f"{'─' * 90}{RESET}")

    if complete_ds:
        print(f"\n  {GREEN}{BOLD}FULLY COMPLETE ({len(complete_ds)} datasets):{RESET}")
        for ds in complete_ds:
            subs = by_dataset[ds]
            total_rows = sum(r["embedding_rows"] for r in subs)
            sub_count = len(subs)
            print(f"    {GREEN}[OK]{RESET}  {ds}  ({sub_count} sub-paths, {format_rows(total_rows)} rows)")

    if partial_ds:
        print(f"\n  {YELLOW}{BOLD}PARTIALLY COMPLETE ({len(partial_ds)} datasets):{RESET}")
        for ds in partial_ds:
            subs = by_dataset[ds]
            done = sum(1 for r in subs if r["status"] == "complete")
            print(f"    {YELLOW}[..]{RESET}  {ds}  ({done}/{len(subs)} sub-paths complete)")

    if not_started_ds:
        print(f"\n  {RED}{BOLD}NOT YET STARTED ({len(not_started_ds)} datasets):{RESET}")
        for ds in not_started_ds:
            subs = by_dataset[ds]
            src = "source downloaded" if subs[0].get("source_available") else "source NOT found"
            print(f"    {RED}[  ]{RESET}  {ds}  ({len(subs)} sub-paths, {src})")

    # ── Detailed breakdown ──
    print(f"\n{BOLD}{'─' * 90}")
    print(f"  DETAILED BREAKDOWN BY SUB-PATH")
    print(f"{'─' * 90}{RESET}")

    for ds_key, subs in by_dataset.items():
        ds_stat = dataset_status(subs)
        ds_colour = GREEN if ds_stat == "complete" else (YELLOW if ds_stat == "partial" else RED)
        print(f"\n  {BOLD}{ds_colour}{ds_key}{RESET}  [{ds_stat.upper()}]")

        header = f"    {'Sub-path':<30} {'Status':<38} {'Pre Files':>10} {'Emb Files':>10} {'Pre Rows':>10} {'Emb Rows':>10} {'Meta':>5}"
        print(f"{DIM}{header}{RESET}")

        for r in subs:
            meta_mark = f"{GREEN}Y{RESET}" if r["has_metadata"] else f"{DIM}-{RESET}"
            mismatch = f" {YELLOW}!row mismatch{RESET}" if r.get("row_mismatch") else ""
            print(
                f"    {r['sub_label']:<30} "
                f"{status_icon(r['status']):<50} "
                f"{r['preprocessed_files']:>10} "
                f"{r['embedding_files']:>10} "
                f"{format_rows(r['preprocessed_rows']):>10} "
                f"{format_rows(r['embedding_rows']):>10} "
                f"{meta_mark:>5}"
                f"{mismatch}"
            )

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
                print(f"    - {r['dataset_key']}/{r['sub_label']}  "
                      f"({r['embedding_files']}/{r['preprocessed_files']} files embedded)")

        if preproc_only:
            print(f"\n  {YELLOW}Preprocessed but no embeddings (need embedding run):{RESET}")
            for r in preproc_only:
                print(f"    - {r['dataset_key']}/{r['sub_label']}  "
                      f"({r['preprocessed_files']} preprocessed files, {format_rows(r['preprocessed_rows'])} rows)")

        if started_empty:
            print(f"\n  {YELLOW}Started but empty (preprocessing may have failed):{RESET}")
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
                    for r in not_started if r["dataset_key"] == ds
                )
                if not src_avail:
                    print(f"      {RED}WARNING: source data not downloaded!{RESET}")

        remaining_rows = sum(r["preprocessed_rows"] for r in pending if r["preprocessed_rows"] > 0)
        print(f"\n  Remaining preprocessed rows needing embedding: {format_rows(remaining_rows)}")
    else:
        print(f"\n  {GREEN}{BOLD}All datasets fully complete!{RESET}")

    print(f"\n{'=' * 90}\n")


if __name__ == "__main__":
    main()
