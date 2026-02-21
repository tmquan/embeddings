#!/usr/bin/env python3
"""
02_embedding_incomplete_removal.py – Remove incomplete embedding artifacts

Scans the embeddings output directory, identifies sub-paths that are partially
complete or not yet started, and removes their preprocessed/ and embeddings/
directories so the extraction pipeline can start fresh on a clean slate.

Complete sub-paths (preprocessed + embeddings files match) are never touched.

Usage
-----
    # Dry-run (default): show what would be deleted
    python 02_embedding_incomplete_removal.py

    # Actually delete
    python 02_embedding_incomplete_removal.py --delete

    # Custom directory
    python 02_embedding_incomplete_removal.py --delete --embeddings-dir /raid/embeddings_curator
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Any, Dict, List

from dataset_configs import DATASET_CONFIGS

DEFAULT_EMBEDDINGS_DIR = "/raid/embeddings_curator"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
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


def classify_sub_path(embeddings_dir: str, ds_key: str, sub_label: str) -> Dict[str, Any]:
    base = os.path.join(embeddings_dir, ds_key, sub_label)
    preprocess_dir = os.path.join(base, "preprocessed")
    emb_dir = os.path.join(base, "embeddings")

    pre_files = count_parquet_files(preprocess_dir)
    emb_files = count_parquet_files(emb_dir)
    has_pre_dir = os.path.isdir(preprocess_dir)
    has_emb_dir = os.path.isdir(emb_dir)

    if emb_files > 0 and pre_files > 0 and emb_files == pre_files:
        status = "complete"
    elif emb_files > 0 and pre_files == 0:
        status = "complete"
    elif emb_files > 0 and pre_files > 0 and emb_files < pre_files:
        status = "partial_embeddings"
    elif pre_files > 0 and emb_files == 0:
        status = "preprocessed_only"
    elif has_pre_dir or has_emb_dir:
        status = "started_empty"
    else:
        status = "not_started"

    dirs_to_remove: List[str] = []
    if status != "complete" and status != "not_started":
        if has_pre_dir:
            dirs_to_remove.append(preprocess_dir)
        if has_emb_dir:
            dirs_to_remove.append(emb_dir)
        meta_path = os.path.join(base, "metadata.json")
        if os.path.isfile(meta_path):
            dirs_to_remove.append(meta_path)

    return {
        "dataset_key": ds_key,
        "sub_label": sub_label,
        "status": status,
        "preprocessed_files": pre_files,
        "embedding_files": emb_files,
        "base_dir": base,
        "dirs_to_remove": dirs_to_remove,
    }


def remove_path(path: str) -> str:
    if os.path.isdir(path):
        n_files = sum(1 for _, _, files in os.walk(path) for _ in files)
        shutil.rmtree(path)
        return f"removed dir  ({n_files} files)"
    elif os.path.isfile(path):
        os.remove(path)
        return "removed file"
    return "not found"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove preprocessed and embedding directories for incomplete sub-paths.",
    )
    parser.add_argument(
        "--embeddings-dir",
        default=DEFAULT_EMBEDDINGS_DIR,
        help=f"Root embeddings directory (default: {DEFAULT_EMBEDDINGS_DIR})",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete. Without this flag, only a dry-run is performed.",
    )
    args = parser.parse_args()

    mode = "DELETE" if args.delete else "DRY RUN"

    print(f"\n{BOLD}{'=' * 80}")
    print(f"  INCOMPLETE EMBEDDING REMOVAL  [{mode}]")
    print(f"{'=' * 80}{RESET}")
    print(f"  Embeddings dir: {args.embeddings_dir}\n")

    all_results: List[Dict[str, Any]] = []
    for ds_key, ds_cfg in DATASET_CONFIGS.items():
        for sub_label, _ in ds_cfg.get("sub_paths", []):
            all_results.append(classify_sub_path(args.embeddings_dir, ds_key, sub_label))

    complete = [r for r in all_results if r["status"] == "complete"]
    to_clean = [r for r in all_results if r["dirs_to_remove"]]
    not_started = [r for r in all_results if r["status"] == "not_started"]

    # Show complete (kept)
    print(f"  {GREEN}Complete (untouched): {len(complete)} sub-paths{RESET}")
    for r in complete:
        print(f"    {GREEN}[OK]{RESET}  {r['dataset_key']}/{r['sub_label']}  "
              f"(pre={r['preprocessed_files']}, emb={r['embedding_files']})")

    # Show not started (nothing to do)
    print(f"\n  {DIM}Not started (nothing on disk): {len(not_started)} sub-paths{RESET}")

    # Show items to clean
    if not to_clean:
        print(f"\n  {GREEN}{BOLD}Nothing to remove — all sub-paths are either complete or not started.{RESET}")
        print(f"\n{'=' * 80}\n")
        return

    print(f"\n  {RED}{BOLD}To remove: {len(to_clean)} sub-paths{RESET}")
    total_removed_dirs = 0
    for r in to_clean:
        print(f"\n    {YELLOW}[{r['status']}]{RESET}  {r['dataset_key']}/{r['sub_label']}  "
              f"(pre={r['preprocessed_files']}, emb={r['embedding_files']})")
        for path in r["dirs_to_remove"]:
            rel = os.path.relpath(path, args.embeddings_dir)
            if args.delete:
                result = remove_path(path)
                print(f"      {RED}DEL{RESET}  {rel}  — {result}")
            else:
                exists = "dir" if os.path.isdir(path) else ("file" if os.path.isfile(path) else "???")
                print(f"      {DIM}would remove{RESET}  {rel}  ({exists})")
            total_removed_dirs += 1

    # Clean up empty parent directories
    if args.delete:
        cleaned_parents = 0
        for r in to_clean:
            base = r["base_dir"]
            if os.path.isdir(base) and not os.listdir(base):
                os.rmdir(base)
                cleaned_parents += 1
                # Check dataset-level dir too
                ds_dir = os.path.dirname(base)
                if os.path.isdir(ds_dir) and not os.listdir(ds_dir):
                    os.rmdir(ds_dir)
                    cleaned_parents += 1
        if cleaned_parents:
            print(f"\n  {DIM}Cleaned up {cleaned_parents} empty parent directories{RESET}")

    # Summary
    print(f"\n{BOLD}{'─' * 80}{RESET}")
    if args.delete:
        print(f"  {RED}{BOLD}Removed {total_removed_dirs} paths across {len(to_clean)} sub-paths{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}Would remove {total_removed_dirs} paths across {len(to_clean)} sub-paths{RESET}")
        print(f"  {DIM}Run with --delete to actually remove them{RESET}")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
