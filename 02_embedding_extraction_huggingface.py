#!/usr/bin/env python3
"""
02_embedding_extraction.py – Multi-GPU Nemotron Dataset Embedding Extraction

Extracts dense embeddings from Nemotron datasets using nvidia/llama-embed-nemotron-8b
across all available GPUs (default 8).  Produces chunked .npy files in float32 format.

Workflow
--------
1. Discovers all dataset files from DATASET_CONFIGS.
2. Creates work units (one per source file / sub-path label).
3. Distributes work across GPU workers using LPT (Longest Processing Time first)
   scheduling for optimal load balance.
4. Each worker loads the model on its assigned GPU, streams through records,
   extracts embeddings in batches, and saves chunked .npy outputs.
5. Resume-safe: existing chunk files are skipped automatically.

Output structure
----------------
    /raid/embeddings/
    └── {dataset_key}/              # e.g. Nemotron-Science-v1
        └── {sub_label}/            # e.g. MCQ
            ├── embeddings_00000.npy   # [≤chunk_size, 4096] float32
            ├── embeddings_00001.npy
            └── metadata.json

Usage
-----
    # Process all available datasets with defaults
    python 02_embedding_extraction.py

    # Process specific datasets
    python 02_embedding_extraction.py --datasets Nemotron-Science-v1 Nemotron-Math-v2

    # Custom batch / sequence / GPU settings
    python 02_embedding_extraction.py --batch-size 4 --max-length 4096 --num-gpus 4

    # Dry-run: show work plan without processing
    python 02_embedding_extraction.py --dry-run

Requirements
------------
    conda activate nemotron
    # torch, transformers, numpy, pyarrow (all pre-installed)
"""

from __future__ import annotations

import argparse
import gc
import glob as glob_module
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "nvidia/llama-embed-nemotron-8b"
EMBEDDING_DIM = 4096

DEFAULT_DATASETS_DIR = "/raid/datasets"
DEFAULT_EMBEDDINGS_DIR = "/raid/embeddings"
DEFAULT_MAX_LENGTH = 8192
DEFAULT_BATCH_SIZE = 8
DEFAULT_CHUNK_SIZE = 50_000   # records per .npy output chunk
DEFAULT_NUM_GPUS = 8

# Rough tokenization estimate (for progress reporting only)
CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FMT = "%(asctime)s [GPU-%(gpu)s] %(levelname)s %(message)s"


def _setup_root_logger() -> logging.Logger:
    """Module-level logger (main process)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [MAIN] %(levelname)s %(message)s")
    )
    logger = logging.getLogger("embed")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _setup_root_logger()


def _get_worker_logger(gpu_id: int) -> logging.Logger:
    """Per-worker logger that tags messages with the GPU index."""
    name = f"embed.gpu{gpu_id}"
    wlogger = logging.getLogger(name)
    if not wlogger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FMT))
        wlogger.addHandler(handler)
        wlogger.setLevel(logging.INFO)
        wlogger.propagate = False
    # Inject gpu into every record via a filter
    for filt in list(wlogger.filters):
        wlogger.removeFilter(filt)

    class _GPUFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.gpu = gpu_id  # type: ignore[attr-defined]
            return True

    wlogger.addFilter(_GPUFilter())
    return wlogger


# ---------------------------------------------------------------------------
# DATASET_CONFIGS – imported from dataset_configs.py (single source of truth)
# ---------------------------------------------------------------------------

from dataset_configs import DATASET_CONFIGS  # noqa: E402
from dataset_preprocess import (  # noqa: E402
    flatten_messages,
    extract_text_from_record,
    count_jsonl_rows_fast,
    estimate_jsonl_rows,
    count_parquet_rows,
    stream_jsonl,
    stream_parquet,
    resolve_glob,
)




# ---------------------------------------------------------------------------
# Work discovery
# ---------------------------------------------------------------------------


def discover_work_units(
    datasets_dir: str,
    embeddings_dir: str,
    dataset_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Walk DATASET_CONFIGS, resolve files on disk, and return work-unit dicts.

    Each work unit represents a single source file (or group of files for a
    glob pattern) under one sub-path label.
    """
    work_units: List[Dict[str, Any]] = []

    for ds_key, ds_cfg in DATASET_CONFIGS.items():
        # Apply dataset filter if provided
        if dataset_filter:
            if not any(f.lower() in ds_key.lower() for f in dataset_filter):
                continue

        hf_name = ds_cfg["hf_name"]
        local_dir_name = hf_name.replace("/", "_")
        base_dir = os.path.join(datasets_dir, local_dir_name)

        if not os.path.isdir(base_dir):
            logger.info("Skipping %s – directory not found: %s", ds_key, base_dir)
            continue

        file_format = ds_cfg["format"]

        for sub_label, rel_pattern in ds_cfg.get("sub_paths", []):
            files = resolve_glob(base_dir, rel_pattern)
            if not files:
                logger.warning(
                    "  [%s/%s] No files matching %s", ds_key, sub_label, rel_pattern
                )
                continue

            # Estimate total records across all files for this sub-path
            # (fast estimate for scheduling; exact count not needed upfront)
            total_records = 0
            for fpath in files:
                if file_format == "jsonl":
                    total_records += estimate_jsonl_rows(fpath)
                elif file_format == "parquet":
                    total_records += count_parquet_rows(fpath)

            output_dir = os.path.join(embeddings_dir, ds_key, sub_label)

            work_units.append(
                {
                    "dataset_key": ds_key,
                    "sub_label": sub_label,
                    "file_paths": files,
                    "file_format": file_format,
                    "total_records": total_records,
                    "output_dir": output_dir,
                    "config": ds_cfg,
                }
            )
            logger.info(
                "  [%s/%s] %d file(s), ~%s records",
                ds_key,
                sub_label,
                len(files),
                f"{total_records:,}",
            )

    return work_units


# ---------------------------------------------------------------------------
# Model loading & embedding extraction
# ---------------------------------------------------------------------------


def load_model(
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    wlogger: Optional[logging.Logger] = None,
) -> Tuple[AutoModel, AutoTokenizer]:
    """Load the embedding model and tokenizer onto *device*."""
    log = wlogger or logger
    log.info("Loading model %s on %s (dtype=%s) ...", MODEL_ID, device, dtype)
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()

    elapsed = time.perf_counter() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    log.info(
        "Model loaded in %.1fs  (%.2fB params, %.1f GB)",
        elapsed,
        param_count,
        param_count * (2 if dtype == torch.bfloat16 else 4),
    )
    return model, tokenizer


@torch.inference_mode()
def encode_batch(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    """
    Encode a list of texts into L2-normalised float32 embeddings.

    Returns
    -------
    np.ndarray of shape ``[len(texts), EMBEDDING_DIM]`` in float32.
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    outputs = model(**encoded)
    last_hidden = outputs.last_hidden_state  # [B, seq, hidden]

    # Average pooling (masked)
    mask = encoded["attention_mask"].unsqueeze(-1).float()  # [B, seq, 1]
    pooled = (last_hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    # L2 normalise
    embeddings = F.normalize(pooled, p=2, dim=1)

    return embeddings.cpu().numpy().astype(np.float32)


def encode_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    wlogger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Encode an arbitrarily long list of texts with automatic OOM recovery.

    If a batch causes an OOM, the batch size is halved and retried.
    Returns ``[len(texts), EMBEDDING_DIM]`` float32 array.
    """
    log = wlogger or logger
    all_embeddings: List[np.ndarray] = []
    current_bs = batch_size
    idx = 0

    while idx < len(texts):
        batch = texts[idx : idx + current_bs]
        try:
            emb = encode_batch(model, tokenizer, batch, device, max_length)
            all_embeddings.append(emb)
            idx += len(batch)
            # Restore batch size if it was reduced
            if current_bs < batch_size:
                current_bs = min(current_bs * 2, batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if current_bs > 1:
                current_bs = max(1, current_bs // 2)
                log.warning(
                    "CUDA OOM at idx %d – reducing batch size to %d", idx, current_bs
                )
            else:
                # Single-sample OOM: emit zero embedding and move on
                log.error(
                    "CUDA OOM on single sample at idx %d (len=%d chars) – "
                    "inserting zero embedding",
                    idx,
                    len(texts[idx]),
                )
                all_embeddings.append(
                    np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
                )
                idx += 1

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# File / work-unit processing
# ---------------------------------------------------------------------------


def _save_chunk(
    embeddings: np.ndarray,
    chunk_idx: int,
    output_dir: str,
) -> str:
    """Save a chunk array and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"embeddings_{chunk_idx:05d}.npy")
    np.save(path, embeddings)
    return path


def _existing_chunks(output_dir: str) -> Dict[int, int]:
    """Return {chunk_idx: num_rows} for already-saved chunks."""
    existing: Dict[int, int] = {}
    if not os.path.isdir(output_dir):
        return existing
    for fname in sorted(os.listdir(output_dir)):
        if fname.startswith("embeddings_") and fname.endswith(".npy"):
            try:
                chunk_idx = int(fname.split("_")[1].split(".")[0])
                arr = np.load(os.path.join(output_dir, fname), mmap_mode="r")
                if arr.ndim == 2 and arr.shape[1] == EMBEDDING_DIM:
                    existing[chunk_idx] = arr.shape[0]
            except Exception:
                continue
    return existing


def process_work_unit(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    work_unit: Dict[str, Any],
    device: torch.device,
    max_length: int,
    batch_size: int,
    chunk_size: int,
    wlogger: logging.Logger,
) -> Dict[str, Any]:
    """
    Process one work unit: read records, extract text, compute embeddings,
    save chunked .npy files.

    Returns a result dict with timing and row-count information.
    """
    ds_key = work_unit["dataset_key"]
    sub_label = work_unit["sub_label"]
    file_paths = work_unit["file_paths"]
    file_format = work_unit["file_format"]
    output_dir = work_unit["output_dir"]
    config = work_unit["config"]
    total_records = work_unit["total_records"]

    wlogger.info(
        "Processing %s/%s  (%s records, %d file(s))",
        ds_key,
        sub_label,
        f"{total_records:,}",
        len(file_paths),
    )

    # Check for existing chunks to support resume
    existing = _existing_chunks(output_dir)
    records_already_done = sum(existing.values())
    if records_already_done > 0:
        wlogger.info(
            "  Resuming: %d chunks (%s records) already saved",
            len(existing),
            f"{records_already_done:,}",
        )

    t0 = time.perf_counter()

    # Build a single record iterator across all files for this sub-path
    def _record_iter() -> Generator[dict, None, None]:
        for fpath in file_paths:
            if file_format == "jsonl":
                yield from stream_jsonl(fpath)
            elif file_format == "parquet":
                yield from stream_parquet(fpath)

    records = _record_iter()

    chunk_idx = 0
    records_processed = 0
    chunks_saved = 0

    while True:
        # Collect texts for one output chunk
        texts: List[str] = []
        for _ in range(chunk_size):
            try:
                record = next(records)
            except StopIteration:
                break
            texts.append(extract_text_from_record(record, config))

        if not texts:
            break  # no more records

        # Skip chunks that already exist on disk
        if chunk_idx in existing and existing[chunk_idx] == len(texts):
            wlogger.info(
                "  Chunk %05d already exists (%d rows) – skipping",
                chunk_idx,
                len(texts),
            )
            records_processed += len(texts)
            chunk_idx += 1
            continue

        # Encode texts into embeddings
        embeddings = encode_texts(
            model, tokenizer, texts, device, max_length, batch_size, wlogger
        )  # [N, 4096] float32

        assert embeddings.shape == (len(texts), EMBEDDING_DIM), (
            f"Shape mismatch: expected ({len(texts)}, {EMBEDDING_DIM}), "
            f"got {embeddings.shape}"
        )

        # Save chunk
        path = _save_chunk(embeddings, chunk_idx, output_dir)
        chunks_saved += 1
        records_processed += len(texts)

        wlogger.info(
            "  Chunk %05d saved: %s  (%d rows, %.1f MB)  [%s/%s total]",
            chunk_idx,
            os.path.basename(path),
            len(texts),
            embeddings.nbytes / 1e6,
            f"{records_processed:,}",
            f"{total_records:,}",
        )

        chunk_idx += 1

        # Periodic memory cleanup
        del embeddings, texts
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0

    # Write metadata
    metadata = {
        "dataset_key": ds_key,
        "hf_name": config["hf_name"],
        "sub_label": sub_label,
        "category": config["category"],
        "total_records": records_processed,
        "num_chunks": chunk_idx,
        "chunk_size": chunk_size,
        "embedding_dim": EMBEDDING_DIM,
        "model_id": MODEL_ID,
        "max_length": max_length,
        "dtype": "float32",
        "elapsed_seconds": round(elapsed, 1),
        "records_per_second": round(records_processed / max(elapsed, 0.1), 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    wlogger.info(
        "Finished %s/%s: %s records, %d chunks, %.1fs (%.1f rec/s)",
        ds_key,
        sub_label,
        f"{records_processed:,}",
        chunk_idx,
        elapsed,
        records_processed / max(elapsed, 0.1),
    )

    return metadata


# ---------------------------------------------------------------------------
# GPU worker (runs in a spawned subprocess)
# ---------------------------------------------------------------------------


def gpu_worker(
    gpu_id: int,
    work_items: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
) -> None:
    """
    Top-level entry point for one GPU worker process.

    Loads the model on *gpu_id*, then processes each assigned work item
    sequentially.
    """
    wlogger = _get_worker_logger(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    wlogger.info(
        "Worker started – %d work item(s) assigned, device=%s",
        len(work_items),
        torch.cuda.get_device_name(device),
    )

    if not work_items:
        wlogger.info("No work items – worker exiting")
        return

    # Parse shared config
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(args_dict.get("dtype", "bfloat16"), torch.bfloat16)
    max_length = args_dict.get("max_length", DEFAULT_MAX_LENGTH)
    batch_size = args_dict.get("batch_size", DEFAULT_BATCH_SIZE)
    chunk_size = args_dict.get("chunk_size", DEFAULT_CHUNK_SIZE)

    # Load model
    try:
        model, tokenizer = load_model(device, dtype, wlogger)
    except Exception:
        wlogger.error("Failed to load model:\n%s", traceback.format_exc())
        return

    # Process work items
    for wi_idx, work_item in enumerate(work_items, 1):
        wlogger.info(
            "--- Work item %d/%d ---", wi_idx, len(work_items)
        )
        try:
            process_work_unit(
                model,
                tokenizer,
                work_item,
                device,
                max_length,
                batch_size,
                chunk_size,
                wlogger,
            )
        except Exception:
            wlogger.error(
                "Error processing %s/%s:\n%s",
                work_item["dataset_key"],
                work_item["sub_label"],
                traceback.format_exc(),
            )
        # Memory cleanup between work items
        gc.collect()
        torch.cuda.empty_cache()

    wlogger.info("Worker finished – all work items processed")


# ---------------------------------------------------------------------------
# Work distribution
# ---------------------------------------------------------------------------


def distribute_work_lpt(
    work_items: List[Dict[str, Any]],
    num_gpus: int,
) -> List[List[Dict[str, Any]]]:
    """
    Longest-Processing-Time-first distribution.

    Assigns the largest remaining work unit to the GPU with the smallest
    current total load.  This minimises the makespan (total wall-clock time).
    """
    sorted_items = sorted(
        work_items, key=lambda w: w["total_records"], reverse=True
    )
    gpu_loads = [0] * num_gpus
    assignments: List[List[Dict[str, Any]]] = [[] for _ in range(num_gpus)]

    for item in sorted_items:
        # Find GPU with smallest current load
        min_idx = int(np.argmin(gpu_loads))
        assignments[min_idx].append(item)
        gpu_loads[min_idx] += item["total_records"]

    return assignments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract embeddings from Nemotron datasets using multi-GPU inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_embedding_extraction.py
  python 02_embedding_extraction.py --datasets Nemotron-Science-v1
  python 02_embedding_extraction.py --batch-size 4 --max-length 4096
  python 02_embedding_extraction.py --dry-run
""",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Filter: only process datasets whose key contains one of these substrings.",
    )
    p.add_argument(
        "--datasets-dir",
        default=DEFAULT_DATASETS_DIR,
        help=f"Root directory of downloaded datasets (default: {DEFAULT_DATASETS_DIR}).",
    )
    p.add_argument(
        "--embeddings-dir",
        default=DEFAULT_EMBEDDINGS_DIR,
        help=f"Root directory for output embeddings (default: {DEFAULT_EMBEDDINGS_DIR}).",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help=f"Number of GPUs to use (default: all available, typically {DEFAULT_NUM_GPUS}).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size per GPU (default: {DEFAULT_BATCH_SIZE}).",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Max token length for truncation (default: {DEFAULT_MAX_LENGTH}).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Records per output .npy chunk (default: {DEFAULT_CHUNK_SIZE:,}).",
    )
    p.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model compute dtype (default: bfloat16).  Output is always float32.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show work plan without actually processing.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Determine GPU count
    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.error("No CUDA GPUs detected.  Aborting.")
        sys.exit(1)

    logger.info("=" * 72)
    logger.info("Nemotron Embedding Extraction Pipeline")
    logger.info("=" * 72)
    logger.info("  Model           : %s", MODEL_ID)
    logger.info("  Embedding dim   : %d", EMBEDDING_DIM)
    logger.info("  GPUs            : %d", num_gpus)
    logger.info("  Batch size      : %d", args.batch_size)
    logger.info("  Max length      : %d tokens", args.max_length)
    logger.info("  Chunk size      : %s records", f"{args.chunk_size:,}")
    logger.info("  Compute dtype   : %s", args.dtype)
    logger.info("  Datasets dir    : %s", args.datasets_dir)
    logger.info("  Embeddings dir  : %s", args.embeddings_dir)
    logger.info("  Dataset filter  : %s", args.datasets or "(all)")

    # ---- Discover work ----
    logger.info("-" * 72)
    logger.info("Discovering datasets and files ...")
    work_units = discover_work_units(
        args.datasets_dir,
        args.embeddings_dir,
        args.datasets,
    )

    if not work_units:
        logger.warning("No work units found.  Nothing to do.")
        sys.exit(0)

    total_records = sum(w["total_records"] for w in work_units)
    logger.info(
        "Found %d work units  (%s total records)",
        len(work_units),
        f"{total_records:,}",
    )

    # ---- Distribute work across GPUs ----
    effective_gpus = min(num_gpus, len(work_units))
    assignments = distribute_work_lpt(work_units, effective_gpus)

    logger.info("-" * 72)
    logger.info("Work distribution (LPT scheduling, %d GPUs):", effective_gpus)
    for gpu_id, items in enumerate(assignments):
        gpu_records = sum(w["total_records"] for w in items)
        labels = [f"{w['dataset_key']}/{w['sub_label']}" for w in items]
        logger.info(
            "  GPU %d: %d item(s), ~%s records  %s",
            gpu_id,
            len(items),
            f"{gpu_records:,}",
            labels,
        )

    # ---- Dry-run exit ----
    if args.dry_run:
        logger.info("-" * 72)
        logger.info("DRY RUN – no processing performed.")

        # Estimate output size
        est_bytes = total_records * EMBEDDING_DIM * 4  # float32
        logger.info(
            "Estimated output size: %.1f GB  (%s records × %d dim × 4 bytes)",
            est_bytes / 1e9,
            f"{total_records:,}",
            EMBEDDING_DIM,
        )
        sys.exit(0)

    # ---- Spawn workers ----
    logger.info("-" * 72)
    logger.info("Launching %d GPU workers ...", effective_gpus)
    t_start = time.perf_counter()

    # Shared config dict (must be pickle-friendly)
    args_dict = {
        "dtype": args.dtype,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
    }

    os.makedirs(args.embeddings_dir, exist_ok=True)

    mp.set_start_method("spawn", force=True)
    processes: List[mp.Process] = []
    for gpu_id in range(effective_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, assignments[gpu_id], args_dict),
            name=f"gpu-worker-{gpu_id}",
        )
        p.start()
        processes.append(p)
        logger.info("  Started worker GPU-%d (pid=%d)", gpu_id, p.pid)

    # Wait for all workers to finish
    exit_codes: List[int] = []
    for gpu_id, p in enumerate(processes):
        p.join()
        code = p.exitcode if p.exitcode is not None else -1
        exit_codes.append(code)
        status = "OK" if code == 0 else f"FAILED (exit={code})"
        logger.info("  GPU-%d worker finished: %s", gpu_id, status)

    elapsed = time.perf_counter() - t_start

    # ---- Summary ----
    logger.info("=" * 72)
    logger.info("Pipeline complete")
    logger.info("  Total wall-clock time : %.1f s  (%.1f min)", elapsed, elapsed / 60)
    logger.info("  Workers succeeded     : %d / %d", exit_codes.count(0), len(exit_codes))
    logger.info("  Output directory      : %s", args.embeddings_dir)

    if any(c != 0 for c in exit_codes):
        failed_gpus = [i for i, c in enumerate(exit_codes) if c != 0]
        logger.error("  Workers FAILED on GPUs: %s", failed_gpus)
        sys.exit(1)

    logger.info("=" * 72)


if __name__ == "__main__":
    main()
