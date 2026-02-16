#!/usr/bin/env python3
"""
02_embedding_extraction_curator.py – NeMo Curator-based Nemotron Dataset Embedding Extraction

Extracts dense embeddings from Nemotron datasets using nvidia/llama-embed-nemotron-8b
via NeMo Curator's Pipeline with Ray executor for distributed GPU processing.
Produces Parquet files with embedding vectors.

Workflow
--------
1. Discovers all dataset files from DATASET_CONFIGS.
2. Pre-processes each dataset sub-path into standardised Parquet files with a "text" column.
3. Builds a NeMo Curator Pipeline (ParquetReader → EmbeddingCreatorStage → ParquetWriter)
   and runs it through a Ray executor, automatically distributing work across GPUs.
4. Outputs Parquet files containing the generated embeddings.
5. Resume-safe: existing output directories with embeddings are skipped automatically.

Output structure
----------------
    /raid/embeddings_curator/
    └── {dataset_key}/                      # e.g. Nemotron-Science-v1
        └── {sub_label}/                    # e.g. MCQ
            ├── preprocessed/               # Intermediate Parquet with "text" column
            │   ├── part_00000.parquet
            │   └── part_00001.parquet
            ├── embeddings/                 # Output Parquet with "embeddings" column
            │   ├── part_00000.parquet
            │   └── part_00001.parquet
            └── metadata.json

Usage
-----
    # Process all available datasets with defaults
    python 02_embedding_extraction_curator.py

    # Process specific datasets
    python 02_embedding_extraction_curator.py --datasets Nemotron-Science-v1 Nemotron-Math-v2

    # Custom batch / sequence settings
    python 02_embedding_extraction_curator.py --batch-size 512 --max-seq-length 4096

    # Use a different executor
    python 02_embedding_extraction_curator.py --executor ray_data

    # Dry-run: show work plan without processing
    python 02_embedding_extraction_curator.py --dry-run

Requirements
------------
    pip install nemo-curator[ray]
    # torch, transformers, numpy, pyarrow, pandas
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter

# NOTE: NeMo Curator v1.0 hard-codes from_pretrained(...) without
# trust_remote_code=True and loads in float32 instead of bfloat16.
# nvidia/llama-embed-nemotron-8b ships custom modelling code that requires
# trust_remote_code, and needs bfloat16 + eager attention to fit in GPU memory.
# We patched the installed source files directly so that Ray workers also
# pick up the fixes:
#   - nemo_curator/stages/text/embedders/base.py  (EmbeddingModelStage.setup)
#     → trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
#   - nemo_curator/stages/text/models/tokenizer.py (TokenizerStage._setup, load_cfg)
#     → trust_remote_code=True

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "nvidia/llama-embed-nemotron-8b"
EMBEDDING_DIM = 4096

DEFAULT_DATASETS_DIR = "/raid/datasets"
DEFAULT_EMBEDDINGS_DIR = "/raid/embeddings_curator"
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_BATCH_SIZE = 8      # model inference batch size (must fit in GPU memory with max_seq_length)
DEFAULT_NUM_GPUS = 8
DEFAULT_CHUNK_SIZE = 10_000   # records per preprocessed Parquet partition
                             # (keep small so Ray Data has enough blocks to utilise all GPUs)
DEFAULT_FILES_PER_PARTITION = 1

# Rough tokenization estimate (for progress reporting only)
CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# DATASET_CONFIGS – imported from dataset_configs.py (single source of truth)
# ---------------------------------------------------------------------------

from dataset_configs import DATASET_CONFIGS  # noqa: E402
from dataset_preprocess import (  # noqa: E402
    flatten_messages,
    extract_text_from_record,
    stream_jsonl,
    stream_parquet,
    resolve_glob,
    estimate_record_count,
)




# ---------------------------------------------------------------------------
# Pre-processing: Convert source data to Parquet with "text" column
# ---------------------------------------------------------------------------


def preprocess_to_parquet(
    file_paths: List[str],
    file_format: str,
    config: dict,
    output_dir: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> List[str]:
    """
    Convert source JSONL / Parquet files into standardised Parquet files
    containing a single ``text`` column ready for NeMo Curator ingestion.

    Each output partition contains up to *chunk_size* rows.

    Returns a list of output Parquet file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check for existing preprocessed files (matches both old and new naming)
    existing_files = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("part_") and f.endswith(".parquet") and not f.startswith(".tmp_")
    )
    if existing_files:
        paths = [os.path.join(output_dir, f) for f in existing_files]
        total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
        logger.info(
            "  Pre-processed data found: {} file(s), {:,} rows – skipping",
            len(paths),
            total_rows,
        )
        return paths

    # Build a single record iterator across all source files
    def _record_iter() -> Generator[dict, None, None]:
        for fpath in file_paths:
            if file_format == "jsonl":
                yield from stream_jsonl(fpath)
            elif file_format == "parquet":
                yield from stream_parquet(fpath)

    # Pass 1: write partitions with temporary names (total unknown yet)
    tmp_files: List[str] = []
    part_idx = 0
    texts: List[str] = []

    for record in _record_iter():
        text = extract_text_from_record(record, config)
        if text.strip():
            texts.append(text)

        if len(texts) >= chunk_size:
            tmp_path = os.path.join(output_dir, f".tmp_part_{part_idx:06d}.parquet")
            table = pa.table({"text": texts})
            pq.write_table(table, tmp_path)
            tmp_files.append(tmp_path)
            texts = []
            part_idx += 1

    # Flush remaining texts
    if texts:
        tmp_path = os.path.join(output_dir, f".tmp_part_{part_idx:06d}.parquet")
        table = pa.table({"text": texts})
        pq.write_table(table, tmp_path)
        tmp_files.append(tmp_path)
        part_idx += 1

    if not tmp_files:
        logger.warning("    No text extracted – 0 output files")
        return []

    # Pass 2: rename with total count  (part_00066_of_00350.parquet)
    total_parts = len(tmp_files)
    output_files: List[str] = []
    for idx, tmp_path in enumerate(tmp_files):
        final_name = f"part_{idx:06d}_of_{total_parts:06d}.parquet"
        final_path = os.path.join(output_dir, final_name)
        os.rename(tmp_path, final_path)
        output_files.append(final_path)

        rows = pq.ParquetFile(final_path).metadata.num_rows
        logger.info(
            "    Pre-processed partition {}: {:,} rows",
            final_name,
            rows,
        )

    return output_files


# ---------------------------------------------------------------------------
# Executor setup
# ---------------------------------------------------------------------------


def setup_executor(executor_type: str, num_gpus: int = DEFAULT_NUM_GPUS):
    """
    Initialise and return a NeMo Curator executor.

    Supported types: ``ray_data``.
    """
    if executor_type == "ray_data":
        import ray
        from nemo_curator.backends.experimental.ray_data.executor import RayDataExecutor

        if not ray.is_initialized():
            ray.init(
                num_gpus=num_gpus,
                log_to_driver=True,
                ignore_reinit_error=True,
            )
            logger.info("Ray initialised with {} GPUs", num_gpus)

        # Suppress noisy memory-per-task warnings (advisory only)
        ctx = ray.data.DataContext.get_current()
        ctx.issue_detectors_config.high_memory_detector_config.detection_time_interval_s = -1

        return RayDataExecutor()
    else:
        raise ValueError(
            f"Unsupported executor type: {executor_type!r}. "
            "Supported: 'ray_data'"
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

    Each work unit represents a single sub-path label (one or more source files).
    """
    work_units: List[Dict[str, Any]] = []

    for ds_key, ds_cfg in DATASET_CONFIGS.items():
        if dataset_filter:
            if not any(f.lower() in ds_key.lower() for f in dataset_filter):
                continue

        hf_name = ds_cfg["hf_name"]
        local_dir_name = hf_name.replace("/", "_")
        base_dir = os.path.join(datasets_dir, local_dir_name)

        if not os.path.isdir(base_dir):
            logger.info("Skipping {} – directory not found: {}", ds_key, base_dir)
            continue

        file_format = ds_cfg["format"]

        for sub_label, rel_pattern in ds_cfg.get("sub_paths", []):
            files = resolve_glob(base_dir, rel_pattern)
            if not files:
                logger.warning(
                    "  [{}/{}] No files matching {}", ds_key, sub_label, rel_pattern
                )
                continue

            total_records = sum(
                estimate_record_count(fpath, file_format) for fpath in files
            )

            preprocess_dir = os.path.join(
                embeddings_dir, ds_key, sub_label, "preprocessed"
            )
            output_dir = os.path.join(
                embeddings_dir, ds_key, sub_label, "embeddings"
            )

            work_units.append(
                {
                    "dataset_key": ds_key,
                    "sub_label": sub_label,
                    "file_paths": files,
                    "file_format": file_format,
                    "total_records": total_records,
                    "preprocess_dir": preprocess_dir,
                    "output_dir": output_dir,
                    "config": ds_cfg,
                }
            )
            logger.info(
                "  [{}/{}] {} file(s), ~{:,} records",
                ds_key,
                sub_label,
                len(files),
                total_records,
            )

    return work_units


# ---------------------------------------------------------------------------
# Embedding generation using NeMo Curator Pipeline
# ---------------------------------------------------------------------------


def _rename_output_files(output_dir: str, input_files: Optional[List[str]] = None) -> None:
    """Rename hash-based Parquet files to indexed names matching preprocessed input order.

    NeMo Curator's ParquetWriter produces filenames like ``1abfc2e27f7c.parquet``
    (deterministic SHA256 hashes derived from source file paths).  This renames
    them to ``part_00000_of_00018.parquet`` so that each embedding file matches
    the index of its corresponding preprocessed input file.

    Parameters
    ----------
    output_dir : str
        Directory containing the hash-named Parquet files.
    input_files : list[str] or None
        Ordered list of preprocessed input file paths.  When provided, the
        function computes the expected hash for each input and maps output
        files back to their original index.  When ``None``, falls back to
        sorting by row count (largest first, partial partition last).
    """
    import hashlib

    hash_files = [
        f for f in os.listdir(output_dir)
        if f.endswith(".parquet") and not f.startswith("part_") and not f.startswith(".")
    ]
    if not hash_files:
        return

    total = len(hash_files)
    hash_set = set(hash_files)

    # --- Strategy 1: deterministic hash mapping (matches input order) ---
    if input_files and len(input_files) == total:
        def _expected_hash(fpath: str, idx: int) -> str:
            """Reproduce NeMo Curator's deterministic output filename."""
            combined = f"{fpath}|file_group_{idx}_processed"
            return hashlib.sha256(combined.encode()).hexdigest()[:12]

        mapping: Dict[str, int] = {}  # hash_filename → input_index
        for i, fpath in enumerate(input_files):
            expected = f"{_expected_hash(fpath, i)}.parquet"
            if expected in hash_set:
                mapping[expected] = i

        if len(mapping) == total:
            # Use temp names to avoid collisions during rename
            for old_name, idx in mapping.items():
                tmp_name = f".tmp_emb_{idx:06d}.parquet"
                os.rename(
                    os.path.join(output_dir, old_name),
                    os.path.join(output_dir, tmp_name),
                )
            for idx in range(total):
                tmp_name = f".tmp_emb_{idx:06d}.parquet"
                new_name = f"part_{idx:06d}_of_{total:06d}.parquet"
                os.rename(
                    os.path.join(output_dir, tmp_name),
                    os.path.join(output_dir, new_name),
                )
            logger.info(
                "  Renamed {} output file(s) → part_XXXXXX_of_{:06d}.parquet (matched to input order)",
                total,
                total,
            )
            return
        else:
            logger.warning(
                "  Hash mapping matched only {}/{} files – falling back to row-count sort",
                len(mapping),
                total,
            )

    # --- Strategy 2: fallback – sort by row count (partial partition last) ---
    def _row_count(fname: str) -> int:
        try:
            return pq.ParquetFile(os.path.join(output_dir, fname)).metadata.num_rows
        except Exception:
            return 0

    hash_files.sort(key=_row_count, reverse=True)

    for idx, old_name in enumerate(hash_files):
        tmp_name = f".tmp_emb_{idx:06d}.parquet"
        os.rename(
            os.path.join(output_dir, old_name),
            os.path.join(output_dir, tmp_name),
        )
    for idx in range(total):
        tmp_name = f".tmp_emb_{idx:06d}.parquet"
        new_name = f"part_{idx:06d}_of_{total:06d}.parquet"
        os.rename(
            os.path.join(output_dir, tmp_name),
            os.path.join(output_dir, new_name),
        )
    logger.info(
        "  Renamed {} output file(s) → part_XXXXXX_of_{:06d}.parquet (row-count sorted)",
        total,
        total,
    )


def run_embedding_pipeline(
    input_files: List[str],
    output_path: str,
    executor: Any,
    model_identifier: str = MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_seq_length: Optional[int] = DEFAULT_MAX_SEQ_LENGTH,
    files_per_partition: int = DEFAULT_FILES_PER_PARTITION,
) -> Dict[str, Any]:
    """
    Run the NeMo Curator embedding generation pipeline.

    Parameters
    ----------
    input_files : list[str]
        Paths to pre-processed Parquet files containing a ``text`` column.
    output_path : str
        Directory for output Parquet files with ``embeddings`` column.
    executor
        NeMo Curator executor (e.g. ``RayDataExecutor``).
    model_identifier : str
        HuggingFace model identifier for the embedding model.
    batch_size : int
        Batch size for model inference.
    max_seq_length : int or None
        Maximum sequence length for tokenisation.
    files_per_partition : int
        Number of input files per processing partition.

    Returns
    -------
    dict
        Result dictionary with timing metrics and task information.
    """
    os.makedirs(output_path, exist_ok=True)

    logger.info("Building NeMo Curator embedding pipeline")
    logger.info("  Input files       : {}", len(input_files))
    logger.info("  Output path       : {}", output_path)
    logger.info("  Model             : {}", model_identifier)
    logger.info("  Batch size        : {}", batch_size)
    logger.info("  Max seq length    : {}", max_seq_length)

    run_start = time.perf_counter()

    pipeline = Pipeline(
        name="embedding_generation_pipeline",
        stages=[
            ParquetReader(
                file_paths=input_files,
                files_per_partition=files_per_partition,
                fields=["text"],
                _generate_ids=False,
            ),
            EmbeddingCreatorStage(
                model_identifier=model_identifier,
                text_field="text",
                max_seq_length=max_seq_length,
                max_chars=None,
                embedding_pooling="mean_pooling",
                model_inference_batch_size=batch_size,
            ),
            ParquetWriter(
                path=output_path,
                fields=["embeddings"],
            ),
        ],
    )

    output_tasks = pipeline.run(executor)

    run_time = time.perf_counter() - run_start

    # Collect metrics
    num_docs = sum(
        task._stage_perf[-1].num_items_processed for task in output_tasks
    )
    throughput = num_docs / run_time if run_time > 0 else 0

    # Rename hash-based output files to sequential indexed names matching input order
    _rename_output_files(output_path, input_files=input_files)

    logger.success(
        "Pipeline completed in {:.2f}s – {:,} documents, {:.1f} docs/s",
        run_time,
        num_docs,
        throughput,
    )

    return {
        "is_success": True,
        "time_taken_s": run_time,
        "num_documents_processed": num_docs,
        "throughput_docs_per_sec": throughput,
        "tasks": output_tasks,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract embeddings from Nemotron datasets using NeMo Curator "
            "Pipeline with Ray executor."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_embedding_extraction_curator.py
  python 02_embedding_extraction_curator.py --datasets Nemotron-Science-v1
  python 02_embedding_extraction_curator.py --batch-size 512 --max-seq-length 4096
  python 02_embedding_extraction_curator.py --dry-run
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
        help=f"Number of GPUs to use (default: auto-detect, typically {DEFAULT_NUM_GPUS}).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Model inference batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Max sequence length for tokenisation (default: {DEFAULT_MAX_SEQ_LENGTH}).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Records per preprocessed Parquet partition (default: {DEFAULT_CHUNK_SIZE:,}).",
    )
    p.add_argument(
        "--files-per-partition",
        type=int,
        default=DEFAULT_FILES_PER_PARTITION,
        help=f"Input files per pipeline partition (default: {DEFAULT_FILES_PER_PARTITION}).",
    )
    p.add_argument(
        "--executor",
        default="ray_data",
        choices=["ray_data"],
        help="Executor backend (default: ray_data).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show work plan without actually processing.",
    )
    p.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Delete existing preprocessed Parquet files and regenerate them "
        "(useful after changing --chunk-size).",
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
        try:
            import torch

            num_gpus = torch.cuda.device_count()
        except ImportError:
            num_gpus = DEFAULT_NUM_GPUS
    if num_gpus == 0:
        logger.error("No CUDA GPUs detected. Aborting.")
        sys.exit(1)

    logger.info("=" * 72)
    logger.info("Nemotron Embedding Extraction Pipeline (NeMo Curator)")
    logger.info("=" * 72)
    logger.info("  Model             : {}", MODEL_ID)
    logger.info("  Embedding dim     : {}", EMBEDDING_DIM)
    logger.info("  GPUs              : {}", num_gpus)
    logger.info("  Batch size        : {}", args.batch_size)
    logger.info("  Max seq length    : {} tokens", args.max_seq_length)
    logger.info("  Chunk size        : {:,} records", args.chunk_size)
    logger.info("  Executor          : {}", args.executor)
    logger.info("  Datasets dir      : {}", args.datasets_dir)
    logger.info("  Embeddings dir    : {}", args.embeddings_dir)
    logger.info("  Dataset filter    : {}", args.datasets or "(all)")

    # ---- Discover work ----
    logger.info("-" * 72)
    logger.info("Discovering datasets and files ...")
    work_units = discover_work_units(
        args.datasets_dir,
        args.embeddings_dir,
        args.datasets,
    )

    if not work_units:
        logger.warning("No work units found. Nothing to do.")
        sys.exit(0)

    total_records = sum(w["total_records"] for w in work_units)
    logger.info(
        "Found {} work units ({:,} total records)",
        len(work_units),
        total_records,
    )

    # ---- Dry-run exit ----
    if args.dry_run:
        logger.info("-" * 72)
        logger.info("DRY RUN – no processing performed.")
        est_bytes = total_records * EMBEDDING_DIM * 4  # float32
        logger.info(
            "Estimated embedding output size: {:.1f} GB  ({:,} records x {} dim x 4 bytes)",
            est_bytes / 1e9,
            total_records,
            EMBEDDING_DIM,
        )
        for wu in work_units:
            logger.info(
                "  {}/{}: ~{:,} records, {} file(s)",
                wu["dataset_key"],
                wu["sub_label"],
                wu["total_records"],
                len(wu["file_paths"]),
            )
        sys.exit(0)

    # ---- Initialise executor ----
    logger.info("-" * 72)
    logger.info("Initialising {} executor with {} GPUs ...", args.executor, num_gpus)
    executor = setup_executor(args.executor, num_gpus)

    # ---- Process work units ----
    os.makedirs(args.embeddings_dir, exist_ok=True)
    t_start = time.perf_counter()

    results: List[Dict[str, Any]] = []

    for wu_idx, work_unit in enumerate(work_units, 1):
        ds_key = work_unit["dataset_key"]
        sub_label = work_unit["sub_label"]

        logger.info("-" * 72)
        logger.info(
            "Work unit {}/{}: {}/{}  (~{:,} records)",
            wu_idx,
            len(work_units),
            ds_key,
            sub_label,
            work_unit["total_records"],
        )

        # Check if embeddings already exist
        emb_dir = work_unit["output_dir"]
        if os.path.isdir(emb_dir) and any(
            f.endswith(".parquet") for f in os.listdir(emb_dir)
        ):
            # Rename any leftover hash-based files from prior runs
            pre_dir = work_unit["preprocess_dir"]
            pre_files = sorted(
                os.path.join(pre_dir, f) for f in os.listdir(pre_dir)
                if f.startswith("part_") and f.endswith(".parquet")
            ) if os.path.isdir(pre_dir) else None
            _rename_output_files(emb_dir, input_files=pre_files)
            logger.info("  Embeddings already exist in {} – skipping", emb_dir)
            results.append(
                {
                    "dataset_key": ds_key,
                    "sub_label": sub_label,
                    "status": "skipped",
                }
            )
            continue

        # Step 1: Pre-process source data into Parquet with "text" column
        preprocess_dir = work_unit["preprocess_dir"]
        if args.force_preprocess and os.path.isdir(preprocess_dir):
            logger.info("  Removing existing preprocessed dir (--force-preprocess): {}", preprocess_dir)
            shutil.rmtree(preprocess_dir)
        logger.info("  Step 1: Pre-processing source data to Parquet ...")
        try:
            input_files = preprocess_to_parquet(
                file_paths=work_unit["file_paths"],
                file_format=work_unit["file_format"],
                config=work_unit["config"],
                output_dir=preprocess_dir,
                chunk_size=args.chunk_size,
            )
        except Exception as exc:
            logger.error("  Pre-processing failed for {}/{}: {}", ds_key, sub_label, exc)
            results.append(
                {
                    "dataset_key": ds_key,
                    "sub_label": sub_label,
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        if not input_files:
            logger.warning("  No input files after pre-processing – skipping")
            results.append(
                {
                    "dataset_key": ds_key,
                    "sub_label": sub_label,
                    "status": "empty",
                }
            )
            continue

        # Step 2: Run NeMo Curator embedding pipeline
        logger.info("  Step 2: Running NeMo Curator embedding pipeline ...")
        try:
            pipeline_result = run_embedding_pipeline(
                input_files=input_files,
                output_path=emb_dir,
                executor=executor,
                model_identifier=MODEL_ID,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                files_per_partition=args.files_per_partition,
            )
            results.append(
                {
                    "dataset_key": ds_key,
                    "sub_label": sub_label,
                    "status": "success",
                    **pipeline_result,
                }
            )
        except Exception as exc:
            logger.error("  Embedding pipeline failed for {}/{}: {}", ds_key, sub_label, exc)
            results.append(
                {
                    "dataset_key": ds_key,
                    "sub_label": sub_label,
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        # Write metadata
        metadata = {
            "dataset_key": ds_key,
            "hf_name": work_unit["config"]["hf_name"],
            "sub_label": sub_label,
            "category": work_unit["config"]["category"],
            "model_id": MODEL_ID,
            "embedding_dim": EMBEDDING_DIM,
            "max_seq_length": args.max_seq_length,
            "batch_size": args.batch_size,
            "executor": args.executor,
            "num_gpus": num_gpus,
            "num_input_files": len(input_files),
            "time_taken_s": pipeline_result.get("time_taken_s"),
            "num_documents_processed": pipeline_result.get("num_documents_processed"),
            "throughput_docs_per_sec": pipeline_result.get("throughput_docs_per_sec"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_dir = os.path.join(args.embeddings_dir, ds_key, sub_label)
        os.makedirs(meta_dir, exist_ok=True)
        meta_path = os.path.join(meta_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("  Metadata saved → {}", meta_path)

    elapsed = time.perf_counter() - t_start

    # ---- Summary ----
    logger.info("=" * 72)
    logger.info("Pipeline complete")
    logger.info("  Total wall-clock time : {:.1f}s ({:.1f} min)", elapsed, elapsed / 60)
    logger.info("  Output directory      : {}", args.embeddings_dir)

    successes = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    empty = sum(1 for r in results if r["status"] == "empty")

    logger.info(
        "  Results: {} succeeded, {} skipped, {} empty, {} errors",
        successes,
        skipped,
        empty,
        errors,
    )

    if errors:
        for r in results:
            if r["status"] == "error":
                logger.error(
                    "  FAILED: {}/{} – {}",
                    r["dataset_key"],
                    r["sub_label"],
                    r.get("error", "unknown"),
                )
        sys.exit(1)

    total_docs = sum(r.get("num_documents_processed", 0) for r in results)
    total_time = sum(r.get("time_taken_s", 0) for r in results)
    logger.info(
        "  Total documents processed : {:,}",
        total_docs,
    )
    if total_time > 0:
        logger.info(
            "  Aggregate throughput      : {:.1f} docs/s",
            total_docs / total_time,
        )

    logger.info("=" * 72)


if __name__ == "__main__":
    main()
