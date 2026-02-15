#!/usr/bin/env python3
"""
04_embedding_reduction_nemocurator.py – Reduce NeMo Curator Embeddings to 2D/3D for Visualization

Reads embedding Parquet files from the NeMo Curator output directory, runs t-SNE and UMAP
via scikit-learn and umap-learn (CPU), and writes a single Parquet (and optional CSV) with
coordinates plus rich metadata columns for hover and color labeling in visualization tools.

Workflow
--------
1. Discover all {dataset_key}/{sub_label}/embeddings/ under the embeddings root.
2. Load metadata.json per split and attach dataset_key, dataset_name, sub_label,
   category, text_strategy (template + fields) for hover/color.
3. Optionally subsample per split to stay within memory/runtime limits.
4. Run t-SNE (2D and 3D) and UMAP (2D and 3D) via scikit-learn and umap-learn (CPU) on the combined matrix.
5. Write reduced output with columns: coordinates (tsne_2d_*, umap_2d_*, tsne_3d_*, umap_3d_*)
   and metadata (dataset_key, dataset_name, sub_label, category, text_strategy_*, etc.).

Output columns (ready for Plotly / Altair / etc.)
------------------------------------------------
- row_id, dataset_key, dataset_name, sub_label, category,
  text_strategy_template, text_strategy_fields (JSON string),
  tsne_2d_x, tsne_2d_y, umap_2d_x, umap_2d_y,
  tsne_3d_x, tsne_3d_y, tsne_3d_z, umap_3d_x, umap_3d_y, umap_3d_z

Usage
-----
    # Reduce all discovered embeddings with default sampling
    python 04_embedding_reduction_nemocurator.py

    # Reduce all (default); optional subsample for a lighter run
    python 04_embedding_reduction_nemocurator.py --max-points-per-split 5000 --output-dir /raid/embeddings_curator/reduced

    # Only specific datasets
    python 04_embedding_reduction_nemocurator.py --datasets Nemotron-Science-v1 Nemotron-Math-v2

    # Dry-run: list what would be loaded
    python 04_embedding_reduction_nemocurator.py --dry-run

Requirements
------------
    pip install pandas pyarrow numpy scikit-learn umap-learn loguru
"""

from __future__ import annotations

import os

# Configure OpenBLAS before NumPy/SciPy load to avoid "Bad memory unallocation" and SIGSEGV
# when running t-SNE/UMAP on large arrays (precompiled NUM_THREADS is often exceeded otherwise).
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
if "OPENBLAS_MAIN_FREE" not in os.environ:
    os.environ["OPENBLAS_MAIN_FREE"] = "1"  # free memory on main thread exit

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

# Embedding dimension from NeMo Curator output (nvidia/llama-embed-nemotron-8b)
EMBEDDING_DIM = 4096

# ---------------------------------------------------------------------------
# Dataset config (metadata for hover/color – must match 02_embedding_extraction_nemocurator)
# ---------------------------------------------------------------------------

DATASET_CONFIGS: Dict[str, dict] = {
    "Nemotron-3-Nano-RL-Training-Blend": {
        "hf_name": "nvidia/Nemotron-3-Nano-RL-Training-Blend",
        "category": "post-training",
        "text_strategy": {"fields": ["responses_create_params", "ground_truth"], "template": "rl_blend"},
    },
    "Nemotron-Science-v1": {
        "hf_name": "nvidia/Nemotron-Science-v1",
        "category": "post-training",
        "text_strategy": {"fields": ["messages"], "template": "messages_list"},
    },
    "Nemotron-Instruction-Following-Chat-v1": {
        "hf_name": "nvidia/Nemotron-Instruction-Following-Chat-v1",
        "category": "post-training",
        "text_strategy": {"fields": ["messages"], "template": "messages_list"},
    },
    "Nemotron-Math-Proofs-v1": {
        "hf_name": "nvidia/Nemotron-Math-Proofs-v1",
        "category": "post-training",
        "text_strategy": {"fields": ["problem", "formal_statement", "lean_header"], "template": "math_proof"},
    },
    "Nemotron-Agentic-v1": {
        "hf_name": "nvidia/Nemotron-Agentic-v1",
        "category": "post-training",
        "text_strategy": {"fields": ["messages", "tools"], "template": "agentic"},
    },
    "Nemotron-Competitive-Programming-v1": {
        "hf_name": "nvidia/Nemotron-Competitive-Programming-v1",
        "category": "post-training",
        "text_strategy": {"fields": ["messages"], "template": "messages_list"},
    },
    "Nemotron-Math-v2": {
        "hf_name": "nvidia/Nemotron-Math-v2",
        "category": "post-training",
        "text_strategy": {"fields": ["problem", "messages"], "template": "math_v2"},
    },
    "Nemotron-SWE-v1": {
        "hf_name": "nvidia/Nemotron-SWE-v1",
        "category": "post-training",
        "text_strategy": {"fields": ["messages", "tools"], "template": "agentic"},
    },
}

DEFAULT_EMBEDDINGS_DIR = "/raid/embeddings_curator"
DEFAULT_OUTPUT_DIR = "/raid/embeddings_reduced"
# Reduce all data by default; no per-split cap. Subsampling is for visualization only (downstream).
DEFAULT_MAX_POINTS_PER_SPLIT = 0
DEFAULT_MAX_TOTAL_POINTS = 0  # 0 = reduce all in batches
BATCH_MAX_POINTS = 16_000_000  # per-batch cap (e.g. 2TB RAM); full ~13.5M fits in one batch
# Subsample to this many points before reduction when >0; 0 = full reduction (clear buffers between steps to avoid OpenBLAS crash).
DEFAULT_MAX_POINTS_FOR_REDUCTION = 0
DEFAULT_TSNE_PERPLEXITY = 30
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_RANDOM_STATE = 42


def discover_embedding_splits(
    embeddings_dir: str,
    dataset_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Discover all dataset_key/sub_label that have an embeddings/ directory and metadata.json.
    Returns list of dicts with paths and metadata for loading.
    """
    splits: List[Dict[str, Any]] = []
    embeddings_root = Path(embeddings_dir)
    if not embeddings_root.is_dir():
        logger.warning("Embeddings root does not exist: {}", embeddings_dir)
        return splits

    for ds_key in sorted(embeddings_root.iterdir()):
        if not ds_key.is_dir():
            continue
        if dataset_filter and not any(f.lower() in ds_key.name.lower() for f in dataset_filter):
            continue
        dataset_key = ds_key.name
        config = DATASET_CONFIGS.get(dataset_key, {})
        hf_name = config.get("hf_name", dataset_key)
        category = config.get("category", "unknown")
        text_strategy = config.get("text_strategy", {})
        text_strategy_template = text_strategy.get("template", "unknown")
        text_strategy_fields = json.dumps(text_strategy.get("fields", []))

        for sub_dir in sorted(ds_key.iterdir()):
            if not sub_dir.is_dir():
                continue
            sub_label = sub_dir.name
            emb_dir = sub_dir / "embeddings"
            meta_path = sub_dir / "metadata.json"
            if not emb_dir.is_dir():
                continue
            parquet_files = sorted(emb_dir.glob("*.parquet"))
            parquet_files = [f for f in parquet_files if f.name.startswith("part_") and not f.name.startswith(".tmp")]
            if not parquet_files:
                logger.debug("  [{}/{}] No part_*.parquet in {}", dataset_key, sub_label, emb_dir)
                continue
            meta: Dict[str, Any] = {}
            if meta_path.is_file():
                try:
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                except Exception as e:
                    logger.warning("  Could not read {}: {}", meta_path, e)
            splits.append({
                "dataset_key": dataset_key,
                "dataset_name": hf_name,
                "sub_label": sub_label,
                "category": category,
                "text_strategy_template": text_strategy_template,
                "text_strategy_fields": text_strategy_fields,
                "embedding_dir": str(emb_dir),
                "parquet_files": [str(p) for p in parquet_files],
                "metadata": meta,
            })
            logger.info(
                "  [{}/{}] {} parquet file(s)",
                dataset_key,
                sub_label,
                len(parquet_files),
            )
    return splits


def get_split_row_counts(splits: List[Dict[str, Any]]) -> List[int]:
    """Return total row count per split from Parquet metadata (no data loaded)."""
    counts: List[int] = []
    for s in splits:
        n = 0
        for path in s["parquet_files"]:
            try:
                n += pq.ParquetFile(path).metadata.num_rows
            except Exception:
                pass
        counts.append(n)
    return counts


def compute_per_split_quotas(
    row_counts: List[int],
    max_total_points: Optional[int],
    max_per_split: Optional[int],
) -> List[int]:
    """
    Compute per-split row quotas so that sum(quotas) <= max_total_points.
    Each quota is at most max_per_split. No single split gets more than DEFAULT_MAX_TOTAL_POINTS
    (avoids cuDF concat row limit within one split).
    """
    if max_per_split is not None:
        capped = [min(rc, max_per_split) for rc in row_counts]
    else:
        capped = list(row_counts)
    # Cap each split so we never concat more than cuDF can handle in one split
    capped = [min(c, BATCH_MAX_POINTS) for c in capped]
    total = sum(capped)
    if max_total_points is None or total <= max_total_points:
        return capped
    # Distribute max_total_points proportionally
    scale = max_total_points / total
    quotas = [max(0, int(c * scale)) for c in capped]
    # Ensure we don't exceed max_total_points due to rounding
    while sum(quotas) > max_total_points and any(q > 0 for q in quotas):
        i = max(range(len(quotas)), key=lambda i: quotas[i])
        quotas[i] -= 1
    return quotas


def compute_batch_quotas(
    row_counts: List[int],
    batch_max: int = BATCH_MAX_POINTS,
) -> List[List[int]]:
    """
    When reducing all data, split into batches of at most batch_max so we never exceed cuDF limit.
    Returns list of per-split quota lists: [batch0_quotas, batch1_quotas, ...].
    """
    total = sum(row_counts)
    if total == 0:
        return []
    batches: List[List[int]] = []
    remaining = list(row_counts)
    while sum(remaining) > 0:
        take = compute_per_split_quotas(remaining, batch_max, None)
        # take[i] must not exceed remaining[i]
        take = [min(take[i], remaining[i]) for i in range(len(remaining))]
        if sum(take) == 0:
            break
        batches.append(take)
        remaining = [remaining[i] - take[i] for i in range(len(remaining))]
    return batches


def load_embeddings_and_metadata(
    split_info: Dict[str, Any],
    max_rows: Optional[int],
    random_state: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load embedding vectors from parquet files for one split (pandas) and build
    a metadata DataFrame. Returns (vectors, meta_df) with numpy/pandas.
    """
    dfs: List[pd.DataFrame] = []
    for path in split_info["parquet_files"]:
        df = pd.read_parquet(path, columns=["embeddings"])
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    n = len(combined)
    if max_rows is not None and n > max_rows:
        combined = combined.sample(n=max_rows, random_state=random_state)
        n = len(combined)
    vectors = np.stack(combined["embeddings"].tolist()).astype(np.float32)
    meta_df = pd.DataFrame({
        "dataset_key": [split_info["dataset_key"]] * n,
        "dataset_name": [split_info["dataset_name"]] * n,
        "sub_label": [split_info["sub_label"]] * n,
        "category": [split_info["category"]] * n,
        "text_strategy_template": [split_info["text_strategy_template"]] * n,
        "text_strategy_fields": [split_info["text_strategy_fields"]] * n,
    })
    return vectors, meta_df


def _clear_reduction_buffers() -> None:
    """Release Python objects and run GC between reduction steps to free OpenBLAS/NumPy buffers and avoid SIGSEGV."""
    gc.collect()


def run_reductions(
    embeddings: np.ndarray,
    tsne_perplexity: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    random_state: int,
) -> Dict[str, np.ndarray]:
    """
    Run t-SNE (2D, 3D) and UMAP (2D, 3D) via scikit-learn and umap-learn (CPU).
    Clears buffers after each step to reduce OpenBLAS memory pressure and avoid SIGSEGV on full-size runs.
    """
    from sklearn.manifold import TSNE
    import umap

    n, dim = embeddings.shape[0], embeddings.shape[1]
    logger.info("Running dimensionality reduction on {} points (dim={}) [sklearn + umap]", n, dim)
    perplexity = min(tsne_perplexity, max(1, n - 1) // 3)
    n_neighbors = min(umap_n_neighbors, n - 1) if n > 1 else 2

    X = embeddings.astype(np.float32)
    coords_tsne_2d = coords_tsne_3d = coords_umap_2d = coords_umap_3d = None

    with tqdm(total=4, desc="Reduction", unit="step") as pbar:
        # t-SNE 2D
        pbar.set_postfix_str("t-SNE 2D")
        tsne_2d = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
            init="pca",
            verbose=1,
        )
        coords_tsne_2d = np.ascontiguousarray(tsne_2d.fit_transform(X), dtype=np.float32)
        del tsne_2d
        _clear_reduction_buffers()
        pbar.update(1)

        # t-SNE 3D
        pbar.set_postfix_str("t-SNE 3D")
        tsne_3d = TSNE(
            n_components=3,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
            init="pca",
            verbose=1,
        )
        coords_tsne_3d = np.ascontiguousarray(tsne_3d.fit_transform(X), dtype=np.float32)
        del tsne_3d
        _clear_reduction_buffers()
        pbar.update(1)

        # UMAP 2D
        pbar.set_postfix_str("UMAP 2D")
        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
            metric="cosine",
            verbose=True,
        )
        coords_umap_2d = np.ascontiguousarray(reducer_2d.fit_transform(X), dtype=np.float32)
        del reducer_2d
        _clear_reduction_buffers()
        pbar.update(1)

        # UMAP 3D
        pbar.set_postfix_str("UMAP 3D")
        reducer_3d = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
            metric="cosine",
            verbose=True,
        )
        coords_umap_3d = np.ascontiguousarray(reducer_3d.fit_transform(X), dtype=np.float32)
        del reducer_3d
        _clear_reduction_buffers()
        pbar.update(1)

    return {
        "tsne_2d": coords_tsne_2d,
        "tsne_3d": coords_tsne_3d,
        "umap_2d": coords_umap_2d,
        "umap_3d": coords_umap_3d,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce NeMo Curator embeddings to 2D/3D (t-SNE, UMAP) for visualization.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=DEFAULT_EMBEDDINGS_DIR,
        help=f"Root directory containing dataset_key/sub_label/embeddings/ (default: {DEFAULT_EMBEDDINGS_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for reduced Parquet/CSV output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="If set, only process these dataset keys (substring match).",
    )
    parser.add_argument(
        "--max-points-per-split",
        type=int,
        default=DEFAULT_MAX_POINTS_PER_SPLIT,
        help="Cap points per split (default: 0 = no limit, reduce all). Set only to subsample for faster runs.",
    )
    parser.add_argument(
        "--max-total-points",
        type=int,
        default=DEFAULT_MAX_TOTAL_POINTS,
        help="Cap total points (default: 0 = reduce all in batches of 12M). Set to 12M for single-run cap.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=int,
        default=DEFAULT_TSNE_PERPLEXITY,
        help="t-SNE perplexity (default: %(default)s).",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=DEFAULT_UMAP_NEIGHBORS,
        help="UMAP n_neighbors (default: %(default)s).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=DEFAULT_UMAP_MIN_DIST,
        help="UMAP min_dist (default: %(default)s).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for subsampling and t-SNE/UMAP (default: %(default)s).",
    )
    parser.add_argument(
        "--max-points-for-reduction",
        type=int,
        default=DEFAULT_MAX_POINTS_FOR_REDUCTION,
        help="Subsample to at most this many points before t-SNE/UMAP (default: 0 = full reduction). Buffers are cleared between each 2D/3D step to reduce OpenBLAS pressure. Set >0 to cap for lighter runs.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write a CSV copy (only Parquet).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list discovered splits and exit.",
    )
    args = parser.parse_args()

    max_per_split = args.max_points_per_split if args.max_points_per_split > 0 else None
    max_total = args.max_total_points if args.max_total_points > 0 else None

    logger.info("Embeddings root : {}", args.embeddings_dir)
    logger.info("Output dir     : {}", args.output_dir)
    logger.info("Max points/split : {}", max_per_split or "no limit")
    logger.info("Max total points : {}", max_total or "no limit")
    logger.info("Max points for reduction : {} (0 = no cap)", args.max_points_for_reduction or "no cap")
    logger.info("Discovering splits ...")
    splits = discover_embedding_splits(args.embeddings_dir, args.datasets)
    if not splits:
        logger.error("No embedding splits found. Check --embeddings-dir and --datasets.")
        sys.exit(1)
    logger.info("Found {} split(s)", len(splits))

    if args.dry_run:
        row_counts = get_split_row_counts(splits)
        for s, rc in zip(splits, row_counts):
            logger.info("  {}/{}  → {} files, ~{:,} rows", s["dataset_key"], s["sub_label"], len(s["parquet_files"]), rc)
        return

    row_counts = get_split_row_counts(splits)
    total_available = sum(row_counts)
    # When total exceeds one-batch cap, always run in batches so we reduce exactly total_available
    if total_available > BATCH_MAX_POINTS:
        batch_quotas_list = compute_batch_quotas(row_counts, BATCH_MAX_POINTS)
        total_in_batches = sum(sum(q) for q in batch_quotas_list)
        logger.info(
            "Reduce all: {:,} points in {} batch(es) (max {:,} per batch)",
            total_in_batches,
            len(batch_quotas_list),
            BATCH_MAX_POINTS,
        )
        assert total_in_batches == total_available, "batch quotas should sum to total_available"
    else:
        # total fits in one batch; optional cap via max_total
        quotas = compute_per_split_quotas(row_counts, max_total, max_per_split)
        batch_quotas_list = [quotas]
        total_planned = sum(quotas)
        logger.info("Per-split quotas → total {:,} points", total_planned)

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.perf_counter()

    for batch_idx, quotas in enumerate(batch_quotas_list):
        total_planned = sum(quotas)
        if total_planned == 0:
            continue
        logger.info(
            "--- Batch {} / {} ({:,} points) ---",
            batch_idx + 1,
            len(batch_quotas_list),
            total_planned,
        )

        all_vectors: List[np.ndarray] = []
        all_meta: List[pd.DataFrame] = []
        splits_to_load = [(s, q) for s, q in zip(splits, quotas) if q > 0]
        for s, quota in tqdm(
            splits_to_load,
            desc="Loading splits",
            unit="split",
            leave=True,
        ):
            tqdm.write("  {}/{} ({:,} rows)".format(s["dataset_key"], s["sub_label"], quota))
            try:
                vec, meta = load_embeddings_and_metadata(s, quota, args.random_state)
                all_vectors.append(vec)
                all_meta.append(meta)
            except Exception as e:
                logger.exception("Failed to load {}/{}: {}", s["dataset_key"], s["sub_label"], e)
                continue

        if not all_vectors:
            logger.warning("Batch {} loaded no data", batch_idx)
            continue

        logger.info("Stacking vectors and building metadata ...")
        X = np.vstack(all_vectors)
        meta_df = pd.concat(all_meta, ignore_index=True)
        meta_df["row_id"] = np.arange(len(meta_df))
        meta_df["batch"] = batch_idx
        logger.info("Total points loaded: {:,}", len(meta_df))
        n_loaded = len(meta_df)
        max_for_reduction = args.max_points_for_reduction
        if max_for_reduction > 0 and n_loaded > max_for_reduction:
            rng = np.random.default_rng(args.random_state)
            ix = rng.choice(n_loaded, size=max_for_reduction, replace=False)
            ix = np.sort(ix)  # preserve rough ordering
            X = X[ix]
            meta_df = meta_df.iloc[ix].reset_index(drop=True)
            meta_df["row_id"] = np.arange(len(meta_df))
            logger.info("Subsampled to {:,} points for t-SNE/UMAP (--max-points-for-reduction)", len(meta_df))
        logger.info("Running t-SNE and UMAP (2D + 3D) on {:,} points ...", len(meta_df))

        coords = run_reductions(
            X,
            tsne_perplexity=args.tsne_perplexity,
            umap_n_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            random_state=args.random_state,
        )

        out_df = meta_df.copy()
        out_df["tsne_2d_x"] = coords["tsne_2d"][:, 0]
        out_df["tsne_2d_y"] = coords["tsne_2d"][:, 1]
        out_df["umap_2d_x"] = coords["umap_2d"][:, 0]
        out_df["umap_2d_y"] = coords["umap_2d"][:, 1]
        out_df["tsne_3d_x"] = coords["tsne_3d"][:, 0]
        out_df["tsne_3d_y"] = coords["tsne_3d"][:, 1]
        out_df["tsne_3d_z"] = coords["tsne_3d"][:, 2]
        out_df["umap_3d_x"] = coords["umap_3d"][:, 0]
        out_df["umap_3d_y"] = coords["umap_3d"][:, 1]
        out_df["umap_3d_z"] = coords["umap_3d"][:, 2]

        # Write per split to mirror embeddings_curator structure: {output_dir}/{dataset_key}/{sub_label}/
        base_name = "reduced_2d_3d_part{}".format(batch_idx) if len(batch_quotas_list) > 1 else "reduced_2d_3d"
        for (ds_key, sub_label), group in out_df.groupby(["dataset_key", "sub_label"]):
            split_dir = os.path.join(args.output_dir, ds_key, sub_label)
            os.makedirs(split_dir, exist_ok=True)
            parquet_path = os.path.join(split_dir, base_name + ".parquet")
            group.to_parquet(parquet_path, index=False)
            logger.info("  Wrote {}/{} → {}", ds_key, sub_label, parquet_path)
            if not args.no_csv:
                csv_path = os.path.join(split_dir, base_name + ".csv")
                group.to_csv(csv_path, index=False)
            # Per-split metadata (mirror embeddings_curator metadata.json)
            meta_path = os.path.join(split_dir, "metadata.json")
            meta = {
                "dataset_key": ds_key,
                "sub_label": sub_label,
                "columns_2d_3d": ["tsne_2d_x", "tsne_2d_y", "umap_2d_x", "umap_2d_y", "tsne_3d_x", "tsne_3d_y", "tsne_3d_z", "umap_3d_x", "umap_3d_y", "umap_3d_z"],
            }
            batch_entry = {"batch": batch_idx, "file": base_name + ".parquet", "num_rows": len(group)}
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                batches = existing.get("batches", [])
                batches.append(batch_entry)
                meta["batches"] = batches
                meta["total_rows"] = sum(b["num_rows"] for b in batches)
            else:
                meta["batches"] = [batch_entry]
                meta["total_rows"] = len(group)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)

    elapsed = time.perf_counter() - t_start
    logger.info("Done in {:.1f}s ({:.1f} min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
