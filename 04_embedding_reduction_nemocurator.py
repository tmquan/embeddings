#!/usr/bin/env python3
"""
04_embedding_reduction_nemocurator.py – Reduce NeMo Curator Embeddings to 2D/3D for Visualization

Reads embedding Parquet files from the NeMo Curator output directory, runs t-SNE (2D/3D) and
UMAP (2D/3D) via scikit-learn and umap-learn (CPU). Processes each dataset/split independently
so memory is bounded per-split, not all 13M+ points at once.

Workflow
--------
1. Discover all {dataset_key}/{sub_label}/embeddings/ under the embeddings root.
2. For each split independently:
   a. Load embeddings (pandas).
   b. Optionally subsample if split exceeds --max-points-for-reduction.
   c. Run t-SNE (2D, 3D) and UMAP (2D, 3D) on CPU.
   d. Write reduced output to {output_dir}/{dataset_key}/{sub_label}/.
   e. Free memory before the next split.
3. Resume-safe: existing output is skipped.

Output columns (ready for Plotly / Altair / etc.)
------------------------------------------------
- row_id, dataset_key, dataset_name, sub_label, category,
  text_strategy_template, text_strategy_fields (JSON string),
  tsne_2d_x, tsne_2d_y, umap_2d_x, umap_2d_y,
  tsne_3d_x, tsne_3d_y, tsne_3d_z, umap_3d_x, umap_3d_y, umap_3d_z

Usage
-----
    # Reduce all discovered splits (each independently on CPU)
    python 04_embedding_reduction_nemocurator.py

    # Cap large splits to 500k (smaller splits use all rows)
    python 04_embedding_reduction_nemocurator.py --max-points-for-reduction 500000

    # Only specific datasets
    python 04_embedding_reduction_nemocurator.py --datasets Nemotron-Science-v1 Nemotron-Math-v2

    # Dry-run: list what would be loaded
    python 04_embedding_reduction_nemocurator.py --dry-run

Requirements
------------
    pip install pandas pyarrow numpy scikit-learn umap-learn loguru tqdm
"""

from __future__ import annotations

import os

# Suppress OpenBLAS "precompiled NUM_THREADS exceeded" warning. Must be set before NumPy import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
# Dataset config – imported from dataset_configs.py (single source of truth)
# ---------------------------------------------------------------------------

from dataset_configs import DATASET_CONFIGS  # noqa: E402

DEFAULT_EMBEDDINGS_DIR = "/raid/embeddings_curator"
DEFAULT_OUTPUT_DIR = "/raid/embeddings_reduced"
# 0 = use all rows per split; >0 = subsample large splits to this many before reduction.
DEFAULT_MAX_POINTS_FOR_REDUCTION = 0
DEFAULT_TSNE_PERPLEXITY = 30
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_embedding_splits(
    embeddings_dir: str,
    dataset_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Discover all dataset_key/sub_label that have an embeddings/ directory.
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


# ---------------------------------------------------------------------------
# Loading (pandas / numpy)
# ---------------------------------------------------------------------------


def load_embeddings_and_metadata(
    split_info: Dict[str, Any],
    max_rows: Optional[int],
    random_state: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load embedding vectors from parquet files for one split via pandas.
    Returns (vectors, meta_df) as (numpy ndarray float32, pandas DataFrame).
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


# ---------------------------------------------------------------------------
# Dimensionality reduction (sklearn + umap-learn, CPU)
# ---------------------------------------------------------------------------


def run_reductions(
    embeddings: np.ndarray,
    tsne_perplexity: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    random_state: int,
) -> Dict[str, np.ndarray]:
    """
    Run t-SNE (2D, 3D) and UMAP (2D, 3D) on CPU via sklearn and umap-learn.
    Frees memory between each step.
    """
    from sklearn.manifold import TSNE
    import umap

    n, dim = embeddings.shape
    logger.info("  Reducing {:,} points (dim={}) [sklearn + umap-learn, CPU]", n, dim)
    perplexity = min(tsne_perplexity, max(1, n - 1) // 3)
    n_neighbors = min(umap_n_neighbors, n - 1) if n > 1 else 2

    X = embeddings  # already float32

    results: Dict[str, np.ndarray] = {}

    with tqdm(total=4, desc="  Reduction", unit="step") as pbar:
        # --- t-SNE 2D ---
        pbar.set_postfix_str("t-SNE 2D")
        model = TSNE(n_components=2, perplexity=perplexity,
                      random_state=random_state, max_iter=1000, init="pca", verbose=1)
        results["tsne_2d"] = model.fit_transform(X).astype(np.float32)
        del model; gc.collect()
        pbar.update(1)

        # --- t-SNE 3D ---
        pbar.set_postfix_str("t-SNE 3D")
        model = TSNE(n_components=3, perplexity=perplexity,
                      random_state=random_state, max_iter=1000, init="pca", verbose=1)
        results["tsne_3d"] = model.fit_transform(X).astype(np.float32)
        del model; gc.collect()
        pbar.update(1)

        # --- UMAP 2D ---
        pbar.set_postfix_str("UMAP 2D")
        model = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                           min_dist=umap_min_dist, random_state=random_state,
                           metric="cosine", verbose=True)
        results["umap_2d"] = model.fit_transform(X).astype(np.float32)
        del model; gc.collect()
        pbar.update(1)

        # --- UMAP 3D ---
        pbar.set_postfix_str("UMAP 3D")
        model = umap.UMAP(n_components=3, n_neighbors=n_neighbors,
                           min_dist=umap_min_dist, random_state=random_state,
                           metric="cosine", verbose=True)
        results["umap_3d"] = model.fit_transform(X).astype(np.float32)
        del model; gc.collect()
        pbar.update(1)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce NeMo Curator embeddings to 2D/3D (t-SNE, UMAP) per split on CPU.",
    )
    parser.add_argument(
        "--embeddings-dir", type=str, default=DEFAULT_EMBEDDINGS_DIR,
        help=f"Root directory containing dataset_key/sub_label/embeddings/ (default: {DEFAULT_EMBEDDINGS_DIR}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for reduced Parquet/CSV output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="*", default=None,
        help="If set, only process these dataset keys (substring match).",
    )
    parser.add_argument(
        "--tsne-perplexity", type=int, default=DEFAULT_TSNE_PERPLEXITY,
        help="t-SNE perplexity (default: %(default)s).",
    )
    parser.add_argument(
        "--umap-neighbors", type=int, default=DEFAULT_UMAP_NEIGHBORS,
        help="UMAP n_neighbors (default: %(default)s).",
    )
    parser.add_argument(
        "--umap-min-dist", type=float, default=DEFAULT_UMAP_MIN_DIST,
        help="UMAP min_dist (default: %(default)s).",
    )
    parser.add_argument(
        "--random-state", type=int, default=DEFAULT_RANDOM_STATE,
        help="Random seed for subsampling and t-SNE/UMAP (default: %(default)s).",
    )
    parser.add_argument(
        "--max-points-for-reduction", type=int, default=DEFAULT_MAX_POINTS_FOR_REDUCTION,
        help="Per-split cap for t-SNE/UMAP (default: 0 = all rows). Splits with more rows are subsampled.",
    )
    parser.add_argument("--no-csv", action="store_true", help="Do not write CSV copies.")
    parser.add_argument("--dry-run", action="store_true", help="Only list discovered splits and exit.")
    args = parser.parse_args()

    max_for_reduction = args.max_points_for_reduction

    logger.info("Embeddings root : {}", args.embeddings_dir)
    logger.info("Output dir      : {}", args.output_dir)
    logger.info("Max points/split for reduction : {}", max_for_reduction or "no cap (full split)")
    logger.info("Discovering splits ...")
    splits = discover_embedding_splits(args.embeddings_dir, args.datasets)
    if not splits:
        logger.error("No embedding splits found. Check --embeddings-dir and --datasets.")
        sys.exit(1)

    row_counts = get_split_row_counts(splits)
    total_available = sum(row_counts)
    logger.info("Found {} split(s), {:,} total rows", len(splits), total_available)

    if args.dry_run:
        for s, rc in zip(splits, row_counts):
            use = min(rc, max_for_reduction) if max_for_reduction > 0 else rc
            logger.info("  {}/{} -> {} files, {:,} rows (reduce {:,})",
                        s["dataset_key"], s["sub_label"], len(s["parquet_files"]), rc, use)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.perf_counter()

    # Process each split independently: load -> (subsample) -> reduce -> write -> free
    for split_idx, (s, rc) in enumerate(zip(splits, row_counts)):
        ds_key = s["dataset_key"]
        sub_label = s["sub_label"]
        use_rows = min(rc, max_for_reduction) if max_for_reduction > 0 else rc

        # Resume-safe: skip if output already exists
        split_dir = os.path.join(args.output_dir, ds_key, sub_label)
        out_parquet = os.path.join(split_dir, "reduced_2d_3d.parquet")
        if os.path.isfile(out_parquet):
            logger.info("[{}/{}] {}/{} already exists, skipping", split_idx + 1, len(splits), ds_key, sub_label)
            continue

        logger.info(
            "=== [{}/{}] {}/{} ({:,} rows, reduce {:,}) ===",
            split_idx + 1, len(splits), ds_key, sub_label, rc, use_rows,
        )

        # Load
        try:
            quota = use_rows if max_for_reduction > 0 and rc > max_for_reduction else None
            X, meta_df = load_embeddings_and_metadata(s, quota, args.random_state)
        except Exception as e:
            logger.exception("Failed to load {}/{}: {}", ds_key, sub_label, e)
            continue

        n = len(meta_df)
        meta_df["row_id"] = np.arange(n, dtype=np.int64)
        logger.info("  Loaded {:,} points", n)

        if n < 5:
            logger.warning("  Too few points ({:,}), skipping reduction", n)
            del X, meta_df; gc.collect()
            continue

        # Reduce
        t_split = time.perf_counter()
        coords = run_reductions(
            X,
            tsne_perplexity=args.tsne_perplexity,
            umap_n_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            random_state=args.random_state,
        )
        del X; gc.collect()
        dt = time.perf_counter() - t_split
        logger.info("  Reduction done in {:.1f}s ({:.1f} min)", dt, dt / 60)

        # Build output DataFrame
        out_df = meta_df.copy()
        del meta_df
        for key, names in [
            ("tsne_2d", ["tsne_2d_x", "tsne_2d_y"]),
            ("umap_2d", ["umap_2d_x", "umap_2d_y"]),
            ("tsne_3d", ["tsne_3d_x", "tsne_3d_y", "tsne_3d_z"]),
            ("umap_3d", ["umap_3d_x", "umap_3d_y", "umap_3d_z"]),
        ]:
            arr = coords[key]
            for col_idx, col_name in enumerate(names):
                out_df[col_name] = arr[:, col_idx]
        del coords; gc.collect()

        # Write
        os.makedirs(split_dir, exist_ok=True)
        out_df.to_parquet(out_parquet, index=False)
        logger.info("  Wrote {} ({:,} rows)", out_parquet, len(out_df))

        if not args.no_csv:
            csv_path = os.path.join(split_dir, "reduced_2d_3d.csv")
            out_df.to_csv(csv_path, index=False)

        # Per-split metadata
        meta_path = os.path.join(split_dir, "metadata.json")
        meta_out = {
            "dataset_key": ds_key,
            "sub_label": sub_label,
            "num_rows": len(out_df),
            "total_available": rc,
            "columns_2d_3d": [
                "tsne_2d_x", "tsne_2d_y", "umap_2d_x", "umap_2d_y",
                "tsne_3d_x", "tsne_3d_y", "tsne_3d_z",
                "umap_3d_x", "umap_3d_y", "umap_3d_z",
            ],
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta_out, fh, indent=2)

        del out_df; gc.collect()

    elapsed = time.perf_counter() - t_start
    logger.info("All done in {:.1f}s ({:.1f} min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
