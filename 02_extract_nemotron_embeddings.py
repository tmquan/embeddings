#!/usr/bin/env python3
"""
Extract embeddings from Nemotron datasets using NeMo Curator pipeline architecture.
Uses nvidia/llama-embed-nemotron-8b model with distributed multi-GPU processing.

Based on NVIDIA NeMo Curator: https://github.com/NVIDIA-NeMo/Curator
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from functools import partial

import torch
import numpy as np
from tqdm import tqdm

# NeMo Curator imports
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, get_num_workers
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import ray

# HuggingFace for datasets and embeddings
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModel

# Local imports
from config import (
    DATASET_CONFIGS,
    DEFAULT_DATASETS_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    MODEL_ID_DEFAULT,
    MODEL_EMBED_SIZE,
    MODEL_MAX_TOKENS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DTYPE,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class NemotronEmbeddingExtractor:
    """
    Embedding extractor using NeMo Curator pipeline patterns.
    Distributed across multiple GPUs using Ray.
    Each instance runs on a single GPU (Ray assigns GPU automatically).
    """
    
    def __init__(
        self,
        model_id: str = MODEL_ID_DEFAULT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_CHUNK_SIZE,
        dtype: str = DEFAULT_DTYPE,
        use_flash_attention: bool = True
    ):
        """
        Initialize the embedding extractor.
        Ray automatically assigns this actor to a GPU.
        
        Args:
            model_id: HuggingFace model ID
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            dtype: Data type (bfloat16/float16/float32)
            use_flash_attention: Whether to use Flash Attention 2
        """
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        
        # Ray automatically sets CUDA_VISIBLE_DEVICES for this actor
        # Get the GPU ID that Ray assigned
        self.device = "cuda"  # Will use the GPU Ray assigned via CUDA_VISIBLE_DEVICES
        self.gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
        
        logger.info(f"Initializing Nemotron Embedding Extractor")
        logger.info(f"  Ray assigned GPU: {self.gpu_id}")
        logger.info(f"  Model: {model_id}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dtype: {dtype}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Max length: {max_length}")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model and tokenizer."""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Loading model...")
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        # Set dtype (use 'dtype' instead of deprecated 'torch_dtype')
        if self.dtype == "bfloat16":
            model_kwargs["dtype"] = torch.bfloat16
        elif self.dtype == "float16":
            model_kwargs["dtype"] = torch.float16
        
        # Try Flash Attention 2 first, fallback to sdpa, then eager
        if self.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Attempting to use Flash Attention 2...")
                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    **model_kwargs
                )
                logger.info("✓ Using Flash Attention 2")
            except (ImportError, Exception) as e:
                logger.warning(f"Flash Attention 2 not available: {e}")
                logger.info("Falling back to SDPA attention...")
                model_kwargs["attn_implementation"] = "sdpa"
                try:
                    self.model = AutoModel.from_pretrained(
                        self.model_id,
                        **model_kwargs
                    )
                    logger.info("✓ Using SDPA attention")
                except Exception as e2:
                    logger.warning(f"SDPA not available: {e2}")
                    logger.info("Falling back to eager attention...")
                    model_kwargs["attn_implementation"] = "eager"
                    self.model = AutoModel.from_pretrained(
                        self.model_id,
                        **model_kwargs
                    )
                    logger.info("✓ Using eager attention")
        else:
            # Use default (eager) attention
            model_kwargs["attn_implementation"] = "eager"
            self.model = AutoModel.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            logger.info("✓ Using eager attention")
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def extract_text_from_example(self, example: Dict[str, Any], dataset_name: str) -> str:
        """
        Extract text from a dataset example using dataset-specific logic.
        
        Args:
            example: Dataset example
            dataset_name: Name of the dataset
        
        Returns:
            Extracted text string
        """
        # Normalize dataset name
        dataset_key = dataset_name.replace("/", "_").replace("-", "_").lower()
        
        # For conversational formats with messages
        if 'messages' in example and example['messages']:
            messages = example['messages']
            if isinstance(messages, list):
                text_parts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        
                        # Include reasoning_content if available
                        if msg.get('reasoning_content'):
                            content = f"[reasoning] {msg['reasoning_content']}\n{content}"
                        
                        text_parts.append(f"{role}: {content}")
                return '\n'.join(text_parts)
        
        # For Llama-Nemotron format with input/output
        if 'output' in example and example['output']:
            parts = []
            if example.get('system_prompt'):
                parts.append(f"System: {example['system_prompt']}")
            if example.get('input'):
                # Input is usually a list of messages
                input_val = example['input']
                if isinstance(input_val, list):
                    for msg in input_val:
                        if isinstance(msg, dict):
                            parts.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
                else:
                    parts.append(f"Input: {input_val}")
            parts.append(f"Output: {example['output']}")
            return '\n'.join(parts)
        
        # For Math-Proofs format
        if 'problem' in example and example['problem']:
            parts = []
            parts.append(f"Problem: {example['problem']}")
            if example.get('formal_statement'):
                parts.append(f"Formal Statement: {example['formal_statement']}")
            if example.get('lean_header'):
                parts.append(f"Lean Header: {example['lean_header']}")
            return '\n'.join(parts)
        
        # Fallback: concatenate all text fields
        text_fields = ['text', 'content', 'prompt', 'response']
        for field in text_fields:
            if field in example and example[field]:
                return str(example[field])
        
        # Last resort: stringify the example
        return str(example)
    
    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            Array of embeddings [batch_size, embed_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embeddings
        outputs = self.model(**encoded)
        
        # Use mean pooling over sequence length
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def process_chunk(
        self,
        examples: List[Dict[str, Any]],
        dataset_name: str
    ) -> np.ndarray:
        """
        Process a chunk of examples and return embeddings.
        This method will be called remotely by Ray.
        
        Args:
            examples: List of dataset examples
            dataset_name: Name of the dataset
        
        Returns:
            Embeddings array
        """
        # Extract texts
        texts = [
            self.extract_text_from_example(example, dataset_name)
            for example in examples
        ]
        
        # Process in batches
        chunk_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.embed_texts(batch_texts)
            chunk_embeddings.append(batch_embeddings)
        
        # Concatenate chunk embeddings
        return np.vstack(chunk_embeddings)


def process_split_distributed(
    dataset_name: str,
    split_name: str,
    dataset_path: Path,
    output_dir: Path,
    extractors: List,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    num_gpus: int = 8
) -> Dict[str, Any]:
    """
    Process a single dataset split using distributed multi-GPU processing.
    
    Args:
        dataset_name: Name of the dataset
        split_name: Name of the split
        dataset_path: Path to dataset directory
        output_dir: Output directory for embeddings
        extractors: List of Ray actor references (one per GPU)
        chunk_size: Number of samples to process per chunk
        num_gpus: Number of GPUs to use
    
    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Processing {dataset_name} / {split_name} on {num_gpus} GPUs")
    
    # Load split
    split_path = dataset_path / split_name
    dataset = load_from_disk(str(split_path))
    
    num_samples = len(dataset)
    logger.info(f"  Total samples: {num_samples:,}")
    
    # Create output directory
    split_output_dir = output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data into chunks
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    logger.info(f"  Processing in {num_chunks} chunks of {chunk_size} across {num_gpus} GPUs")
    
    # Distribute chunks across GPUs
    chunk_futures = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_samples)
        
        # Get chunk data
        chunk_data = [dataset[i] for i in range(start_idx, end_idx)]
        
        # Assign to GPU in round-robin fashion
        gpu_idx = chunk_idx % num_gpus
        extractor = extractors[gpu_idx]
        
        # Submit task to GPU
        future = extractor.process_chunk.remote(chunk_data, dataset_name)
        chunk_futures.append((chunk_idx, future))
    
    # Collect results with progress bar
    logger.info(f"  Waiting for {len(chunk_futures)} chunk(s) to complete...")
    
    completed = 0
    with tqdm(total=len(chunk_futures), desc=f"  {split_name}") as pbar:
        for chunk_idx, future in chunk_futures:
            try:
                chunk_embeddings = ray.get(future)
                
                # Save chunk
                chunk_file = split_output_dir / f"embeddings_chunk_{chunk_idx:04d}.npy"
                np.save(chunk_file, chunk_embeddings)
                
                completed += 1
                pbar.update(1)
            except Exception as e:
                logger.error(f"  ✗ Error processing chunk {chunk_idx}: {e}")
    
    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "num_samples": num_samples,
        "num_chunks": num_chunks,
        "chunks_completed": completed,
        "chunk_size": chunk_size,
        "embedding_dim": MODEL_EMBED_SIZE,
        "model_id": MODEL_ID_DEFAULT,
        "num_gpus": num_gpus,
    }
    
    metadata_file = split_output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"  ✓ Processed {completed}/{num_chunks} chunks ({completed * chunk_size:,} samples)")
    logger.info(f"  ✓ Saved to {split_output_dir}")
    
    return metadata


def create_dask_dataset_from_hf(hf_dataset, text_column: str = 'text') -> DocumentDataset:
    """
    Convert HuggingFace dataset to NeMo Curator DocumentDataset.
    
    Args:
        hf_dataset: HuggingFace dataset
        text_column: Name of the text column
    
    Returns:
        DocumentDataset for NeMo Curator pipeline
    """
    # Convert to pandas DataFrame
    df = hf_dataset.to_pandas()
    
    # Convert to Dask DataFrame
    dask_df = dd.from_pandas(df, npartitions=max(1, len(df) // 10000))
    
    # Create DocumentDataset
    return DocumentDataset(dask_df)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from Nemotron datasets using NeMo Curator"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to process (default: all)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Specific splits to process (default: all)"
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID_DEFAULT,
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of samples per chunk"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype"
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable Flash Attention 2"
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path(DEFAULT_DATASETS_DIR),
        help="Directory containing datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_EMBEDDINGS_DIR),
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        help="Number of GPUs to use (default: all available)"
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Ray cluster address (default: start local cluster)"
    )
    
    args = parser.parse_args()
    
    # Banner
    logger.info("="*80)
    logger.info("NeMo Curator Multi-GPU Embedding Extraction Pipeline")
    logger.info("="*80)
    logger.info(f"Datasets directory: {args.datasets_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of GPUs: {args.num_gpus}")
    logger.info(f"Batch size per GPU: {args.batch_size}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Dtype: {args.dtype}")
    logger.info("")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Ray
    if args.ray_address:
        logger.info(f"Connecting to Ray cluster at {args.ray_address}")
        ray.init(address=args.ray_address)
    else:
        logger.info(f"Starting local Ray cluster with {args.num_gpus} GPUs")
        ray.init(num_gpus=args.num_gpus)
    
    logger.info(f"Ray cluster resources: {ray.available_resources()}")
    logger.info("")
    
    # Create extractors (one per GPU)
    logger.info(f"Initializing {args.num_gpus} embedding extractors (one per GPU)...")
    extractors = []
    
    for i in range(args.num_gpus):
        extractor = NemotronEmbeddingExtractor.remote(
            model_id=args.model_id,
            batch_size=args.batch_size,
            max_length=MODEL_MAX_TOKENS,
            dtype=args.dtype,
            use_flash_attention=not args.no_flash_attention
        )
        extractors.append(extractor)
    
    # Wait for all extractors to initialize (this loads models on each GPU)
    logger.info("Waiting for all extractors to load models...")
    init_results = ray.get([e.__ray_ready__.remote() for e in extractors])
    logger.info(f"✓ All {len(extractors)} extractors ready!")
    logger.info("")
    
    # Get datasets to process
    datasets_to_process = []
    for key, config in DATASET_CONFIGS.items():
        dataset_name = config["hf_name"]
        
        # Filter by dataset name if specified
        if args.datasets and dataset_name not in args.datasets:
            continue
        
        dataset_path = args.datasets_dir / dataset_name.replace("/", "_")
        
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_name}")
            continue
        
        datasets_to_process.append((dataset_name, dataset_path, config))
    
    logger.info(f"Processing {len(datasets_to_process)} datasets\n")
    
    # Process each dataset
    all_results = []
    
    for i, (dataset_name, dataset_path, config) in enumerate(datasets_to_process, 1):
        logger.info(f"\n[{i}/{len(datasets_to_process)}] {dataset_name}")
        logger.info("-" * 80)
        
        try:
            # Find splits
            splits = [d.name for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            
            # Filter splits if specified
            if args.splits:
                splits = [s for s in splits if s in args.splits]
            
            if not splits:
                logger.warning(f"  No splits found for {dataset_name}")
                continue
            
            logger.info(f"  Found {len(splits)} splits: {splits}")
            
            # Create dataset output directory
            dataset_output_dir = args.output_dir / config["subdir"]
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each split
            split_results = []
            for split_name in splits:
                try:
                    result = process_split_distributed(
                        dataset_name=dataset_name,
                        split_name=split_name,
                        dataset_path=dataset_path,
                        output_dir=dataset_output_dir,
                        extractors=extractors,
                        chunk_size=args.chunk_size,
                        num_gpus=args.num_gpus
                    )
                    split_results.append(result)
                except Exception as e:
                    logger.error(f"  ✗ Error processing split {split_name}: {e}", exc_info=True)
            
            # Save dataset summary
            dataset_result = {
                "dataset_name": dataset_name,
                "config_key": key,
                "subdir": config["subdir"],
                "num_splits": len(split_results),
                "splits": split_results,
                "output_dir": str(dataset_output_dir)
            }
            all_results.append(dataset_result)
            
        except Exception as e:
            logger.error(f"  ✗ Error processing dataset: {e}")
    
    # Save overall summary
    summary_file = args.output_dir / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "model_id": args.model_id,
            "embedding_dim": MODEL_EMBED_SIZE,
            "num_gpus": args.num_gpus,
            "num_datasets": len(all_results),
            "datasets": all_results
        }, f, indent=2)
    
    # Shutdown Ray
    ray.shutdown()
    
    logger.info("\n" + "="*80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Processed {len(all_results)} datasets across {args.num_gpus} GPUs")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

