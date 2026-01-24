#!/usr/bin/env python3
"""
Explore Nemotron datasets using NeMo Curator to collect data distributions,
number of splits, samples, columns, and identify important text-based columns
for embedding extraction.

Based on NVIDIA NeMo Curator: https://github.com/NVIDIA-NeMo/Curator
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import warnings

# HuggingFace datasets for reading saved datasets
from datasets import load_from_disk
import pyarrow as pa

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from config.py
from config import DATASET_CONFIGS, DEFAULT_DATASETS_DIR

# Output paths
SUMMARY_JSON = Path("dataset_exploration_summary.json")
DETAILED_JSON = Path("dataset_exploration_detailed.json")
EXTRACTION_FUNCTIONS_PY = Path("embedding_extraction_functions.py")


def is_text_column(column_name: str, sample_value: Any, column_type: str) -> bool:
    """
    Determine if a column contains meaningful text content for embeddings.
    
    Args:
        column_name: Name of the column
        sample_value: Sample value from the column
        column_type: PyArrow data type
    
    Returns:
        True if column should be considered for embedding extraction
    """
    # Skip metadata columns (but not 'metadata' itself for pretraining datasets)
    skip_columns = {'uuid', 'license', 'version', 'category', 'reasoning', 
                    'source', 'used_in', 'used_in_training', 'tools', 'url',
                    'user_name', 'user_url', 'sft_line_number', 'id', 'timestamp',
                    'language', 'language_score', 'quality_score'}
    
    if column_name in skip_columns:
        return False
    
    # Primary text columns for pretraining datasets
    primary_text_columns = ['text', 'content', 'document', 'passage']
    if column_name.lower() in primary_text_columns:
        return True
    
    # Check for text-like columns (including metadata for pretraining datasets)
    text_indicators = ['text', 'content', 'message', 'output', 'input', 
                       'prompt', 'response', 'statement', 'problem', 
                       'generator', 'metadata']
    
    if any(ind in column_name.lower() for ind in text_indicators):
        return True
    
    # Check if it's a string or list type
    if 'string' in column_type.lower() or 'list' in column_type.lower():
        if sample_value:
            # Check string length to filter out short metadata strings
            if isinstance(sample_value, str):
                return len(sample_value) > 20
            elif isinstance(sample_value, list) and sample_value:
                # Check if it's a list of dicts (messages format)
                if isinstance(sample_value[0], dict):
                    return 'content' in sample_value[0] or 'role' in sample_value[0]
    
    return False


def detect_conversational_format(sample_rows: List[Dict], columns: List[str]) -> Dict[str, Any]:
    """
    Detect if dataset uses conversational/message format.
    
    Returns:
        Dictionary with format information
    """
    result = {
        "is_conversational": False,
        "format_type": None,
        "message_keys": []
    }
    
    # Check for 'messages' column
    if 'messages' in columns:
        for row in sample_rows[:5]:
            if row and 'messages' in row:
                messages = row['messages']
                if isinstance(messages, list) and messages and isinstance(messages[0], dict):
                    result["is_conversational"] = True
                    result["format_type"] = "messages_list"
                    # Collect all possible keys in messages
                    all_keys = set()
                    for msg in messages:
                        all_keys.update(msg.keys())
                    result["message_keys"] = sorted(list(all_keys))
                    break
    
    # Check for 'input'/'output' conversational format
    if 'input' in columns and 'output' in columns:
        for row in sample_rows[:5]:
            if row and 'input' in row:
                input_val = row['input']
                if isinstance(input_val, list) and input_val and isinstance(input_val[0], dict):
                    if 'role' in input_val[0] and 'content' in input_val[0]:
                        result["is_conversational"] = True
                        result["format_type"] = "input_output_messages"
                        result["message_keys"] = ['role', 'content']
                        break
    
    return result


def explore_split(dataset_path: Path, split_name: str) -> Dict[str, Any]:
    """
    Explore a single dataset split using HuggingFace datasets.
    
    Args:
        dataset_path: Path to the dataset directory
        split_name: Name of the split (e.g., 'train', 'code', 'chat')
    
    Returns:
        Dictionary containing split information
    """
    split_path = dataset_path / split_name
    
    # Load HuggingFace dataset split
    hf_dataset = load_from_disk(str(split_path))
    
    # Get schema and sample rows
    num_rows = len(hf_dataset)
    columns = hf_dataset.column_names
    
    # Get sample rows (up to 10)
    sample_size = min(10, num_rows)
    sample_rows = [hf_dataset[i] for i in range(sample_size)]
    
    # Get sample rows (up to 10)
    sample_size = min(10, num_rows)
    sample_rows = [hf_dataset[i] for i in range(sample_size)]
    
    # Analyze columns
    column_analysis = {}
    text_columns = []
    
    for col_name in columns:
        # Get column type from HuggingFace features
        col_feature = hf_dataset.features[col_name]
        col_type = str(col_feature)
        sample_val = sample_rows[0].get(col_name) if sample_rows else None
        
        is_text = is_text_column(col_name, sample_val, col_type)
        
        column_analysis[col_name] = {
            "type": col_type,
            "is_text": is_text,
            "sample": str(sample_val)[:200] if sample_val is not None else None
        }
        
        if is_text:
            text_columns.append(col_name)
    
    # Detect conversational format
    conv_format = detect_conversational_format(sample_rows, columns)
    
    return {
        "split_name": split_name,
        "num_rows": num_rows,
        "columns": columns,
        "num_columns": len(columns),
        "text_columns": text_columns,
        "column_analysis": column_analysis,
        "conversational_format": conv_format,
        "sample_rows": sample_rows[:3]  # Keep 3 samples
    }


def determine_embedding_strategy(split_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine the best strategy for combining columns for embedding extraction.
    
    Returns:
        Dictionary describing the embedding strategy
    """
    conv_format = split_info["conversational_format"]
    text_columns = split_info["text_columns"]
    
    if conv_format["is_conversational"]:
        # For conversational formats, concatenate messages
        example = None
        if split_info["sample_rows"]:
            sample = split_info["sample_rows"][0]
            if 'messages' in sample and sample['messages']:
                messages = sample['messages']
                parts = []
                for msg in messages[:3]:  # Show first 3 messages
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if msg.get('reasoning_content'):
                        content = f"[reasoning] {msg['reasoning_content']}\n{content}"
                    parts.append(f"{role}: {content[:200]}")
                example = '\n'.join(parts)
        
        return {
            "strategy": "concatenate_messages",
            "description": "Concatenate all messages in conversation with role prefixes",
            "columns_to_combine": ["messages"],
            "example": example
        }
    
    elif text_columns:
        # Check if this is a simple pretraining dataset (just 'text' column)
        if len(text_columns) == 1 and text_columns[0] in ['text', 'content', 'document']:
            example = None
            if split_info["sample_rows"]:
                sample = split_info["sample_rows"][0]
                text_val = sample.get(text_columns[0], '')
                if isinstance(text_val, str):
                    example = text_val[:200]
            
            return {
                "strategy": "direct_text",
                "description": f"Direct text extraction from '{text_columns[0]}' column (pretraining format)",
                "columns_to_combine": text_columns,
                "example": example
            }
        
        # Multiple text columns - combine them
        example = None
        if split_info["sample_rows"]:
            sample = split_info["sample_rows"][0]
            parts = []
            for col in text_columns[:3]:  # Show first 3 columns
                val = sample.get(col)
                if val:
                    if isinstance(val, str):
                        parts.append(f"{col}: {val[:200]}")
                    elif isinstance(val, list):
                        parts.append(f"{col}: {str(val)[:200]}")
            example = '\n'.join(parts)
        
        return {
            "strategy": "combine_text_columns",
            "description": "Combine identified text columns",
            "columns_to_combine": text_columns,
            "example": example
        }
    
    else:
        return {
            "strategy": "no_text_found",
            "description": "No suitable text columns identified",
            "columns_to_combine": [],
            "example": None
        }


def explore_dataset(dataset_name: str, dataset_path: Path) -> Optional[Dict[str, Any]]:
    """
    Explore a complete dataset (all splits).
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset directory
    
    Returns:
        Dictionary containing dataset information or None if not found
    """
    if not dataset_path.exists():
        logger.warning(f"Dataset not found at {dataset_path}")
        return {
            "name": dataset_name,
            "status": "not_downloaded",
            "path": str(dataset_path)
        }
    
    logger.info(f"Exploring: {dataset_name}")
    logger.info(f"Path: {dataset_path}")
    
    # Find all splits (subdirectories with arrow files)
    splits = []
    split_info_map = {}
    
    for item in dataset_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it contains arrow files
            arrow_files = list(item.glob("*.arrow"))
            if arrow_files:
                splits.append(item.name)
    
    if not splits:
        logger.warning(f"No splits found in {dataset_path}")
        return None
    
    splits.sort()
    logger.info(f"Found {len(splits)} splits: {splits}")
    
    # Explore each split
    split_sizes = {}
    all_columns = set()
    
    for split_name in splits:
        logger.info(f"  Processing split: {split_name}")
        split_data = explore_split(dataset_path, split_name)
        split_info_map[split_name] = split_data
        split_sizes[split_name] = split_data["num_rows"]
        all_columns.update(split_data["columns"])
    
    # Aggregate information
    total_rows = sum(split_sizes.values())
    
    # Use first split for overall analysis
    first_split = split_info_map[splits[0]]
    
    result = {
        "name": dataset_name,
        "status": "loaded_from_disk",
        "path": str(dataset_path),
        "splits": splits,
        "split_sizes": split_sizes,
        "total_rows": total_rows,
        "columns": first_split["columns"],
        "num_columns": first_split["num_columns"],
        "text_columns": first_split["text_columns"],
        "column_analysis": first_split["column_analysis"],
        "conversational_format": first_split["conversational_format"],
        "embedding_strategy": determine_embedding_strategy(first_split),
        "split_details": split_info_map
    }
    
    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Text columns: {first_split['text_columns']}")
    
    return result


def generate_extraction_functions(datasets_info: List[Dict[str, Any]]) -> str:
    """
    Generate Python functions for extracting text from each dataset format.
    
    Returns:
        Python source code as string
    """
    code_lines = [
        "# Auto-generated embedding extraction functions",
        "# Generated by 01_explore_nemotron_datasets.py",
        "",
        ""
    ]
    
    for ds_info in datasets_info:
        if ds_info.get("status") != "loaded_from_disk":
            continue
        
        dataset_name = ds_info["name"]
        function_name = dataset_name.replace("/", "_").replace("-", "_").lower()
        strategy = ds_info["embedding_strategy"]
        conv_format = ds_info["conversational_format"]
        
        # Generate function
        code_lines.append(f"def extract_text_{function_name}(example):")
        code_lines.append(f'    """Extract embeddable text from {dataset_name}"""')
        
        if strategy["strategy"] == "concatenate_messages":
            if 'messages' in strategy["columns_to_combine"]:
                code_lines.append("    # Conversational format - concatenate messages")
                code_lines.append("    messages = example['messages']")
                code_lines.append("    text_parts = []")
                code_lines.append("    for msg in messages:")
                code_lines.append("        role = msg.get('role', 'unknown')")
                code_lines.append("        content = msg.get('content', '')")
                
                if 'reasoning_content' in conv_format.get("message_keys", []):
                    code_lines.append("        # Include reasoning_content if available")
                    code_lines.append("        if 'reasoning_content' in msg and msg['reasoning_content']:")
                    code_lines.append("            content = msg['reasoning_content'] + '\\n' + content")
                
                code_lines.append("        text_parts.append(f'{role}: {content}')")
                code_lines.append("    return '\\n'.join(text_parts)")
        
        elif strategy["strategy"] == "direct_text":
            # Pretraining datasets with single text column
            text_col = strategy["columns_to_combine"][0]
            code_lines.append(f"    # Pretraining format - direct text extraction")
            code_lines.append(f"    return str(example.get('{text_col}', ''))")
        
        elif strategy["strategy"] == "combine_text_columns":
            code_lines.append("    # Combine text columns")
            code_lines.append("    parts = []")
            for col in strategy["columns_to_combine"][:5]:  # Limit to 5 columns
                code_lines.append(f"    if example.get('{col}'):")
                code_lines.append(f"        parts.append(f'{col.title()}: {{example[\"{col}\"]}}')") 
            code_lines.append("    return '\\n'.join(parts)")
        
        else:
            code_lines.append("    # No suitable text extraction strategy found")
            code_lines.append("    return str(example)")
        
        code_lines.append("")
        code_lines.append("")
    
    return '\n'.join(code_lines)


def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("NeMo Curator Dataset Exploration")
    logger.info("="*80)
    
    # Get datasets directory
    datasets_dir = Path(DEFAULT_DATASETS_DIR)
    logger.info(f"Datasets directory: {datasets_dir}")
    
    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return
    
    # Get list of datasets to explore (only downloaded ones)
    # Use a dict to deduplicate datasets (e.g., Llama-Nemotron has both SFT and RL configs)
    datasets_dict = {}
    for key, config in DATASET_CONFIGS.items():
        dataset_name = config["hf_name"]
        dataset_path = datasets_dir / dataset_name.replace("/", "_")
        # Only add if not already in dict (avoids duplicates)
        if dataset_name not in datasets_dict:
            datasets_dict[dataset_name] = dataset_path
    
    # Convert to list
    datasets_to_explore = list(datasets_dict.items())
    
    logger.info(f"Exploring {len(datasets_to_explore)} unique datasets\n")
    
    # Explore each dataset
    all_results = []
    
    for i, (dataset_name, dataset_path) in enumerate(datasets_to_explore, 1):
        logger.info(f"\n[{i}/{len(datasets_to_explore)}] {dataset_name}")
        logger.info("-" * 80)
        
        try:
            result = explore_dataset(dataset_name, dataset_path)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"Error exploring {dataset_name}: {e}", exc_info=True)
            all_results.append({
                "name": dataset_name,
                "status": "error",
                "error": str(e)
            })
    
    # Generate summary statistics
    logger.info("\n" + "="*80)
    logger.info("EXPLORATION SUMMARY")
    logger.info("="*80)
    
    downloaded_count = sum(1 for r in all_results if r.get("status") == "loaded_from_disk")
    total_rows = sum(r.get("total_rows", 0) for r in all_results if r.get("status") == "loaded_from_disk")
    
    logger.info(f"Datasets explored: {len(all_results)}")
    logger.info(f"Successfully loaded: {downloaded_count}")
    logger.info(f"Total rows across all datasets: {total_rows:,}")
    
    # Print summary table
    logger.info("\nDataset Summary:")
    logger.info("-" * 80)
    for result in all_results:
        if result.get("status") == "loaded_from_disk":
            name = result["name"].split("/")[-1]
            rows = result["total_rows"]
            splits = len(result["splits"])
            cols = result["num_columns"]
            text_cols = len(result["text_columns"])
            logger.info(f"{name:50s} {rows:12,d} rows  {splits:2d} splits  {cols:2d} cols  {text_cols:2d} text")
    
    # Save detailed JSON
    logger.info(f"\nSaving detailed results to: {DETAILED_JSON}")
    with open(DETAILED_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary JSON (without sample rows for smaller size)
    summary_results = []
    for r in all_results:
        r_copy = r.copy()
        if 'split_details' in r_copy:
            # Remove sample_rows from split details
            split_details = {}
            for split_name, split_data in r_copy['split_details'].items():
                split_copy = split_data.copy()
                split_copy.pop('sample_rows', None)
                split_details[split_name] = split_copy
            r_copy['split_details'] = split_details
        summary_results.append(r_copy)
    
    logger.info(f"Saving summary to: {SUMMARY_JSON}")
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    # Generate extraction functions
    logger.info(f"\nGenerating extraction functions: {EXTRACTION_FUNCTIONS_PY}")
    extraction_code = generate_extraction_functions(all_results)
    with open(EXTRACTION_FUNCTIONS_PY, 'w') as f:
        f.write(extraction_code)
    
    logger.info("\n" + "="*80)
    logger.info("EXPLORATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOutputs:")
    logger.info(f"  - {DETAILED_JSON} (full details with samples)")
    logger.info(f"  - {SUMMARY_JSON} (compact summary)")
    logger.info(f"  - {EXTRACTION_FUNCTIONS_PY} (auto-generated functions)")


if __name__ == "__main__":
    main()
