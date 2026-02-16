"""
Centralised dataset configuration for all Nemotron embedding pipeline scripts.

Each entry maps a short dataset key to its configuration:
    hf_name        – HuggingFace repo id (used for the local directory name)
    category       – "post-training" or "pretraining"
    format         – "jsonl" or "parquet"
    sub_paths      – list of (label, relative_glob_or_path) within the downloaded repo
    text_strategy  – dict describing how to concatenate fields into embedding text:
        fields         : list of field names or special extractors
        template       : extraction template name (messages_list, messages_concat, rl_blend, etc.)
        description    : human-readable explanation (optional)
        preprocessing  : preprocessing notes (optional)

Imported by:
    - 01_explore_nemotron_datasets.py
    - 02_embedding_extraction_huggingface.py
    - 02_embedding_extraction_nemocurator.py
    - 04_embedding_reduction_nemocurator.py
"""

from __future__ import annotations

from typing import Dict

DATASET_CONFIGS: Dict[str, dict] = {
    # =====================================================================
    # POST-TRAINING DATASETS (from 00_download: Llama / Nemotron-Post-Training v1/v2)
    # =====================================================================
    "Llama-Nemotron-Post-Training-Dataset": {
        "hf_name": "nvidia/Llama-Nemotron-Post-Training-Dataset",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("SFT_chat", "SFT/chat/chat.jsonl"),
            ("SFT_code_v1", "SFT/code/code_v1.jsonl"),
            ("SFT_code_v1.1", "SFT/code/code_v1.1.jsonl"),
            ("SFT_math_v1", "SFT/math/math_v1.jsonl"),
            ("SFT_math_v1.1", "SFT/math/math_v1.1.jsonl"),
            ("SFT_safety", "SFT/safety/safety.jsonl"),
            ("SFT_science", "SFT/science/science.jsonl"),
            ("RL_instruction_following", "RL/instruction_following/instruction_following.jsonl"),
            ("train_when2call_sft", "train/when2call_train_sft.jsonl"),
            ("train_when2call_pref", "train/when2call_train_pref.jsonl"),
        ],
        "text_strategy": {
            "fields": ["input", "output"],
            "template": "messages_concat",
            "description": (
                "Flatten the 'input' message list (role: content pairs) and append "
                "the 'output' field. Format: 'User: ... \\nAssistant: ...' preserving "
                "multi-turn structure. For RL/when2call subsets, flatten the 'messages' "
                "list instead."
            ),
            "preprocessing": (
                "Strip system_prompt if it duplicates the first system message. "
                "Remove <think> tags for non-reasoning variants if needed."
            ),
        },
    },
    "Nemotron-Post-Training-Dataset-v1": {
        "hf_name": "nvidia/Nemotron-Post-Training-Dataset-v1",
        "category": "post-training",
        "format": "parquet",
        "sub_paths": [
            ("code", "data/code-*.parquet"),
            ("math", "data/math-*.parquet"),
            ("stem", "data/stem-*.parquet"),
            ("tool", "data/tool-*.parquet"),
            ("chat", "data/chat-*.parquet"),
        ],
        "text_strategy": {
            "fields": ["messages"],
            "template": "messages_list",
            "description": (
                "Flatten the 'messages' column (list of {role, content} dicts) into a "
                "single text. Format: 'User: ...\\nAssistant: ...' Multi-turn "
                "conversations are concatenated in order."
            ),
            "preprocessing": (
                "Handle nested tool_calls structs—serialize tool call names/arguments "
                "as text if present. Skip metadata column (JSON string)."
            ),
        },
    },
    "Nemotron-Post-Training-Dataset-v2": {
        "hf_name": "nvidia/Nemotron-Post-Training-Dataset-v2",
        "category": "post-training",
        "format": "parquet",
        "sub_paths": [
            ("chat", "data/chat-*.parquet"),
            ("code", "data/code-*.parquet"),
            ("math", "data/math-*.parquet"),
            ("stem", "data/stem-*.parquet"),
            ("multilingual", "data/multilingual-*.parquet"),
            ("multilingual_de", "data/multilingual_de-*.parquet"),
            ("multilingual_es", "data/multilingual_es-*.parquet"),
            ("multilingual_fr", "data/multilingual_fr-*.parquet"),
            ("multilingual_it", "data/multilingual_it-*.parquet"),
            ("multilingual_ja", "data/multilingual_ja-*.parquet"),
        ],
        "text_strategy": {
            "fields": ["messages"],
            "template": "messages_list",
            "description": (
                "Same as v1: flatten 'messages' list into role-prefixed text. "
                "Includes multilingual subsets (DE, ES, FR, IT, JA)—embeddings will "
                "capture cross-lingual semantics."
            ),
            "preprocessing": (
                "No tool_calls in v2 schema. Some shards have 'metadata' column "
                "(string)—ignore for embedding. Handle non-Latin scripts (JA) carefully."
            ),
        },
    },

    # =====================================================================
    # POST-TRAINING DATASETS (Nemotron v3 collection)
    # =====================================================================
    "Nemotron-3-Nano-RL-Training-Blend": {
        "hf_name": "nvidia/Nemotron-3-Nano-RL-Training-Blend",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("train", "train.jsonl"),
        ],
        "text_strategy": {
            "fields": ["responses_create_params", "ground_truth"],
            "template": "rl_blend",
            "description": (
                "Extract 'input' messages from responses_create_params dict, flatten "
                "into role-prefixed text. Optionally append ground_truth tool calls "
                "as structured text."
            ),
            "preprocessing": (
                "Deep-extract nested dict: responses_create_params['input'] contains "
                "the message list. Ground truth is a list of tool-call dicts."
            ),
        },
    },
    "Nemotron-Science-v1": {
        "hf_name": "nvidia/Nemotron-Science-v1",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("MCQ", "data/MCQ.jsonl"),
            ("RQA", "data/RQA.jsonl"),
        ],
        "text_strategy": {
            "fields": ["messages"],
            "template": "messages_list",
            "description": (
                "Flatten 'messages' list (user question + assistant answer). "
                "MCQ: multiple-choice science questions. RQA: research Q&A with "
                "boxed answers."
            ),
            "preprocessing": "None needed—clean chat format.",
        },
    },
    "Nemotron-Instruction-Following-Chat-v1": {
        "hf_name": "nvidia/Nemotron-Instruction-Following-Chat-v1",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("chat_if", "data/chat_if.jsonl"),
            ("structured_outputs", "data/structured_outputs.jsonl"),
        ],
        "text_strategy": {
            "fields": ["messages"],
            "template": "messages_list",
            "description": (
                "Flatten 'messages' into text. Includes system prompts with specific "
                "instruction-following constraints and structured output schemas."
            ),
            "preprocessing": (
                "System messages may contain XML/JSON schemas for structured outputs. "
                "Keep them as-is for semantic richness."
            ),
        },
    },
    "Nemotron-Math-Proofs-v1": {
        "hf_name": "nvidia/Nemotron-Math-Proofs-v1",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("lean", "data/lean.jsonl"),
        ],
        "text_strategy": {
            "fields": ["problem", "formal_statement", "lean_header"],
            "template": "math_proof",
            "description": (
                "Concatenate: 'Problem: {problem}\\n\\nFormal Statement (Lean 4):\\n"
                "{lean_header}\\n{formal_statement}'. Combines natural-language math "
                "with formal verification code."
            ),
            "preprocessing": (
                "Many fields (url, user_name, sft_line_number) are None—skip them. "
                "The 'messages' field is often empty; use problem + formal_statement."
            ),
        },
    },
    "Nemotron-Agentic-v1": {
        "hf_name": "nvidia/Nemotron-Agentic-v1",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("tool_calling", "data/tool_calling.jsonl"),
            ("interactive_agent", "data/interactive_agent.jsonl"),
        ],
        "text_strategy": {
            "fields": ["messages", "tools"],
            "template": "agentic",
            "description": (
                "Flatten 'messages' into text. Prepend serialized 'tools' definitions "
                "(function name + description) as context. This captures the agentic "
                "task structure: available tools + conversation."
            ),
            "preprocessing": (
                "Serialize tool definitions to compact text: "
                "'Tool: {name} - {description}'. Skip detailed parameter schemas."
            ),
        },
    },
    "Nemotron-Competitive-Programming-v1": {
        "hf_name": "nvidia/Nemotron-Competitive-Programming-v1",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("cpp_part0", "data/competitive_coding_cpp.part_00.jsonl"),
            ("cpp_part1", "data/competitive_coding_cpp.part_01.jsonl"),
            ("python_part0", "data/competitive_coding_python.part_00.jsonl"),
            ("python_part1", "data/competitive_coding_python.part_01.jsonl"),
            ("infinibyte_part0", "data/infinibyte.part_00.jsonl"),
            ("infinibyte_part1", "data/infinibyte.part_01.jsonl"),
        ],
        "text_strategy": {
            "fields": ["messages"],
            "template": "messages_list",
            "description": (
                "Flatten 'messages' (user problem + assistant solution with code). "
                "Competitive programming: problem statement + reasoning + code solution."
            ),
            "preprocessing": (
                "Solutions contain code blocks—preserve formatting. "
                "Some messages have '-' as user content (placeholder); may need filtering."
            ),
        },
    },
    "Nemotron-Math-v2": {
        "hf_name": "nvidia/Nemotron-Math-v2",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("low", "data/low.jsonl"),
            ("medium", "data/medium.jsonl"),
            ("high_part0", "data/high.part_00.jsonl"),
            ("high_part1", "data/high.part_01.jsonl"),
            ("high_part2", "data/high.part_02.jsonl"),
        ],
        "text_strategy": {
            "fields": ["problem", "messages"],
            "template": "math_v2",
            "description": (
                "Primary text: flatten 'messages' (user prompt with \\\\boxed{} instruction "
                "+ assistant solution). Fallback: use 'problem' field directly. "
                "Difficulty tiers: low, medium, high."
            ),
            "preprocessing": (
                "Remove 'Solve the following math problem...' boilerplate prefix from "
                "user messages if desired. Keep \\\\boxed{answer} in text."
            ),
        },
    },
    "Nemotron-SWE-v1": {
        "hf_name": "nvidia/Nemotron-SWE-v1",
        "category": "post-training",
        "format": "jsonl",
        "sub_paths": [
            ("r2e_gym", "data/r2e_gym.jsonl"),
        ],
        "text_strategy": {
            "fields": ["messages", "tools"],
            "template": "agentic",
            "description": (
                "Software engineering agent data: flatten long multi-turn 'messages' "
                "(system prompt with agent instructions + user bug report + assistant "
                "tool calls + tool outputs). Very long conversations."
            ),
            "preprocessing": (
                "WARNING: Messages can be extremely long (>100k chars). May need "
                "truncation to MODEL_MAX_TOKENS. System prompt is ~2k chars of "
                "agent instructions—consider summarizing or truncating."
            ),
        },
    },

    # =====================================================================
    # PRETRAINING DATASETS (commented out – ~24 TB total)
    # =====================================================================
    # "Nemotron-Pretraining-Dataset-sample": {
    #     "hf_name": "nvidia/Nemotron-Pretraining-Dataset-sample",
    #     "category": "pretraining",
    #     "format": "parquet",
    #     "sub_paths": [
    #         ("CC-High-Quality", "Nemotron-CC-High-Quality/part_*.parquet"),
    #         ("CC-High-Quality-Synthetic", "Nemotron-CC-High-Quality-Synthetic/part_*.parquet"),
    #         ("CC-Diverse-QA", "Nemotron-CC-Diverse-QA/part_*.parquet"),
    #         ("CC-Translated-Diverse-QA", "Nemotron-CC-Translated-Diverse-QA/part_*.parquet"),
    #         ("CC-MATH", "Nemotron-CC-MATH/part_*.parquet"),
    #         ("Code-Metadata", "Nemotron-Code-Metadata/part_*.parquet"),
    #         ("SFT-Code", "Nemotron-SFT-Code/part_*.parquet"),
    #         ("SFT-General", "Nemotron-SFT-General/part_*.parquet"),
    #         ("SFT-MATH", "Nemotron-SFT-MATH/part_*.parquet"),
    #         ("Synthetic-Code", "Nemotron-Synthetic-Code/part_*.parquet"),
    #     ],
    #     "text_strategy": {
    #         "fields": ["text"],
    #         "template": "raw_text",
    #         "description": (
    #             "Use the 'text' column directly—already contains the full document. "
    #             "For CC subsets: web-crawled, cleaned text. For SFT subsets: "
    #             "instruction-response pairs pre-formatted as text."
    #         ),
    #         "preprocessing": (
    #             "Code-Metadata subset has NO 'text' column (only repo, commit_id, "
    #             "rel_path)—skip or use as metadata only. "
    #             "CC-Translated-Diverse-QA has a 'language' column for filtering."
    #         ),
    #     },
    # },
}
