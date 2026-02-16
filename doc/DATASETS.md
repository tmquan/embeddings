# Nemotron Post-Training Datasets Reference

Comprehensive metadata for all 11 NVIDIA Nemotron post-training datasets used in the embedding extraction pipeline.

**Total: 78,564,106 records across 45 splits (11 datasets)**

*Data sourced from exploration run on 2026-02-16.*

---

## Quick Reference

| # | Dataset | HuggingFace | Format | Splits | Files | Total Rows | Text Strategy | Median Tokens | % Over 32K |
|---|---------|-------------|--------|--------|-------|------------|---------------|---------------|------------|
| 1 | Llama-Nemotron-Post-Training-Dataset | `nvidia/Llama-Nemotron-Post-Training-Dataset` | jsonl | 10 | 10 | 33,035,757 | `messages_concat` | — | — |
| 2 | Nemotron-Post-Training-Dataset-v1 | `nvidia/Nemotron-Post-Training-Dataset-v1` | parquet | 5 | 1,023 | 25,659,642 | `messages_list` | — | — |
| 3 | Nemotron-Post-Training-Dataset-v2 | `nvidia/Nemotron-Post-Training-Dataset-v2` | parquet | 10 | 202 | 6,341,514 | `messages_list` | — | — |
| 4 | Nemotron-3-Nano-RL-Training-Blend | `nvidia/Nemotron-3-Nano-RL-Training-Blend` | jsonl | 1 | 1 | 93,244 | `rl_blend` | 152 | 0.0% |
| 5 | Nemotron-Science-v1 | `nvidia/Nemotron-Science-v1` | jsonl | 2 | 2 | 226,334 | `messages_list` | 443 | 0.0% |
| 6 | Nemotron-Instruction-Following-Chat-v1 | `nvidia/Nemotron-Instruction-Following-Chat-v1` | jsonl | 2 | 2 | 430,978 | `messages_list` | 2,100 | 0.0% |
| 7 | Nemotron-Math-Proofs-v1 | `nvidia/Nemotron-Math-Proofs-v1` | jsonl | 1 | 1 | 1,376,663 | `math_proof` | 1,200 | 0.0% |
| 8 | Nemotron-Agentic-v1 | `nvidia/Nemotron-Agentic-v1` | jsonl | 2 | 2 | 335,122 | `agentic` | 1,600 | 0.0% |
| 9 | Nemotron-Competitive-Programming-v1 | `nvidia/Nemotron-Competitive-Programming-v1` | jsonl | 6 | 6 | 3,927,984 | `messages_list` | 1,200 | 0.0% |
| 10 | Nemotron-Math-v2 | `nvidia/Nemotron-Math-v2` | jsonl | 5 | 5 | 7,085,839 | `math_v2` | 697 | 0.0% |
| 11 | Nemotron-SWE-v1 | `nvidia/Nemotron-SWE-v1` | jsonl | 1 | 1 | 51,029 | `agentic` | 34,000 | 54.9% |

> Token estimates use ~4 chars/token approximation. "% Over 32K" = percentage of sampled records exceeding MODEL_MAX_TOKENS (32,768).
> Datasets 1-3 token stats are available in `doc/dataset_exploration_summary.md`.

---

## Detailed Dataset Cards

### 1. Llama-Nemotron-Post-Training-Dataset

| Property | Value |
|----------|-------|
| **Dataset name** | Llama-Nemotron-Post-Training-Dataset |
| **HuggingFace** | [`nvidia/Llama-Nemotron-Post-Training-Dataset`](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) |
| **Category** | post-training |
| **Format** | JSONL |
| **Total rows** | 33,035,757 |
| **Splits** | 10 |
| **Files** | 10 |
| **Text strategy** | `messages_concat` |
| **Local path** | `/raid/datasets/nvidia_Llama-Nemotron-Post-Training-Dataset` |

**Columns:** `input` (list of message dicts), `output` (str), `system_prompt` (str)

**Text-bearing columns:** `input`, `output`

**Concatenation strategy:** Flatten the `input` message list (role: content pairs) and append the `output` field. Format: `User: ... \nAssistant: ...` preserving multi-turn structure. For RL/when2call subsets, flatten the `messages` list instead.

**Preprocessing:** Strip `system_prompt` if it duplicates the first system message. Remove `<think>` tags for non-reasoning variants if needed.

**Splits:**

| Split | File Path | Files |
|-------|-----------|-------|
| SFT_chat | `SFT/chat/chat.jsonl` | 1 |
| SFT_code_v1 | `SFT/code/code_v1.jsonl` | 1 |
| SFT_code_v1.1 | `SFT/code/code_v1.1.jsonl` | 1 |
| SFT_math_v1 | `SFT/math/math_v1.jsonl` | 1 |
| SFT_math_v1.1 | `SFT/math/math_v1.1.jsonl` | 1 |
| SFT_safety | `SFT/safety/safety.jsonl` | 1 |
| SFT_science | `SFT/science/science.jsonl` | 1 |
| RL_instruction_following | `RL/instruction_following/instruction_following.jsonl` | 1 |
| train_when2call_sft | `train/when2call_train_sft.jsonl` | 1 |
| train_when2call_pref | `train/when2call_train_pref.jsonl` | 1 |

---

### 2. Nemotron-Post-Training-Dataset-v1

| Property | Value |
|----------|-------|
| **Dataset name** | Nemotron-Post-Training-Dataset-v1 |
| **HuggingFace** | [`nvidia/Nemotron-Post-Training-Dataset-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) |
| **Category** | post-training |
| **Format** | Parquet (sharded) |
| **Total rows** | 25,659,642 |
| **Splits** | 5 |
| **Files** | 1,023 |
| **Text strategy** | `messages_list` |
| **Local path** | `/raid/datasets/nvidia_Nemotron-Post-Training-Dataset-v1` |

**Columns:** `messages` (list of {role, content} dicts), `metadata` (JSON string)

**Text-bearing columns:** `messages`

**Concatenation strategy:** Flatten the `messages` column (list of {role, content} dicts) into a single text. Format: `User: ...\nAssistant: ...`. Multi-turn conversations are concatenated in order.

**Preprocessing:** Handle nested `tool_calls` structs — serialize tool call names/arguments as text if present. Skip `metadata` column (JSON string).

**Splits:**

| Split | File Pattern | Files |
|-------|-------------|-------|
| code | `data/code-*.parquet` | 183 |
| math | `data/math-*.parquet` | 159 |
| stem | `data/stem-*.parquet` | 660 |
| tool | `data/tool-*.parquet` | 13 |
| chat | `data/chat-*.parquet` | 8 |

---

### 3. Nemotron-Post-Training-Dataset-v2

| Property | Value |
|----------|-------|
| **Dataset name** | Nemotron-Post-Training-Dataset-v2 |
| **HuggingFace** | [`nvidia/Nemotron-Post-Training-Dataset-v2`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) |
| **Category** | post-training |
| **Format** | Parquet (sharded) |
| **Total rows** | 6,341,514 |
| **Splits** | 10 |
| **Files** | 202 |
| **Text strategy** | `messages_list` |
| **Local path** | `/raid/datasets/nvidia_Nemotron-Post-Training-Dataset-v2` |

**Columns:** `messages` (list of {role, content} dicts), `metadata` (string, optional)

**Text-bearing columns:** `messages`

**Concatenation strategy:** Same as v1: flatten `messages` list into role-prefixed text. Includes multilingual subsets (DE, ES, FR, IT, JA) — embeddings will capture cross-lingual semantics.

**Preprocessing:** No `tool_calls` in v2 schema. Some shards have `metadata` column (string) — ignore for embedding. Handle non-Latin scripts (JA) carefully.

**Splits:**

| Split | File Pattern | Files |
|-------|-------------|-------|
| chat | `data/chat-*.parquet` | 12 |
| code | `data/code-*.parquet` | 2 |
| math | `data/math-*.parquet` | 2 |
| stem | `data/stem-*.parquet` | 2 |
| multilingual | `data/multilingual-*.parquet` | 1 |
| multilingual_de | `data/multilingual_de-*.parquet` | 38 |
| multilingual_es | `data/multilingual_es-*.parquet` | 33 |
| multilingual_fr | `data/multilingual_fr-*.parquet` | 37 |
| multilingual_it | `data/multilingual_it-*.parquet` | 38 |
| multilingual_ja | `data/multilingual_ja-*.parquet` | 37 |

---

### 4. Nemotron-3-Nano-RL-Training-Blend

| Property | Value |
|----------|-------|
| **Dataset name** | Nemotron-3-Nano-RL-Training-Blend |
| **HuggingFace** | [`nvidia/Nemotron-3-Nano-RL-Training-Blend`](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend) |
| **Category** | post-training |
| **Format** | JSONL |
| **Total rows** | 93,244 |
| **Total tokens (est.)** | ~28.1M (mean 301 tokens/record) |
| **Splits** | 1 |
| **Files** | 1 |
| **Text strategy** | `rl_blend` |
| **Local path** | `/raid/datasets/nvidia_Nemotron-3-Nano-RL-Training-Blend` |

**Columns:** `id` (int), `responses_create_params` (dict), `ground_truth` (list), `category` (str), `environment_name` (str), `agent_ref` (dict), `pass_rate` (float), `pass_rate_total` (int), `pass_rate_passed` (int), `dataset` (str), `expected_answer` (str), `uuid` (str), `options` (list), `reward_profiles` (list), `template_metadata` (dict), `instruction_id_list` (list), `prompt` (str), `kwargs` (list), `_hf_placeholder` (dict), `verifier_metadata` (dict), `hash_id` (str), `source` (str)

**Text-bearing columns:** `responses_create_params` (nested dict containing `input` message list), `ground_truth` (list of tool-call dicts)

**Concatenation strategy:** Extract `input` messages from `responses_create_params` dict, flatten into role-prefixed text. Optionally append `ground_truth` tool calls as structured text.

**Preprocessing:** Deep-extract nested dict: `responses_create_params['input']` contains the message list. Ground truth is a list of tool-call dicts.

**Token statistics (sampled):**

| Stat | Tokens |
|------|--------|
| Min | 0 |
| Median | 152 |
| Mean | 301 |
| Max | 3,200 |
| % over 32K | 0.0% |

**Splits:**

| Split | File | Files | Rows |
|-------|------|-------|------|
| train | `train.jsonl` | 1 | 93,244 |

**Data quality notes:**
- `id`: 70% null
- `responses_create_params`: 24% null
- `ground_truth`: 89% null
- `category`: 89% null
- `environment_name`: 89% null
- `expected_answer`: 79% null

---

### 5. Nemotron-Science-v1

| Property | Value |
|----------|-------|
| **Dataset name** | Nemotron-Science-v1 |
| **HuggingFace** | [`nvidia/Nemotron-Science-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) |
| **Category** | post-training |
| **Format** | JSONL |
| **Total rows** | 226,334 |
| **Total tokens (est.)** | ~113.8M (mean 503 tokens/record) |
| **Splits** | 2 |
| **Files** | 2 |
| **Text strategy** | `messages_list` |
| **Local path** | `/raid/datasets/nvidia_Nemotron-Science-v1` |

**Columns:** `uuid` (str), `messages` (list), `license` (str), `used_in` (list), `tools` (list)

**Text-bearing columns:** `messages`

**Concatenation strategy:** Flatten `messages` list (user question + assistant answer). MCQ: multiple-choice science questions. RQA: research Q&A with boxed answers.

**Preprocessing:** None needed — clean chat format.

**Token statistics (sampled):**

| Stat | Tokens |
|------|--------|
| Min | 82 |
| Median | 443 |
| Mean | 503 |
| Max | 1,800 |
| % over 32K | 0.0% |

**Splits:**

| Split | File | Files | Rows |
|-------|------|-------|------|
| MCQ | `data/MCQ.jsonl` | 1 | 174,200 |
| RQA | `data/RQA.jsonl` | 1 | 52,200 |

---

### 6. Nemotron-Instruction-Following-Chat-v1

| Property | Value |
|----------|-------|
| **Dataset name** | Nemotron-Instruction-Following-Chat-v1 |
| **HuggingFace** | [`nvidia/Nemotron-Instruction-Following-Chat-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) |
| **Category** | post-training |
| **Format** | JSONL |
| **Total rows** | 430,978 |
| **Total tokens (est.)** | ~1.29B (mean 3,000 tokens/record) |
| **Splits** | 2 |
| **Files** | 2 |
| **Text strategy** | `messages_list` |
| **Local path** | `/raid/datasets/nvidia_Nemotron-Instruction-Following-Chat-v1` |

**Columns (chat_if):** `uuid` (str), `messages` (list), `license` (str), `used_in` (list), `tools` (list), `reasoning` (str), `capability_target` (str)

**Columns (structured_outputs):** `uuid` (str), `messages` (list), `license` (str), `used_in` (list), `tools` (list)

**Text-bearing columns:** `messages`

**Concatenation strategy:** Flatten `messages` into text. Includes system prompts with specific instruction-following constraints and structured output schemas.

**Preprocessing:** System messages may contain XML/JSON schemas for structured outputs. Keep them as-is for semantic richness.

**Token statistics (sampled):**

| Stat | Tokens |
|------|--------|
| Min | 50 |
| Median | 2,100 |
| Mean | 3,000 |
| Max | 22,000 |
| % over 32K | 0.0% |

**Splits:**

| Split | File | Files | Rows |
|-------|------|-------|------|
| chat_if | `data/chat_if.jsonl` | 1 | 426,000 |
| structured_outputs | `data/structured_outputs.jsonl` | 1 | 5,000 |

---

### 7. Nemotron-Math-Proofs-v1

| Property | Value |
|----------|-------|
| **Dataset name** | Nemotron-Math-Proofs-v1 |
| **HuggingFace** | [`nvidia/Nemotron-Math-Proofs-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1) |
| **Category** | post-training |
| **Format** | JSONL |
| **Total rows** | 1,376,663 |
| **Total tokens (est.)** | ~2.75B (mean 2,000 tokens/record) |
| **Splits** | 1 |
| **Files** | 1 |
| **Text strategy** | `math_proof` |
| **Local path** | `/raid/datasets/nvidia_Nemotron-Math-Proofs-v1` |

**Columns:** `problem` (str), `source` (str), `formal_statement` (str), `lean_header` (str), `url` (NoneType), `user_name` (NoneType), `user_url` (NoneType), `sft_line_number` (NoneType), `messages` (list), `uuid` (str), `used_in` (list), `tools` (list), `license` (str)

**Text-bearing columns:** `problem`, `formal_statement`, `lean_header`

**Concatenation strategy:** Concatenate: `Problem: {problem}\n\nFormal Statement (Lean 4):\n{lean_header}\n{formal_statement}`. Combines natural-language math with formal verification code (Lean 4).

**Preprocessing:** Many fields (`url`, `user_name`, `sft_line_number`) are None — skip them. The `messages` field is often empty; use `problem` + `formal_statement`.

**Token statistics (sampled):**

| Stat | Tokens |
|------|--------|
| Min | 30 |
| Median | 1,200 |
| Mean | 2,000 |
| Max | 18,000 |
| % over 32K | 0.0% |

**Splits:**

| Split | File | Files | Rows |
|-------|------|-------|------|
| lean | `data/lean.jsonl` | 1 | 1,376,663 |

**Data quality notes:**
- `url`: 13% null
- `user_name`: 13% null
- `user_url`: 20% null
- `sft_line_number`: 33% null

---

### 8. Nemotron-Agentic-v1

| Property | Value |
|----------|-------|
| **Dataset name** | Nemotron-Agentic-v1 |
| **HuggingFace** | [`nvidia/Nemotron-Agentic-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) |
| **Category** | post-training |
| **Format** | JSONL |
| **Total rows** | 335,122 |
| **Total tokens (est.)** | ~502.7M (mean 1,500 tokens/record) |
| **Splits** | 2 |
| **Files** | 2 |
| **Text strategy** | `agentic` |
| **Local path** | `/raid/datasets/nvidia_Nemotron-Agentic-v1` |

**Columns (tool_calling):** `uuid` (str), `messages` (list), `license` (str), `used_in` (list), `tools` (list)

**Columns (interactive_agent):** `uuid` (str), `messages` (list), `license` (str), `used_in` (list), `tools` (list), `reasoning` (str)

**Text-bearing columns:** `messages`, `tools`

**Concatenation strategy:** Flatten `messages` into text. Prepend serialized `tools` definitions (function name + description) as context. This captures the agentic task structure: available tools + conversation.

**Preprocessing:** Serialize tool definitions to compact text: `Tool: {name} - {description}`. Skip detailed parameter schemas.

**Token statistics (sampled):**

| Stat | Tokens |
|------|--------|
| Min | 52 |
| Median | 1,600 |
| Mean | 1,500 |
| Max | 6,400 |
| % over 32K | 0.0% |

**Splits:**

| Split | File | Files | Rows |
|-------|------|-------|------|
| tool_calling | `data/tool_calling.jsonl` | 1 | 316,100 |
| interactive_agent | `data/interactive_agent.jsonl` | 1 | 19,000 |

---

### 9. Nemotron-Competitive-Programming-v1

| Property | Value |
|----------|-------|
| **Dataset name** | Nemotron-Competitive-Programming-v1 |
| **HuggingFace** | [`nvidia/Nemotron-Competitive-Programming-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) |
| **Category** | post-training |
| **Format** | JSONL |
| **Total rows** | 3,927,984 |
| **Total tokens (est.)** | ~4.71B (mean 1,200 tokens/record) |
| **Splits** | 6 |
| **Files** | 6 |
| **Text strategy** | `messages_list` |
| **Local path** | `/raid/datasets/nvidia_Nemotron-Competitive-Programming-v1` |

**Columns (cpp/python):** `uuid` (str), `messages` (list), `license` (str), `used_in` (list), `tools` (list), `dataset` (str), `split` (str), `index` (str), `source` (str), `difficulty` (str), `question_id` (str)

**Columns (infinibyte):** `uuid` (str), `messages` (list), `license` (str), `used_in` (list), `tools` (list)

**Text-bearing columns:** `messages`

**Concatenation strategy:** Flatten `messages` (user problem + assistant solution with code). Competitive programming: problem statement + reasoning + code solution.

**Preprocessing:** Solutions contain code blocks — preserve formatting. Some messages have `-` as user content (placeholder); may need filtering.

**Token statistics (sampled):**

| Stat | Tokens |
|------|--------|
| Min | 165 |
| Median | 1,200 |
| Mean | 1,200 |
| Max | 5,000 |
| % over 32K | 0.0% |

**Splits:**

| Split | File | Files | Rows |
|-------|------|-------|------|
| cpp_part0 | `data/competitive_coding_cpp.part_00.jsonl` | 1 | 466,000 |
| cpp_part1 | `data/competitive_coding_cpp.part_01.jsonl` | 1 | 466,000 |
| python_part0 | `data/competitive_coding_python.part_00.jsonl` | 1 | 910,600 |
| python_part1 | `data/competitive_coding_python.part_01.jsonl` | 1 | 910,600 |
| infinibyte_part0 | `data/infinibyte.part_00.jsonl` | 1 | 587,300 |
| infinibyte_part1 | `data/infinibyte.part_01.jsonl` | 1 | 587,300 |

---

### 10. Nemotron-Math-v2

| Property | Value |
|----------|-------|
| **Dataset name** | Nemotron-Math-v2 |
| **HuggingFace** | [`nvidia/Nemotron-Math-v2`](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) |
| **Category** | post-training |
| **Format** | JSONL |
| **Total rows** | 7,085,839 |
| **Total tokens (est.)** | ~5.17B (mean 729 tokens/record) |
| **Splits** | 5 |
| **Files** | 5 |
| **Text strategy** | `math_v2` |
| **Local path** | `/raid/datasets/nvidia_Nemotron-Math-v2` |

**Columns:** `problem` (str), `expected_answer` (str), `original_expected_answer` (str/NoneType), `changed_answer_to_majority` (bool), `data_source` (str), `messages` (list), `metadata` (dict), `license` (str), `used_in` (list), `uuid` (str), `url` (str/NoneType), `user_url` (str/NoneType), `user_name` (str/NoneType), `tools` (list)

**Text-bearing columns:** `messages`, `problem`

**Concatenation strategy:** Primary text: flatten `messages` (user prompt with `\boxed{}` instruction + assistant solution). Fallback: use `problem` field directly. Difficulty tiers: low, medium, high.

**Preprocessing:** Remove `Solve the following math problem...` boilerplate prefix from user messages if desired. Keep `\boxed{answer}` in text.

**Token statistics (sampled):**

| Stat | Tokens |
|------|--------|
| Min | 53 |
| Median | 697 |
| Mean | 729 |
| Max | 3,900 |
| % over 32K | 0.0% |

**Splits:**

| Split | File | Files | Rows |
|-------|------|-------|------|
| low | `data/low.jsonl` | 1 | 1,700,000 |
| medium | `data/medium.jsonl` | 1 | 2,500,000 |
| high_part0 | `data/high.part_00.jsonl` | 1 | 695,900 |
| high_part1 | `data/high.part_01.jsonl` | 1 | 1,100,000 |
| high_part2 | `data/high.part_02.jsonl` | 1 | 1,100,000 |

**Data quality notes:**
- `original_expected_answer`: 100% null in `low` and `high_part1` splits
- `url`, `user_name`, `user_url`: 100% null in `high_part0` and `high_part1` splits

---

### 11. Nemotron-SWE-v1

| Property | Value |
|----------|-------|
| **Dataset name** | Nemotron-SWE-v1 |
| **HuggingFace** | [`nvidia/Nemotron-SWE-v1`](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1) |
| **Category** | post-training |
| **Format** | JSONL |
| **Total rows** | 51,029 |
| **Total tokens (est.)** | ~1.87B (mean 36,700 tokens/record) |
| **Splits** | 1 |
| **Files** | 1 |
| **Text strategy** | `agentic` |
| **Local path** | `/raid/datasets/nvidia_Nemotron-SWE-v1` |

**Columns:** `uuid` (str), `messages` (list), `license` (str), `used_in` (list), `tools` (list), `dataset` (str), `repo` (str)

**Text-bearing columns:** `messages`, `tools`

**Concatenation strategy:** Software engineering agent data: flatten long multi-turn `messages` (system prompt with agent instructions + user bug report + assistant tool calls + tool outputs). Very long conversations.

**Preprocessing:** WARNING: Messages can be extremely long (>100k chars). May need truncation to MODEL_MAX_TOKENS. System prompt is ~2k chars of agent instructions — consider summarizing or truncating.

**Token statistics (sampled):**

| Stat | Tokens |
|------|--------|
| Min | 13,500 |
| Median | 34,000 |
| Mean | 36,700 |
| Max | 106,800 |
| % over 32K | **54.9%** |

**Splits:**

| Split | File | Files | Rows |
|-------|------|-------|------|
| r2e_gym | `data/r2e_gym.jsonl` | 1 | 51,029 |

---

## Text Concatenation Strategies

Summary of how raw records are converted into embedding-ready text for each strategy template.

| Strategy | Template | Datasets | Text Fields | Description |
|----------|----------|----------|-------------|-------------|
| Messages list | `messages_list` | v1, v2, Science, Instruction-Following, Competitive-Programming | `messages` | Flatten `messages` list: `Role: content` blocks joined by `\n\n` |
| Messages concat | `messages_concat` | Llama-Nemotron-Post-Training | `input`, `output` | Flatten `input` message list + append `output`; fallback to `messages` |
| RL blend | `rl_blend` | Nano-RL-Training-Blend | `responses_create_params`, `ground_truth` | Deep-extract `input` from nested dict + append ground_truth as JSON |
| Math proof | `math_proof` | Math-Proofs-v1 | `problem`, `formal_statement`, `lean_header` | Concatenate problem + Lean 4 formal statement + header |
| Math v2 | `math_v2` | Math-v2 | `messages`, `problem` | Prefer `messages`; fallback to `problem` field |
| Agentic | `agentic` | Agentic-v1, SWE-v1 | `messages`, `tools` | Prepend serialized tool definitions + flatten messages |

---

## Estimated Output Sizes

Embedding output: 4096 dimensions x 4 bytes (float32) per record.

| Dataset | Records | Embedding Output Size |
|---------|---------|----------------------|
| Llama-Nemotron-Post-Training-Dataset | 33,035,757 | ~541 GB |
| Nemotron-Post-Training-Dataset-v1 | 25,659,642 | ~420 GB |
| Nemotron-Post-Training-Dataset-v2 | 6,341,514 | ~104 GB |
| Nemotron-3-Nano-RL-Training-Blend | 93,244 | ~1.5 GB |
| Nemotron-Science-v1 | 226,334 | ~3.7 GB |
| Nemotron-Instruction-Following-Chat-v1 | 430,978 | ~7.1 GB |
| Nemotron-Math-Proofs-v1 | 1,376,663 | ~22.6 GB |
| Nemotron-Agentic-v1 | 335,122 | ~5.5 GB |
| Nemotron-Competitive-Programming-v1 | 3,927,984 | ~64.3 GB |
| Nemotron-Math-v2 | 7,085,839 | ~116.1 GB |
| Nemotron-SWE-v1 | 51,029 | ~0.8 GB |
| **Total** | **78,564,106** | **~1,287 GB (~1.3 TB)** |

---

## Notes

- All 11 datasets are post-training format (SFT / RL / chat / code / math / agentic).
- Pretraining datasets exist (~24 TB) but are not included in this pipeline (commented out in `dataset_configs.py`).
- Token statistics are sampled estimates (~4 chars/token) from the first 1,000 records per split.
- The `messages` column across most datasets contains a list of `{role: str, content: str}` dicts representing multi-turn chat conversations.
- Configuration source: `dataset_configs.py`. Exploration data: `doc/dataset_exploration_summary.md`.
