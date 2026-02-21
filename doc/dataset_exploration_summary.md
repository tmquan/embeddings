# Nemotron Dataset Exploration Summary

*Generated: 2026-02-16 08:42:05*

## Configuration

- **Datasets directory**: `/raid/datasets`
- **Default chunk size**: 100,000 rows
- **Model max tokens**: 32,768
- **Token estimation**: ~4 chars/token

## Dataset Summary Table

| # | Dataset | Category | Format | Row Count | Text Strategy | Median Tokens | % Over Max |
|---|---------|----------|--------|-----------|---------------|---------------|------------|
| 1 | Llama-Nemotron-Post-Training-Dataset | post-training | jsonl | 33.0M | `messages_concat` | 762 | 0.0% |
| 2 | Nemotron-Post-Training-Dataset-v1 | post-training | parquet | 25.7M | `messages_list` | 0 | 0% |
| 3 | Nemotron-Post-Training-Dataset-v2 | post-training | parquet | 6.3M | `messages_list` | 0 | 0% |
| 4 | Nemotron-3-Nano-RL-Training-Blend | post-training | jsonl | 93.2K | `rl_blend` | 152 | 0.0% |
| 5 | Nemotron-Science-v1 | post-training | jsonl | 226.3K | `messages_list` | 443 | 0.0% |
| 6 | Nemotron-Instruction-Following-Chat-v1 | post-training | jsonl | 431.0K | `messages_list` | 2.1K | 0.0% |
| 7 | Nemotron-Math-Proofs-v1 | post-training | jsonl | 1.4M | `math_proof` | 1.2K | 0.0% |
| 8 | Nemotron-Agentic-v1 | post-training | jsonl | 335.1K | `agentic` | 1.6K | 0.0% |
| 9 | Nemotron-Competitive-Programming-v1 | post-training | jsonl | 3.9M | `messages_list` | 1.2K | 0.0% |
| 10 | Nemotron-Math-v2 | post-training | jsonl | 7.1M | `math_v2` | 697 | 0.0% |
| 11 | Nemotron-SWE-v1 | post-training | jsonl | 51.0K | `agentic` | 34.0K | 54.9% |

---

## Detailed Findings

### Post-Training Datasets

#### Llama-Nemotron-Post-Training-Dataset

- **HuggingFace**: `nvidia/Llama-Nemotron-Post-Training-Dataset`
- **Format**: jsonl
- **Total rows**: 33,035,757
- **Text concatenation strategy**: Flatten the 'input' message list (role: content pairs) and append the 'output' field. Format: 'User: ... \nAssistant: ...' preserving multi-turn structure. For RL/when2call subsets, flatten the 'messages' list instead.
- **Preprocessing notes**: Strip system_prompt if it duplicates the first system message. Remove <think> tags for non-reasoning variants if needed.
- **Text length stats** (sampled): min=0 / median=762 / mean=2.4K / max=37.1K tokens

**Sub-path breakdown:**

| Sub-path | Files | Rows | Schema Fields |
|----------|-------|------|---------------|
| SFT_chat | 1 | 39.8K | input, output, category, license, reasoning, generator, u... |
| SFT_code_v1 | 1 | 9.6M | input, output, category, license, reasoning, generator, u... |
| SFT_code_v1.1 | 1 | 496.2K | input, output, category, license, reasoning, generator, u... |
| SFT_math_v1 | 1 | 19.8M | input, output, category, license, reasoning, generator, v... |
| SFT_math_v1.1 | 1 | 2.2M | input, output, category, license, reasoning, generator, u... |
| SFT_safety | 1 | 31.4K | input, output, category, generator, license, reasoning, u... |
| SFT_science | 1 | 708.9K | input, output, category, license, reasoning, generator, u... |
| RL_instruction_following | 1 | 56.3K | input, args, category, license, reasoning, used_in_traini... |
| train_when2call_sft | 1 | 15.0K | tools, messages |
| train_when2call_pref | 1 | 9.0K | tools, messages, chosen_response, rejected_response |

**Processing recommendation:**

- Chunked processing: ~330 chunks of 100,000 rows each

#### Nemotron-Post-Training-Dataset-v1

- **HuggingFace**: `nvidia/Nemotron-Post-Training-Dataset-v1`
- **Format**: parquet
- **Total rows**: 25,659,642
- **Text concatenation strategy**: Flatten the 'messages' column (list of {role, content} dicts) into a single text. Format: 'User: ...\nAssistant: ...' Multi-turn conversations are concatenated in order.
- **Preprocessing notes**: Handle nested tool_calls structs—serialize tool call names/arguments as text if present. Skip metadata column (JSON string).

**Sub-path breakdown:**

| Sub-path | Files | Rows | Schema Fields |
|----------|-------|------|---------------|
| code | 183 | 1.9M | uuid, license, generator, version, category, reasoning, m... |
| math | 159 | 2.0M | uuid, license, generator, version, category, reasoning, m... |
| stem | 660 | 20.7M | uuid, license, generator, version, category, reasoning, m... |
| tool | 13 | 310.1K | uuid, license, generator, version, category, reasoning, m... |
| chat | 8 | 746.6K | uuid, license, generator, version, category, reasoning, m... |

**Data quality observations:**

- [code] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
- [math] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
- [stem] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
- [tool] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
- [chat] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

**Processing recommendation:**

- Chunked processing: ~256 chunks of 100,000 rows each

#### Nemotron-Post-Training-Dataset-v2

- **HuggingFace**: `nvidia/Nemotron-Post-Training-Dataset-v2`
- **Format**: parquet
- **Total rows**: 6,341,514
- **Text concatenation strategy**: Same as v1: flatten 'messages' list into role-prefixed text. Includes multilingual subsets (DE, ES, FR, IT, JA)—embeddings will capture cross-lingual semantics.
- **Preprocessing notes**: No tool_calls in v2 schema. Some shards have 'metadata' column (string)—ignore for embedding. Handle non-Latin scripts (JA) carefully.

**Sub-path breakdown:**

| Sub-path | Files | Rows | Schema Fields |
|----------|-------|------|---------------|
| chat | 12 | 627.7K | uuid, license, generator, version, category, reasoning, m... |
| code | 2 | 175.0K | uuid, license, generator, version, category, reasoning, m... |
| math | 2 | 239.5K | uuid, license, generator, version, category, reasoning, m... |
| stem | 2 | 355.0K | uuid, license, generator, version, category, reasoning, m... |
| multilingual | 1 | 100 | uuid, license, generator, version, category, reasoning, m... |
| multilingual_de | 38 | 1.0M | uuid, license, generator, version, category, reasoning, m... |
| multilingual_es | 33 | 935.7K | uuid, license, generator, version, category, reasoning, m... |
| multilingual_fr | 37 | 1.0M | uuid, license, generator, version, category, reasoning, m... |
| multilingual_it | 38 | 1.0M | uuid, license, generator, version, category, reasoning, m... |
| multilingual_ja | 37 | 975.2K | uuid, license, generator, version, category, reasoning, m... |

**Data quality observations:**

- [chat] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
- [code] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
- [math] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
- [stem] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
- [multilingual] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
- [multilingual_de] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
- [multilingual_es] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
- [multilingual_fr] Error sampling data: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

**Processing recommendation:**

- Chunked processing: ~63 chunks of 100,000 rows each

#### Nemotron-3-Nano-RL-Training-Blend

- **HuggingFace**: `nvidia/Nemotron-3-Nano-RL-Training-Blend`
- **Format**: jsonl
- **Total rows**: 93,244
- **Text concatenation strategy**: Extract 'input' messages from responses_create_params dict, flatten into role-prefixed text. Optionally append ground_truth tool calls as structured text.
- **Preprocessing notes**: Deep-extract nested dict: responses_create_params['input'] contains the message list. Ground truth is a list of tool-call dicts.
- **Text length stats** (sampled): min=0 / median=152 / mean=301 / max=3.2K tokens

**Sub-path breakdown:**

| Sub-path | Files | Rows | Schema Fields |
|----------|-------|------|---------------|
| train | 1 | 93.2K | id, responses_create_params, ground_truth, category, envi... |

**Data quality observations:**

- [train] Field 'id': 70% null values
- [train] Field 'responses_create_params': 24% null values
- [train] Field 'ground_truth': 89% null values
- [train] Field 'category': 89% null values
- [train] Field 'environment_name': 89% null values
- [train] Field 'pass_rate_total': 20% null values
- [train] Field 'pass_rate_passed': 20% null values
- [train] Field 'expected_answer': 79% null values

**Processing recommendation:**

- Single batch processing (total rows 93,244 < chunk size 100,000)

#### Nemotron-Science-v1

- **HuggingFace**: `nvidia/Nemotron-Science-v1`
- **Format**: jsonl
- **Total rows**: 226,334
- **Text concatenation strategy**: Flatten 'messages' list (user question + assistant answer). MCQ: multiple-choice science questions. RQA: research Q&A with boxed answers.
- **Preprocessing notes**: None needed—clean chat format.
- **Text length stats** (sampled): min=82 / median=443 / mean=503 / max=1.8K tokens

**Sub-path breakdown:**

| Sub-path | Files | Rows | Schema Fields |
|----------|-------|------|---------------|
| MCQ | 1 | 174.2K | uuid, messages, license, used_in, tools |
| RQA | 1 | 52.2K | uuid, messages, license, used_in, tools |

**Processing recommendation:**

- Chunked processing: ~2 chunks of 100,000 rows each

#### Nemotron-Instruction-Following-Chat-v1

- **HuggingFace**: `nvidia/Nemotron-Instruction-Following-Chat-v1`
- **Format**: jsonl
- **Total rows**: 430,978
- **Text concatenation strategy**: Flatten 'messages' into text. Includes system prompts with specific instruction-following constraints and structured output schemas.
- **Preprocessing notes**: System messages may contain XML/JSON schemas for structured outputs. Keep them as-is for semantic richness.
- **Text length stats** (sampled): min=50 / median=2.1K / mean=3.0K / max=22.0K tokens

**Sub-path breakdown:**

| Sub-path | Files | Rows | Schema Fields |
|----------|-------|------|---------------|
| chat_if | 1 | 426.0K | uuid, messages, license, used_in, tools, reasoning, capab... |
| structured_outputs | 1 | 5.0K | uuid, messages, license, used_in, tools |

**Processing recommendation:**

- Chunked processing: ~4 chunks of 100,000 rows each

#### Nemotron-Math-Proofs-v1

- **HuggingFace**: `nvidia/Nemotron-Math-Proofs-v1`
- **Format**: jsonl
- **Total rows**: 1,376,663
- **Text concatenation strategy**: Concatenate: 'Problem: {problem}\n\nFormal Statement (Lean 4):\n{lean_header}\n{formal_statement}'. Combines natural-language math with formal verification code.
- **Preprocessing notes**: Many fields (url, user_name, sft_line_number) are None—skip them. The 'messages' field is often empty; use problem + formal_statement.
- **Text length stats** (sampled): min=30 / median=1.2K / mean=2.0K / max=18.0K tokens

**Sub-path breakdown:**

| Sub-path | Files | Rows | Schema Fields |
|----------|-------|------|---------------|
| lean | 1 | 1.4M | problem, source, formal_statement, lean_header, url, user... |

**Data quality observations:**

- [lean] Field 'url': 13% null values
- [lean] Field 'user_name': 13% null values
- [lean] Field 'user_url': 20% null values
- [lean] Field 'sft_line_number': 33% null values

**Processing recommendation:**

- Chunked processing: ~13 chunks of 100,000 rows each

#### Nemotron-Agentic-v1

- **HuggingFace**: `nvidia/Nemotron-Agentic-v1`
- **Format**: jsonl
- **Total rows**: 335,122
- **Text concatenation strategy**: Flatten 'messages' into text. Prepend serialized 'tools' definitions (function name + description) as context. This captures the agentic task structure: available tools + conversation.
- **Preprocessing notes**: Serialize tool definitions to compact text: 'Tool: {name} - {description}'. Skip detailed parameter schemas.
- **Text length stats** (sampled): min=52 / median=1.6K / mean=1.5K / max=6.4K tokens

**Sub-path breakdown:**

| Sub-path | Files | Rows | Schema Fields |
|----------|-------|------|---------------|
| tool_calling | 1 | 316.1K | uuid, messages, license, used_in, tools |
| interactive_agent | 1 | 19.0K | uuid, messages, license, used_in, tools, reasoning |

**Processing recommendation:**

- Chunked processing: ~3 chunks of 100,000 rows each

#### Nemotron-Competitive-Programming-v1

- **HuggingFace**: `nvidia/Nemotron-Competitive-Programming-v1`
- **Format**: jsonl
- **Total rows**: 3,927,984
- **Text concatenation strategy**: Flatten 'messages' (user problem + assistant solution with code). Competitive programming: problem statement + reasoning + code solution.
- **Preprocessing notes**: Solutions contain code blocks—preserve formatting. Some messages have '-' as user content (placeholder); may need filtering.
- **Text length stats** (sampled): min=165 / median=1.2K / mean=1.2K / max=5.0K tokens

**Sub-path breakdown:**

| Sub-path | Files | Rows | Schema Fields |
|----------|-------|------|---------------|
| cpp_part0 | 1 | 466.0K | uuid, messages, license, used_in, tools, dataset, split, ... |
| cpp_part1 | 1 | 466.0K | uuid, messages, license, used_in, tools, dataset, split, ... |
| python_part0 | 1 | 910.6K | uuid, messages, license, used_in, tools, dataset, split, ... |
| python_part1 | 1 | 910.6K | uuid, messages, license, used_in, tools, dataset, split, ... |
| infinibyte_part0 | 1 | 587.3K | uuid, messages, license, used_in, tools |
| infinibyte_part1 | 1 | 587.3K | uuid, messages, license, used_in, tools |

**Processing recommendation:**

- Chunked processing: ~39 chunks of 100,000 rows each

#### Nemotron-Math-v2

- **HuggingFace**: `nvidia/Nemotron-Math-v2`
- **Format**: jsonl
- **Total rows**: 7,085,839
- **Text concatenation strategy**: Primary text: flatten 'messages' (user prompt with \\boxed{} instruction + assistant solution). Fallback: use 'problem' field directly. Difficulty tiers: low, medium, high.
- **Preprocessing notes**: Remove 'Solve the following math problem...' boilerplate prefix from user messages if desired. Keep \\boxed{answer} in text.
- **Text length stats** (sampled): min=53 / median=697 / mean=729 / max=3.9K tokens

**Sub-path breakdown:**

| Sub-path | Files | Rows | Schema Fields |
|----------|-------|------|---------------|
| low | 1 | 1.7M | problem, expected_answer, original_expected_answer, chang... |
| medium | 1 | 2.5M | problem, expected_answer, original_expected_answer, chang... |
| high_part0 | 1 | 695.9K | expected_answer, problem, original_expected_answer, chang... |
| high_part1 | 1 | 1.1M | expected_answer, problem, original_expected_answer, chang... |
| high_part2 | 1 | 1.1M | problem, expected_answer, original_expected_answer, chang... |

**Data quality observations:**

- [low] Field 'original_expected_answer': 100% null values
- [high_part0] Field 'url': 100% null values
- [high_part0] Field 'user_name': 100% null values
- [high_part0] Field 'user_url': 100% null values
- [high_part1] Field 'original_expected_answer': 100% null values
- [high_part1] Field 'url': 100% null values
- [high_part1] Field 'user_name': 100% null values
- [high_part1] Field 'user_url': 100% null values

**Processing recommendation:**

- Chunked processing: ~70 chunks of 100,000 rows each

#### Nemotron-SWE-v1

- **HuggingFace**: `nvidia/Nemotron-SWE-v1`
- **Format**: jsonl
- **Total rows**: 51,029
- **Text concatenation strategy**: Software engineering agent data: flatten long multi-turn 'messages' (system prompt with agent instructions + user bug report + assistant tool calls + tool outputs). Very long conversations.
- **Preprocessing notes**: WARNING: Messages can be extremely long (>100k chars). May need truncation to MODEL_MAX_TOKENS. System prompt is ~2k chars of agent instructions—consider summarizing or truncating.
- **Text length stats** (sampled): min=13.5K / median=34.0K / mean=36.7K / max=106.8K tokens
- **Texts exceeding MODEL_MAX_TOKENS (32,768)**: 54.9% of sampled records

**Sub-path breakdown:**

| Sub-path | Files | Rows | Schema Fields |
|----------|-------|------|---------------|
| r2e_gym | 1 | 51.0K | uuid, messages, license, used_in, tools, dataset, repo |

**Processing recommendation:**

- Single batch processing (total rows 51,029 < chunk size 100,000)
- **WARNING**: 54.9% of texts exceed MODEL_MAX_TOKENS—truncation required
- Consider chunking long texts into overlapping windows (max observed: 106.8K tokens)

### Pretraining Datasets

---

## Processing Recommendations by Category

### Post-Training Datasets

Post-training datasets use **chat/instruction format** with `messages` lists. The primary text concatenation strategy is to flatten multi-turn conversations into role-prefixed text blocks.

**Common approach:**
1. Extract `messages` list from each record
2. Format as `Role: content` with double-newline separators
3. For datasets with `input`/`output` fields (Llama-Nemotron), combine both
4. For agentic datasets, prepend tool definitions as context
5. Truncate to MODEL_MAX_TOKENS if text exceeds limit

**Special cases:**
- **Math-Proofs-v1**: Use `problem` + `formal_statement` (Lean 4 code) instead of empty `messages`
- **Math-v2**: Use `messages` with `problem` as fallback
- **SWE-v1**: Very long agent conversations—truncation essential
- **Competitive-Programming**: Contains code blocks—preserve formatting
- **Nano-RL-Blend**: Nested dict structure requires deep extraction

### Pretraining Datasets

Pretraining datasets use **raw text format** with a simple `text` column. No concatenation needed—use the text column directly.

**Common approach:**
1. Read `text` column directly from parquet files
2. Skip `Code-Metadata` subset (no text column—metadata only)
3. For `CC-Translated-Diverse-QA`, note the `language` column for filtering
4. The sample dataset is small (~27K rows total); full pretraining sets are TB-scale

---

## Appendix: Schema Reference

### Llama-Nemotron-Post-Training-Dataset

**SFT_chat:**
```
  input: list
  output: str
  category: str
  license: str
  reasoning: str
  generator: str
  used_in_training: str
  version: str
  system_prompt: str
```

**SFT_code_v1:**
```
  input: list
  output: str
  category: str
  license: str
  reasoning: str
  generator: str
  used_in_training: str
  version: str
  system_prompt: str
```

**SFT_code_v1.1:**
```
  input: list
  output: str
  category: str
  license: str
  reasoning: str
  generator: str
  used_in_training: str
  version: str
  system_prompt: str
```

**SFT_math_v1:**
```
  input: list
  output: str
  category: str
  license: str
  reasoning: str
  generator: str
  version: str
  system_prompt: str
  used_in_training: str
```

**SFT_math_v1.1:**
```
  input: list
  output: str
  category: str
  license: str
  reasoning: str
  generator: str
  used_in_training: str
  version: str
  system_prompt: str
```

**SFT_safety:**
```
  input: list
  output: str
  category: str
  generator: str
  license: str
  reasoning: str
  used_in_training: str
  version: str
  system_prompt: str
```

**SFT_science:**
```
  input: list
  output: str
  category: str
  license: str
  reasoning: str
  generator: str
  used_in_training: str
  version: str
  system_prompt: str
```

**RL_instruction_following:**
```
  input: list
  args: dict
  category: str
  license: str
  reasoning: str
  used_in_training: str
  version: str
  system_prompt: str
```

**train_when2call_sft:**
```
  tools: list
  messages: list
```

**train_when2call_pref:**
```
  tools: list
  messages: list
  chosen_response: dict
  rejected_response: dict
```

### Nemotron-Post-Training-Dataset-v1

**code:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string, tool_calls: list<element: struct<id: string, type: string, function: struct<name: string, arguments: string>>>>>
  metadata: string
```

**math:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string, tool_calls: list<element: struct<id: string, type: string, function: struct<name: string, arguments: string>>>>>
  metadata: string
```

**stem:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string, tool_calls: list<element: struct<id: string, type: string, function: struct<name: string, arguments: string>>>>>
  metadata: string
```

**tool:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string, tool_calls: list<element: struct<id: string, type: string, function: struct<name: string, arguments: string>>>>>
  metadata: string
```

**chat:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string, tool_calls: list<element: struct<id: string, type: string, function: struct<name: string, arguments: string>>>>>
  metadata: string
```

### Nemotron-Post-Training-Dataset-v2

**chat:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string>>
```

**code:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string>>
```

**math:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string>>
```

**stem:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string>>
```

**multilingual:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string>>
  metadata: string
```

**multilingual_de:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string>>
```

**multilingual_es:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string>>
```

**multilingual_fr:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string>>
```

**multilingual_it:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string>>
```

**multilingual_ja:**
```
  uuid: string
  license: string
  generator: string
  version: string
  category: string
  reasoning: string
  messages: list<element: struct<role: string, content: string>>
```

### Nemotron-3-Nano-RL-Training-Blend

**train:**
```
  id: int
  responses_create_params: dict
  ground_truth: list
  category: str
  environment_name: str
  agent_ref: dict
  pass_rate: float
  pass_rate_total: int
  pass_rate_passed: int
  dataset: str
  expected_answer: str
  uuid: str
  options: list
  reward_profiles: list
  template_metadata: dict
  instruction_id_list: list
  prompt: str
  kwargs: list
  _hf_placeholder: dict
  verifier_metadata: dict
  hash_id: str
  source: str
```

### Nemotron-Science-v1

**MCQ:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
```

**RQA:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
```

### Nemotron-Instruction-Following-Chat-v1

**chat_if:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
  reasoning: str
  capability_target: str
```

**structured_outputs:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
```

### Nemotron-Math-Proofs-v1

**lean:**
```
  problem: str
  source: str
  formal_statement: str
  lean_header: str
  url: NoneType
  user_name: NoneType
  user_url: NoneType
  sft_line_number: NoneType
  messages: list
  uuid: str
  used_in: list
  tools: list
  license: str
```

### Nemotron-Agentic-v1

**tool_calling:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
```

**interactive_agent:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
  reasoning: str
```

### Nemotron-Competitive-Programming-v1

**cpp_part0:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
  dataset: str
  split: str
  index: str
  source: str
  difficulty: str
  question_id: str
```

**cpp_part1:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
  dataset: str
  split: str
  index: str
  source: str
  difficulty: str
  question_id: str
```

**python_part0:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
  dataset: str
  split: str
  index: str
  source: str
  difficulty: str
  question_id: str
```

**python_part1:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
  dataset: str
  split: str
  index: str
  source: str
  difficulty: str
  question_id: str
```

**infinibyte_part0:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
```

**infinibyte_part1:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
```

### Nemotron-Math-v2

**low:**
```
  problem: str
  expected_answer: str
  original_expected_answer: NoneType
  changed_answer_to_majority: bool
  data_source: str
  messages: list
  metadata: dict
  license: str
  used_in: list
  uuid: str
  url: str
  user_url: str
  user_name: str
  tools: list
```

**medium:**
```
  problem: str
  expected_answer: str
  original_expected_answer: str
  changed_answer_to_majority: bool
  data_source: str
  messages: list
  metadata: dict
  license: str
  used_in: list
  uuid: str
  url: str
  user_url: str
  user_name: str
  tools: list
```

**high_part0:**
```
  expected_answer: str
  problem: str
  original_expected_answer: str
  changed_answer_to_majority: bool
  data_source: str
  messages: list
  used_in: list
  metadata: dict
  license: str
  tools: list
  url: NoneType
  user_name: NoneType
  user_url: NoneType
```

**high_part1:**
```
  expected_answer: str
  problem: str
  original_expected_answer: NoneType
  changed_answer_to_majority: bool
  data_source: str
  messages: list
  tools: list
  used_in: list
  metadata: dict
  license: str
  url: NoneType
  user_name: NoneType
  user_url: NoneType
```

**high_part2:**
```
  problem: str
  expected_answer: str
  original_expected_answer: str
  changed_answer_to_majority: bool
  data_source: str
  messages: list
  url: str
  user_url: str
  user_name: str
  used_in: list
  license: str
  metadata: dict
  tools: list
```

### Nemotron-SWE-v1

**r2e_gym:**
```
  uuid: str
  messages: list
  license: str
  used_in: list
  tools: list
  dataset: str
  repo: str
```
