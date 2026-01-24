# Nemotron Datasets Statistics

Generated from `/raid/datasets` on 2026-01-24.

## Summary

| Dataset | Format | Total Rows | Splits | Strategy |
|---------|--------|------------|--------|----------|
| Llama-Nemotron-Post-Training-Dataset | arrow | 32,955,418 | 5 | input_output |
| Nemotron-3-Nano-RL-Training-Blend | jsonl | 93,244 | 1 | combine_columns |
| Nemotron-Agentic-v1 | jsonl | 335,122 | 2 | concatenate_messages |
| Nemotron-Competitive-Programming-v1 | jsonl | 3,927,984 | 6 | concatenate_messages |
| Nemotron-Instruction-Following-Chat-v1 | arrow | 430,978 | 2 | concatenate_messages |
| Nemotron-Math-Proofs-v1 | arrow | 1,376,663 | 1 | combine_columns |
| Nemotron-Math-v2 | jsonl | 7,085,839 | 5 | concatenate_messages |
| Nemotron-Post-Training-Dataset-v1 | arrow | 25,659,642 | 5 | concatenate_messages |
| Nemotron-Post-Training-Dataset-v2 | arrow | 6,341,414 | 9 | concatenate_messages |
| Nemotron-Pretraining-Dataset-sample | parquet | 27,706 | 10 | direct_text |
| Nemotron-SWE-v1 | arrow | 51,029 | 1 | concatenate_messages |
| Nemotron-Science-v1 | arrow | 226,334 | 2 | concatenate_messages |

**Total: 78,511,373 rows across 12 datasets**

---

## Detailed Statistics

### nvidia/Llama-Nemotron-Post-Training-Dataset

- **Format**: Arrow (HuggingFace)
- **Total Rows**: 32,955,418
- **Strategy**: `input_output`

**Columns**:
- `input` (text)
- `output` (text)
- `category`
- `license`
- `reasoning`
- `generator`
- `used_in_training`
- `version`
- `system_prompt`

**Splits**:
| Split | Rows |
|-------|------|
| chat | 39,792 |
| code | 10,108,883 |
| math | 22,066,397 |
| safety | 31,426 |
| science | 708,920 |

---

### nvidia/Nemotron-3-Nano-RL-Training-Blend

- **Format**: JSONL
- **Total Rows**: 93,244
- **Strategy**: `combine_columns`

**Columns**:
- `id`
- `responses_create_params` (text)
- `ground_truth`
- `category`
- `environment_name`
- `agent_ref`
- `pass_rate`
- `pass_rate_total`
- `pass_rate_passed`
- `dataset`

**Splits**:
| Split | Rows |
|-------|------|
| train | 93,244 |

---

### nvidia/Nemotron-Agentic-v1

- **Format**: JSONL
- **Total Rows**: 335,122
- **Strategy**: `concatenate_messages`

**Columns**:
- `uuid`
- `messages` (text)
- `license`
- `used_in`
- `tools`
- `reasoning`

**Splits**:
| Split | Rows |
|-------|------|
| interactive_agent | ~167,561 |
| tool_calling | ~167,561 |

---

### nvidia/Nemotron-Competitive-Programming-v1

- **Format**: JSONL
- **Total Rows**: 3,927,984
- **Strategy**: `concatenate_messages`

**Columns**:
- `uuid`
- `messages` (text)
- `license`
- `used_in`
- `tools`
- `dataset`
- `split`
- `index`
- `source`
- `difficulty`
- `question_id`

**Splits**:
| Split | Rows |
|-------|------|
| competitive_coding_cpp.part_00 | ~654,664 |
| competitive_coding_cpp.part_01 | ~654,664 |
| competitive_coding_python.part_00 | ~654,664 |
| competitive_coding_python.part_01 | ~654,664 |
| infinibyte.part_00 | ~654,664 |
| infinibyte.part_01 | ~654,664 |

---

### nvidia/Nemotron-Instruction-Following-Chat-v1

- **Format**: Arrow (HuggingFace)
- **Total Rows**: 430,978
- **Strategy**: `concatenate_messages`

**Columns**:
- `uuid`
- `messages` (text)
- `license`
- `used_in`
- `tools`
- `reasoning`
- `capability_target`

**Splits**:
| Split | Rows |
|-------|------|
| chat_if | 426,009 |
| structured_outputs | 4,969 |

---

### nvidia/Nemotron-Math-Proofs-v1

- **Format**: Arrow (HuggingFace)
- **Total Rows**: 1,376,663
- **Strategy**: `combine_columns`

**Columns**:
- `problem` (text)
- `source`
- `formal_statement` (text)
- `lean_header` (text)
- `url`
- `user_name`
- `user_url`
- `sft_line_number`
- `messages`
- `uuid`
- `used_in`
- `tools`
- `license`

**Splits**:
| Split | Rows |
|-------|------|
| lean | 1,376,663 |

---

### nvidia/Nemotron-Math-v2

- **Format**: JSONL
- **Total Rows**: 7,085,839
- **Strategy**: `concatenate_messages`

**Columns**:
- `expected_answer`
- `problem`
- `original_expected_answer`
- `changed_answer_to_majority`
- `data_source`
- `messages` (text)
- `used_in`
- `metadata`
- `license`

**Splits**:
| Split | Rows |
|-------|------|
| high.part_00 | ~1,417,168 |
| high.part_01 | ~1,417,168 |
| high.part_02 | ~1,417,168 |
| low | ~1,417,168 |
| medium | ~1,417,167 |

---

### nvidia/Nemotron-Post-Training-Dataset-v1

- **Format**: Arrow (HuggingFace)
- **Total Rows**: 25,659,642
- **Strategy**: `concatenate_messages`

**Columns**:
- `uuid`
- `license`
- `generator`
- `version`
- `category`
- `reasoning`
- `messages` (text)
- `metadata`

**Splits**:
| Split | Rows |
|-------|------|
| chat | 746,622 |
| code | 1,896,395 |
| math | 2,044,407 |
| stem | 20,662,167 |
| tool_calling | 310,051 |

---

### nvidia/Nemotron-Post-Training-Dataset-v2

- **Format**: Arrow (HuggingFace)
- **Total Rows**: 6,341,414
- **Strategy**: `concatenate_messages`

**Columns**:
- `uuid`
- `license`
- `generator`
- `version`
- `category`
- `reasoning`
- `messages` (text)

**Splits**:
| Split | Rows |
|-------|------|
| chat | 627,720 |
| code | 175,000 |
| math | 239,467 |
| multilingual_de | 1,015,314 |
| multilingual_es | 935,704 |
| multilingual_fr | 1,001,504 |
| multilingual_it | 1,016,503 |
| multilingual_ja | 975,202 |
| stem | 355,000 |

---

### nvidia/Nemotron-Pretraining-Dataset-sample

- **Format**: Parquet
- **Total Rows**: 27,706
- **Strategy**: `direct_text`

**Columns**:
- `id`
- `text` (text)

**Splits**:
| Split | Rows |
|-------|------|
| Nemotron-CC-Diverse-QA | ~2,771 |
| Nemotron-CC-High-Quality | ~2,771 |
| Nemotron-CC-High-Quality-Synthetic | ~2,771 |
| Nemotron-CC-MATH | ~2,771 |
| Nemotron-CC-Translated-Diverse-QA | ~2,771 |
| Nemotron-Code-Metadata | ~2,771 |
| Nemotron-SFT-Code | ~2,771 |
| Nemotron-SFT-General | ~2,771 |
| Nemotron-SFT-MATH | ~2,771 |
| Nemotron-Synthetic-Code | ~2,771 |

---

### nvidia/Nemotron-SWE-v1

- **Format**: Arrow (HuggingFace)
- **Total Rows**: 51,029
- **Strategy**: `concatenate_messages`

**Columns**:
- `uuid`
- `messages` (text)
- `license`
- `used_in`
- `tools`
- `dataset`
- `repo`

**Splits**:
| Split | Rows |
|-------|------|
| r2e_gym | 51,029 |

---

### nvidia/Nemotron-Science-v1

- **Format**: Arrow (HuggingFace)
- **Total Rows**: 226,334
- **Strategy**: `concatenate_messages`

**Columns**:
- `uuid`
- `messages` (text)
- `license`
- `used_in`
- `tools`

**Splits**:
| Split | Rows |
|-------|------|
| MCQ | 174,155 |
| RQA | 52,179 |

---

## Embedding Extraction Strategies

| Strategy | Description | Datasets |
|----------|-------------|----------|
| `concatenate_messages` | Concatenate all messages with `role: content` format | Most post-training datasets |
| `input_output` | Combine input messages + output | Llama-Nemotron-Post-Training-Dataset |
| `combine_columns` | Combine multiple text columns | Math-Proofs-v1, RL-Training-Blend |
| `direct_text` | Direct text extraction from `text` column | Pretraining-Dataset-sample |

---

## Model Configuration

- **Model**: `nvidia/llama-embed-nemotron-8b`
- **Embedding Size**: 4096
- **Max Tokens**: 32,768 (32K)
- **Default Batch Size**: 1
- **Default Dtype**: bfloat16
