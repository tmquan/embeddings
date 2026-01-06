2026-01-05 22:24:14,010 - INFO - Starting Nemotron Datasets Exploration
2026-01-05 22:24:14,010 - INFO - Datasets directory: /raid/datasets
2026-01-05 22:24:14,010 - INFO - Exploring 6 datasets

2026-01-05 22:24:14,010 - INFO - 
[1/6] Processing: nvidia/Llama-Nemotron-Post-Training-Dataset
2026-01-05 22:24:14,010 - INFO - 
================================================================================
2026-01-05 22:24:14,010 - INFO - Exploring: nvidia/Llama-Nemotron-Post-Training-Dataset
2026-01-05 22:24:14,010 - INFO - ================================================================================
2026-01-05 22:24:15,076 - INFO - 
Split: code (10,108,883 rows)
2026-01-05 22:24:15,076 - INFO - Columns (9): ['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt']
2026-01-05 22:24:15,080 - INFO - 
Column Details:
2026-01-05 22:24:15,080 - INFO -   - input
2026-01-05 22:24:15,080 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:15,080 - INFO -     Value types: ['list']
2026-01-05 22:24:15,080 - INFO -     Is list of dicts: True
2026-01-05 22:24:15,080 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:15,080 - INFO -   - output
2026-01-05 22:24:15,080 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,080 - INFO -     Value types: ['str']
2026-01-05 22:24:15,080 - INFO -     Is text: True
2026-01-05 22:24:15,080 - INFO -     Sample: <think>
Okay, I need to solve this problem where I have to split a unique array s into two arrays a and b such that both a and b are almost unique. Almost unique means that after removing at most n/2 ...
2026-01-05 22:24:15,080 - INFO -   - category
2026-01-05 22:24:15,080 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,080 - INFO -     Value types: ['str']
2026-01-05 22:24:15,080 - INFO -     Is text: True
2026-01-05 22:24:15,080 - INFO -     Sample: code
2026-01-05 22:24:15,080 - INFO -   - license
2026-01-05 22:24:15,080 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,080 - INFO -     Value types: ['str']
2026-01-05 22:24:15,080 - INFO -     Is text: True
2026-01-05 22:24:15,080 - INFO -     Sample: cc-by-4.0
2026-01-05 22:24:15,080 - INFO -   - reasoning
2026-01-05 22:24:15,080 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,080 - INFO -     Value types: ['str']
2026-01-05 22:24:15,080 - INFO -     Is text: True
2026-01-05 22:24:15,080 - INFO -     Sample: on
2026-01-05 22:24:15,080 - INFO -   - generator
2026-01-05 22:24:15,080 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,080 - INFO -     Value types: ['str']
2026-01-05 22:24:15,080 - INFO -     Is text: True
2026-01-05 22:24:15,080 - INFO -     Sample: DeepSeek-R1
2026-01-05 22:24:15,080 - INFO -   - used_in_training
2026-01-05 22:24:15,080 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,080 - INFO -     Value types: ['str']
2026-01-05 22:24:15,080 - INFO -     Is text: True
2026-01-05 22:24:15,080 - INFO -     Sample: Ultra, Nano
2026-01-05 22:24:15,080 - INFO -   - version
2026-01-05 22:24:15,080 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,080 - INFO -     Value types: ['str']
2026-01-05 22:24:15,080 - INFO -     Is text: True
2026-01-05 22:24:15,080 - INFO -     Sample: v1.1
2026-01-05 22:24:15,080 - INFO -   - system_prompt
2026-01-05 22:24:15,080 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,080 - INFO -     Value types: ['str']
2026-01-05 22:24:15,080 - INFO -     Is text: True
2026-01-05 22:24:15,080 - INFO -     Sample: detailed thinking on
2026-01-05 22:24:15,080 - INFO - 
Recommended Embedding Columns: ['output', 'system_prompt']
2026-01-05 22:24:15,080 - INFO - 
Split: math (22,066,397 rows)
2026-01-05 22:24:15,080 - INFO - Columns (9): ['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt']
2026-01-05 22:24:15,083 - INFO - 
Column Details:
2026-01-05 22:24:15,083 - INFO -   - input
2026-01-05 22:24:15,083 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:15,083 - INFO -     Value types: ['list']
2026-01-05 22:24:15,083 - INFO -     Is list of dicts: True
2026-01-05 22:24:15,083 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:15,083 - INFO -   - output
2026-01-05 22:24:15,083 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,083 - INFO -     Value types: ['str']
2026-01-05 22:24:15,083 - INFO -     Is text: True
2026-01-05 22:24:15,083 - INFO -     Sample: <think>
Okay, so I need to sketch the graph of the function g(x) = ln(x⁵ + 5) - x³. Hmm, let me think about how to approach this. I remember that sketching functions usually involves finding key point...
2026-01-05 22:24:15,083 - INFO -   - category
2026-01-05 22:24:15,083 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,083 - INFO -     Value types: ['str']
2026-01-05 22:24:15,083 - INFO -     Is text: True
2026-01-05 22:24:15,083 - INFO -     Sample: math
2026-01-05 22:24:15,083 - INFO -   - license
2026-01-05 22:24:15,083 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,083 - INFO -     Value types: ['str']
2026-01-05 22:24:15,083 - INFO -     Is text: True
2026-01-05 22:24:15,083 - INFO -     Sample: cc-by-4.0
2026-01-05 22:24:15,083 - INFO -   - reasoning
2026-01-05 22:24:15,083 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,083 - INFO -     Value types: ['str']
2026-01-05 22:24:15,083 - INFO -     Is text: True
2026-01-05 22:24:15,083 - INFO -     Sample: on
2026-01-05 22:24:15,083 - INFO -   - generator
2026-01-05 22:24:15,083 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,083 - INFO -     Value types: ['str']
2026-01-05 22:24:15,083 - INFO -     Is text: True
2026-01-05 22:24:15,083 - INFO -     Sample: Qwen-2.5-32B-Instruct, DeepSeek-R1
2026-01-05 22:24:15,084 - INFO -   - used_in_training
2026-01-05 22:24:15,084 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,084 - INFO -     Value types: ['str']
2026-01-05 22:24:15,084 - INFO -     Is text: True
2026-01-05 22:24:15,084 - INFO -     Sample: Ultra, Super, Nano
2026-01-05 22:24:15,084 - INFO -   - version
2026-01-05 22:24:15,084 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,084 - INFO -     Value types: ['str']
2026-01-05 22:24:15,084 - INFO -     Is text: True
2026-01-05 22:24:15,084 - INFO -     Sample: v1.1
2026-01-05 22:24:15,084 - INFO -   - system_prompt
2026-01-05 22:24:15,084 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,084 - INFO -     Value types: ['str']
2026-01-05 22:24:15,084 - INFO -     Is text: True
2026-01-05 22:24:15,084 - INFO -     Sample: detailed thinking on
2026-01-05 22:24:15,084 - INFO - 
Recommended Embedding Columns: ['output', 'system_prompt']
2026-01-05 22:24:15,084 - INFO - 
Split: science (708,920 rows)
2026-01-05 22:24:15,084 - INFO - Columns (9): ['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt']
2026-01-05 22:24:15,087 - INFO - 
Column Details:
2026-01-05 22:24:15,087 - INFO -   - input
2026-01-05 22:24:15,087 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:15,087 - INFO -     Value types: ['list']
2026-01-05 22:24:15,087 - INFO -     Is list of dicts: True
2026-01-05 22:24:15,087 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:15,087 - INFO -   - output
2026-01-05 22:24:15,087 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,087 - INFO -     Value types: ['str']
2026-01-05 22:24:15,087 - INFO -     Is text: True
2026-01-05 22:24:15,087 - INFO -     Sample: <think>
Okay, let me try to work through this question. The question is asking about the primary benefit of no-till farming practices. Hmm, no-till farming. So first, I need to remember what no-till i...
2026-01-05 22:24:15,087 - INFO -   - category
2026-01-05 22:24:15,087 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,087 - INFO -     Value types: ['str']
2026-01-05 22:24:15,087 - INFO -     Is text: True
2026-01-05 22:24:15,087 - INFO -     Sample: science
2026-01-05 22:24:15,087 - INFO -   - license
2026-01-05 22:24:15,087 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,087 - INFO -     Value types: ['str']
2026-01-05 22:24:15,087 - INFO -     Is text: True
2026-01-05 22:24:15,087 - INFO -     Sample: cc-by-4.0
2026-01-05 22:24:15,087 - INFO -   - reasoning
2026-01-05 22:24:15,087 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,087 - INFO -     Value types: ['str']
2026-01-05 22:24:15,087 - INFO -     Is text: True
2026-01-05 22:24:15,087 - INFO -     Sample: on
2026-01-05 22:24:15,087 - INFO -   - generator
2026-01-05 22:24:15,087 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,087 - INFO -     Value types: ['str']
2026-01-05 22:24:15,087 - INFO -     Is text: True
2026-01-05 22:24:15,087 - INFO -     Sample: DeepSeek-R1, Qwen-2.5-72B-Instruct
2026-01-05 22:24:15,087 - INFO -   - used_in_training
2026-01-05 22:24:15,087 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,087 - INFO -     Value types: ['str']
2026-01-05 22:24:15,087 - INFO -     Is text: True
2026-01-05 22:24:15,087 - INFO -     Sample: Ultra, Nano
2026-01-05 22:24:15,087 - INFO -   - version
2026-01-05 22:24:15,087 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,087 - INFO -     Value types: ['str']
2026-01-05 22:24:15,087 - INFO -     Is text: True
2026-01-05 22:24:15,087 - INFO -     Sample: v1
2026-01-05 22:24:15,087 - INFO -   - system_prompt
2026-01-05 22:24:15,087 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,087 - INFO -     Value types: ['str']
2026-01-05 22:24:15,087 - INFO -     Is text: True
2026-01-05 22:24:15,087 - INFO -     Sample: detailed thinking on
2026-01-05 22:24:15,087 - INFO - 
Recommended Embedding Columns: ['output', 'system_prompt']
2026-01-05 22:24:15,087 - INFO - 
Split: chat (39,792 rows)
2026-01-05 22:24:15,087 - INFO - Columns (9): ['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt']
2026-01-05 22:24:15,090 - INFO - 
Column Details:
2026-01-05 22:24:15,090 - INFO -   - input
2026-01-05 22:24:15,090 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:15,090 - INFO -     Value types: ['list']
2026-01-05 22:24:15,090 - INFO -     Is list of dicts: True
2026-01-05 22:24:15,090 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:15,090 - INFO -   - output
2026-01-05 22:24:15,090 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,090 - INFO -     Value types: ['str']
2026-01-05 22:24:15,090 - INFO -     Is text: True
2026-01-05 22:24:15,090 - INFO -     Sample: Here's a comprehensive overview of an aluminium plant, covering various aspects from its definition to operational details. 

### Definition and Purpose

- **Definition**: An aluminium plant is an ind...
2026-01-05 22:24:15,090 - INFO -   - category
2026-01-05 22:24:15,090 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,090 - INFO -     Value types: ['str']
2026-01-05 22:24:15,090 - INFO -     Is text: True
2026-01-05 22:24:15,090 - INFO -     Sample: chat
2026-01-05 22:24:15,090 - INFO -   - license
2026-01-05 22:24:15,090 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,090 - INFO -     Value types: ['str']
2026-01-05 22:24:15,090 - INFO -     Is text: True
2026-01-05 22:24:15,090 - INFO -     Sample: cc-by-4.0
2026-01-05 22:24:15,090 - INFO -   - reasoning
2026-01-05 22:24:15,090 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,090 - INFO -     Value types: ['str']
2026-01-05 22:24:15,090 - INFO -     Is text: True
2026-01-05 22:24:15,090 - INFO -     Sample: off
2026-01-05 22:24:15,090 - INFO -   - generator
2026-01-05 22:24:15,090 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,090 - INFO -     Value types: ['str']
2026-01-05 22:24:15,090 - INFO -     Is text: True
2026-01-05 22:24:15,090 - INFO -     Sample: Llama-3.1-Nemotron-70B-Instruct, ITS models
2026-01-05 22:24:15,090 - INFO -   - used_in_training
2026-01-05 22:24:15,090 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,090 - INFO -     Value types: ['str']
2026-01-05 22:24:15,090 - INFO -     Is text: True
2026-01-05 22:24:15,090 - INFO -     Sample: Ultra, Super, Nano
2026-01-05 22:24:15,090 - INFO -   - version
2026-01-05 22:24:15,090 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,090 - INFO -     Value types: ['str']
2026-01-05 22:24:15,090 - INFO -     Is text: True
2026-01-05 22:24:15,090 - INFO -     Sample: v1
2026-01-05 22:24:15,090 - INFO -   - system_prompt
2026-01-05 22:24:15,090 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,090 - INFO -     Value types: ['str']
2026-01-05 22:24:15,090 - INFO -     Is text: True
2026-01-05 22:24:15,090 - INFO -     Sample: detailed thinking off
2026-01-05 22:24:15,090 - INFO - 
Recommended Embedding Columns: ['output', 'system_prompt']
2026-01-05 22:24:15,090 - INFO - 
Split: safety (31,426 rows)
2026-01-05 22:24:15,090 - INFO - Columns (9): ['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt']
2026-01-05 22:24:15,093 - INFO - 
Column Details:
2026-01-05 22:24:15,093 - INFO -   - input
2026-01-05 22:24:15,093 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:15,093 - INFO -     Value types: ['list']
2026-01-05 22:24:15,093 - INFO -     Is list of dicts: True
2026-01-05 22:24:15,093 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:15,093 - INFO -   - output
2026-01-05 22:24:15,093 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,093 - INFO -     Value types: ['str']
2026-01-05 22:24:15,093 - INFO -     Is text: True
2026-01-05 22:24:15,093 - INFO -     Sample: I appreciate your curiosity and willingness to learn about potentially offensive or harmful language. The term "gypsy" is a racial slur, as it has been historically used to stereotype and discriminate...
2026-01-05 22:24:15,093 - INFO -   - category
2026-01-05 22:24:15,093 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,093 - INFO -     Value types: ['str']
2026-01-05 22:24:15,093 - INFO -     Is text: True
2026-01-05 22:24:15,093 - INFO -     Sample: safety
2026-01-05 22:24:15,093 - INFO -   - license
2026-01-05 22:24:15,093 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,093 - INFO -     Value types: ['str']
2026-01-05 22:24:15,093 - INFO -     Is text: True
2026-01-05 22:24:15,093 - INFO -     Sample: cc-by-4.0
2026-01-05 22:24:15,093 - INFO -   - reasoning
2026-01-05 22:24:15,093 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,093 - INFO -     Value types: ['str']
2026-01-05 22:24:15,093 - INFO -     Is text: True
2026-01-05 22:24:15,093 - INFO -     Sample: off
2026-01-05 22:24:15,093 - INFO -   - generator
2026-01-05 22:24:15,093 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,093 - INFO -     Value types: ['str']
2026-01-05 22:24:15,093 - INFO -     Is text: True
2026-01-05 22:24:15,093 - INFO -     Sample: Mixtral-8x22B-Instruct-v0.1
2026-01-05 22:24:15,093 - INFO -   - used_in_training
2026-01-05 22:24:15,093 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,093 - INFO -     Value types: ['str']
2026-01-05 22:24:15,093 - INFO -     Is text: True
2026-01-05 22:24:15,093 - INFO -     Sample: Ultra, Super, Nano
2026-01-05 22:24:15,093 - INFO -   - version
2026-01-05 22:24:15,093 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,093 - INFO -     Value types: ['str']
2026-01-05 22:24:15,093 - INFO -     Is text: True
2026-01-05 22:24:15,093 - INFO -     Sample: v1
2026-01-05 22:24:15,093 - INFO -   - system_prompt
2026-01-05 22:24:15,093 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:15,093 - INFO -     Value types: ['str']
2026-01-05 22:24:15,093 - INFO -     Is text: True
2026-01-05 22:24:15,093 - INFO -     Sample: detailed thinking off
2026-01-05 22:24:15,093 - INFO - 
Recommended Embedding Columns: ['output', 'system_prompt']
2026-01-05 22:24:15,093 - INFO - 
================================================================================
2026-01-05 22:24:15,093 - INFO - Dataset Summary: nvidia/Llama-Nemotron-Post-Training-Dataset
2026-01-05 22:24:15,093 - INFO -   Total rows: 32,955,418
2026-01-05 22:24:15,093 - INFO -   Number of splits: 5
2026-01-05 22:24:15,093 - INFO -   All columns: ['category', 'generator', 'input', 'license', 'output', 'reasoning', 'system_prompt', 'used_in_training', 'version']
2026-01-05 22:24:15,093 - INFO -   Embedding columns: ['output', 'system_prompt']
2026-01-05 22:24:15,093 - INFO - ================================================================================
2026-01-05 22:24:15,274 - INFO - 
[2/6] Processing: nvidia/Nemotron-Post-Training-Dataset-v1
2026-01-05 22:24:15,274 - INFO - 
================================================================================
2026-01-05 22:24:15,274 - INFO - Exploring: nvidia/Nemotron-Post-Training-Dataset-v1
2026-01-05 22:24:15,274 - INFO - ================================================================================
2026-01-05 22:24:16,173 - INFO - 
Split: chat (746,622 rows)
2026-01-05 22:24:16,173 - INFO - Columns (8): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages', 'metadata']
2026-01-05 22:24:16,176 - INFO - 
Column Details:
2026-01-05 22:24:16,176 - INFO -   - uuid
2026-01-05 22:24:16,176 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,176 - INFO -     Value types: ['str']
2026-01-05 22:24:16,176 - INFO -     Is text: True
2026-01-05 22:24:16,176 - INFO -     Sample: 1b07b912-0135-4f23-b704-2ceea567f617
2026-01-05 22:24:16,176 - INFO -   - license
2026-01-05 22:24:16,176 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,176 - INFO -     Value types: ['str']
2026-01-05 22:24:16,176 - INFO -     Is text: True
2026-01-05 22:24:16,176 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,176 - INFO -   - generator
2026-01-05 22:24:16,176 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,176 - INFO -     Value types: ['str']
2026-01-05 22:24:16,176 - INFO -     Is text: True
2026-01-05 22:24:16,176 - INFO -     Sample: Qwen3-235B-A22B
2026-01-05 22:24:16,176 - INFO -   - version
2026-01-05 22:24:16,176 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,176 - INFO -     Value types: ['str']
2026-01-05 22:24:16,176 - INFO -     Is text: True
2026-01-05 22:24:16,176 - INFO -     Sample: v1
2026-01-05 22:24:16,176 - INFO -   - category
2026-01-05 22:24:16,176 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,176 - INFO -     Value types: ['str']
2026-01-05 22:24:16,176 - INFO -     Is text: True
2026-01-05 22:24:16,176 - INFO -     Sample: chat
2026-01-05 22:24:16,176 - INFO -   - reasoning
2026-01-05 22:24:16,176 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,176 - INFO -     Value types: ['str']
2026-01-05 22:24:16,176 - INFO -     Is text: True
2026-01-05 22:24:16,176 - INFO -     Sample: off
2026-01-05 22:24:16,176 - INFO -   - messages
2026-01-05 22:24:16,176 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None), 'tool_calls': [{'id': Value(dtype='string', id=None), 'type': Value(dtype='string', id=None), 'function': {'name': Value(dtype='string', id=None), 'arguments': Value(dtype='string', id=None)}}]}]
2026-01-05 22:24:16,176 - INFO -     Value types: ['list']
2026-01-05 22:24:16,176 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,176 - INFO -     Sample structure: ['role', 'content', 'tool_calls']
2026-01-05 22:24:16,177 - INFO -   - metadata
2026-01-05 22:24:16,177 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,177 - INFO -     Value types: ['str']
2026-01-05 22:24:16,177 - INFO -     Is text: True
2026-01-05 22:24:16,177 - INFO -     Sample: {"conversation_id": "8e31a022d01d49748f6053a8805dfbd2", "source": "https://huggingface.co/datasets/lmsys/lmsys-chat-1m"}
2026-01-05 22:24:16,177 - INFO - 
Message Structure:
2026-01-05 22:24:16,177 - INFO -   Roles: ['assistant', 'user']
2026-01-05 22:24:16,177 - INFO -   Message keys: ['content', 'role', 'tool_calls']
2026-01-05 22:24:16,177 - INFO -   Sample conversation length: 2 messages
2026-01-05 22:24:16,177 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,177 - INFO - 
Split: code (1,896,395 rows)
2026-01-05 22:24:16,177 - INFO - Columns (8): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages', 'metadata']
2026-01-05 22:24:16,180 - INFO - 
Column Details:
2026-01-05 22:24:16,180 - INFO -   - uuid
2026-01-05 22:24:16,180 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,180 - INFO -     Value types: ['str']
2026-01-05 22:24:16,180 - INFO -     Is text: True
2026-01-05 22:24:16,180 - INFO -     Sample: 7e914d4b-67d4-4e31-a466-65a5fc2dfe7e
2026-01-05 22:24:16,180 - INFO -   - license
2026-01-05 22:24:16,180 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,180 - INFO -     Value types: ['str']
2026-01-05 22:24:16,180 - INFO -     Is text: True
2026-01-05 22:24:16,180 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,180 - INFO -   - generator
2026-01-05 22:24:16,180 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,180 - INFO -     Value types: ['str']
2026-01-05 22:24:16,180 - INFO -     Is text: True
2026-01-05 22:24:16,180 - INFO -     Sample: DeepSeek-R1-0528
2026-01-05 22:24:16,180 - INFO -   - version
2026-01-05 22:24:16,180 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,180 - INFO -     Value types: ['str']
2026-01-05 22:24:16,180 - INFO -     Is text: True
2026-01-05 22:24:16,180 - INFO -     Sample: v1
2026-01-05 22:24:16,180 - INFO -   - category
2026-01-05 22:24:16,180 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,180 - INFO -     Value types: ['str']
2026-01-05 22:24:16,180 - INFO -     Is text: True
2026-01-05 22:24:16,180 - INFO -     Sample: code
2026-01-05 22:24:16,180 - INFO -   - reasoning
2026-01-05 22:24:16,180 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,180 - INFO -     Value types: ['str']
2026-01-05 22:24:16,180 - INFO -     Is text: True
2026-01-05 22:24:16,180 - INFO -     Sample: on
2026-01-05 22:24:16,180 - INFO -   - messages
2026-01-05 22:24:16,180 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None), 'tool_calls': [{'id': Value(dtype='string', id=None), 'type': Value(dtype='string', id=None), 'function': {'name': Value(dtype='string', id=None), 'arguments': Value(dtype='string', id=None)}}]}]
2026-01-05 22:24:16,180 - INFO -     Value types: ['list']
2026-01-05 22:24:16,180 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,180 - INFO -     Sample structure: ['role', 'content', 'tool_calls']
2026-01-05 22:24:16,180 - INFO -   - metadata
2026-01-05 22:24:16,180 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,181 - INFO -     Value types: ['str']
2026-01-05 22:24:16,181 - INFO -     Is text: True
2026-01-05 22:24:16,181 - INFO -     Sample: {"source": "codeforces", "dataset": "taco", "index": 16131, "split": "train"}
2026-01-05 22:24:16,181 - INFO - 
Message Structure:
2026-01-05 22:24:16,181 - INFO -   Roles: ['assistant', 'user']
2026-01-05 22:24:16,181 - INFO -   Message keys: ['content', 'role', 'tool_calls']
2026-01-05 22:24:16,181 - INFO -   Sample conversation length: 2 messages
2026-01-05 22:24:16,181 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,181 - INFO - 
Split: math (2,044,407 rows)
2026-01-05 22:24:16,181 - INFO - Columns (8): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages', 'metadata']
2026-01-05 22:24:16,184 - INFO - 
Column Details:
2026-01-05 22:24:16,184 - INFO -   - uuid
2026-01-05 22:24:16,184 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,184 - INFO -     Value types: ['str']
2026-01-05 22:24:16,184 - INFO -     Is text: True
2026-01-05 22:24:16,184 - INFO -     Sample: fb215f9b-9372-4a08-875e-94184764ad51
2026-01-05 22:24:16,184 - INFO -   - license
2026-01-05 22:24:16,184 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,184 - INFO -     Value types: ['str']
2026-01-05 22:24:16,184 - INFO -     Is text: True
2026-01-05 22:24:16,184 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,184 - INFO -   - generator
2026-01-05 22:24:16,184 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,184 - INFO -     Value types: ['str']
2026-01-05 22:24:16,184 - INFO -     Is text: True
2026-01-05 22:24:16,184 - INFO -     Sample: DeepSeek-R1-0528
2026-01-05 22:24:16,184 - INFO -   - version
2026-01-05 22:24:16,184 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,184 - INFO -     Value types: ['str']
2026-01-05 22:24:16,184 - INFO -     Is text: True
2026-01-05 22:24:16,184 - INFO -     Sample: v1
2026-01-05 22:24:16,184 - INFO -   - category
2026-01-05 22:24:16,184 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,184 - INFO -     Value types: ['str']
2026-01-05 22:24:16,184 - INFO -     Is text: True
2026-01-05 22:24:16,184 - INFO -     Sample: math
2026-01-05 22:24:16,184 - INFO -   - reasoning
2026-01-05 22:24:16,184 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,184 - INFO -     Value types: ['str']
2026-01-05 22:24:16,184 - INFO -     Is text: True
2026-01-05 22:24:16,184 - INFO -     Sample: on
2026-01-05 22:24:16,184 - INFO -   - messages
2026-01-05 22:24:16,184 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None), 'tool_calls': [{'id': Value(dtype='string', id=None), 'type': Value(dtype='string', id=None), 'function': {'name': Value(dtype='string', id=None), 'arguments': Value(dtype='string', id=None)}}]}]
2026-01-05 22:24:16,184 - INFO -     Value types: ['list']
2026-01-05 22:24:16,184 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,184 - INFO -     Sample structure: ['role', 'content', 'tool_calls']
2026-01-05 22:24:16,184 - INFO -   - metadata
2026-01-05 22:24:16,184 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,184 - INFO -     Value types: ['str']
2026-01-05 22:24:16,184 - INFO -     Is text: True
2026-01-05 22:24:16,184 - INFO -     Sample: {"expected_answer": "$V=\\frac{2}{3}m^3 \\cos^2 \\alpha \\sin \\alpha =\\frac{m^3\\sin 2\\alpha \\cos \\alpha }{3}$", "problem_source": "aops_c6_high_school_olympiads"}
2026-01-05 22:24:16,184 - INFO - 
Message Structure:
2026-01-05 22:24:16,184 - INFO -   Roles: ['assistant', 'user']
2026-01-05 22:24:16,184 - INFO -   Message keys: ['content', 'role', 'tool_calls']
2026-01-05 22:24:16,184 - INFO -   Sample conversation length: 2 messages
2026-01-05 22:24:16,184 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,184 - INFO - 
Split: stem (20,662,167 rows)
2026-01-05 22:24:16,184 - INFO - Columns (8): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages', 'metadata']
2026-01-05 22:24:16,187 - INFO - 
Column Details:
2026-01-05 22:24:16,187 - INFO -   - uuid
2026-01-05 22:24:16,187 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,187 - INFO -     Value types: ['str']
2026-01-05 22:24:16,187 - INFO -     Is text: True
2026-01-05 22:24:16,187 - INFO -     Sample: 03f14657-2565-4550-88c1-b72c9c990422
2026-01-05 22:24:16,187 - INFO -   - license
2026-01-05 22:24:16,187 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,187 - INFO -     Value types: ['str']
2026-01-05 22:24:16,187 - INFO -     Is text: True
2026-01-05 22:24:16,187 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,187 - INFO -   - generator
2026-01-05 22:24:16,187 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,187 - INFO -     Value types: ['str']
2026-01-05 22:24:16,187 - INFO -     Is text: True
2026-01-05 22:24:16,187 - INFO -     Sample: DeepSeek-R1-0528
2026-01-05 22:24:16,187 - INFO -   - version
2026-01-05 22:24:16,187 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,187 - INFO -     Value types: ['str']
2026-01-05 22:24:16,187 - INFO -     Is text: True
2026-01-05 22:24:16,187 - INFO -     Sample: v1
2026-01-05 22:24:16,187 - INFO -   - category
2026-01-05 22:24:16,187 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,187 - INFO -     Value types: ['str']
2026-01-05 22:24:16,187 - INFO -     Is text: True
2026-01-05 22:24:16,187 - INFO -     Sample: stem
2026-01-05 22:24:16,187 - INFO -   - reasoning
2026-01-05 22:24:16,187 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,187 - INFO -     Value types: ['str']
2026-01-05 22:24:16,187 - INFO -     Is text: True
2026-01-05 22:24:16,187 - INFO -     Sample: on
2026-01-05 22:24:16,187 - INFO -   - messages
2026-01-05 22:24:16,187 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None), 'tool_calls': [{'id': Value(dtype='string', id=None), 'type': Value(dtype='string', id=None), 'function': {'name': Value(dtype='string', id=None), 'arguments': Value(dtype='string', id=None)}}]}]
2026-01-05 22:24:16,187 - INFO -     Value types: ['list']
2026-01-05 22:24:16,187 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,187 - INFO -     Sample structure: ['role', 'content', 'tool_calls']
2026-01-05 22:24:16,187 - INFO -   - metadata
2026-01-05 22:24:16,187 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,187 - INFO -     Value types: ['str']
2026-01-05 22:24:16,187 - INFO -     Is text: True
2026-01-05 22:24:16,187 - INFO -     Sample: {"expected_answer": "\\text{A}"}
2026-01-05 22:24:16,187 - INFO - 
Message Structure:
2026-01-05 22:24:16,187 - INFO -   Roles: ['assistant', 'user']
2026-01-05 22:24:16,187 - INFO -   Message keys: ['content', 'role', 'tool_calls']
2026-01-05 22:24:16,187 - INFO -   Sample conversation length: 2 messages
2026-01-05 22:24:16,187 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,187 - INFO - 
Split: tool_calling (310,051 rows)
2026-01-05 22:24:16,188 - INFO - Columns (8): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages', 'metadata']
2026-01-05 22:24:16,193 - INFO - 
Column Details:
2026-01-05 22:24:16,193 - INFO -   - uuid
2026-01-05 22:24:16,193 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,193 - INFO -     Value types: ['str']
2026-01-05 22:24:16,193 - INFO -     Is text: True
2026-01-05 22:24:16,193 - INFO -     Sample: a094c815-7f5c-4604-8cb1-a173c5107d7d
2026-01-05 22:24:16,193 - INFO -   - license
2026-01-05 22:24:16,193 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,193 - INFO -     Value types: ['str']
2026-01-05 22:24:16,193 - INFO -     Is text: True
2026-01-05 22:24:16,193 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,193 - INFO -   - generator
2026-01-05 22:24:16,193 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,193 - INFO -     Value types: ['str']
2026-01-05 22:24:16,193 - INFO -     Is text: True
2026-01-05 22:24:16,193 - INFO -     Sample: Qwen3-235B-A22B
2026-01-05 22:24:16,193 - INFO -   - version
2026-01-05 22:24:16,193 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,193 - INFO -     Value types: ['str']
2026-01-05 22:24:16,193 - INFO -     Is text: True
2026-01-05 22:24:16,193 - INFO -     Sample: v1
2026-01-05 22:24:16,193 - INFO -   - category
2026-01-05 22:24:16,193 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,193 - INFO -     Value types: ['str']
2026-01-05 22:24:16,193 - INFO -     Is text: True
2026-01-05 22:24:16,193 - INFO -     Sample: tool_calling
2026-01-05 22:24:16,193 - INFO -   - reasoning
2026-01-05 22:24:16,193 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,193 - INFO -     Value types: ['str']
2026-01-05 22:24:16,193 - INFO -     Is text: True
2026-01-05 22:24:16,193 - INFO -     Sample: on
2026-01-05 22:24:16,193 - INFO -   - messages
2026-01-05 22:24:16,193 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None), 'tool_calls': [{'id': Value(dtype='string', id=None), 'type': Value(dtype='string', id=None), 'function': {'name': Value(dtype='string', id=None), 'arguments': Value(dtype='string', id=None)}}]}]
2026-01-05 22:24:16,193 - INFO -     Value types: ['list']
2026-01-05 22:24:16,193 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,193 - INFO -     Sample structure: ['role', 'content', 'tool_calls']
2026-01-05 22:24:16,193 - INFO -   - metadata
2026-01-05 22:24:16,193 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,193 - INFO -     Value types: ['str']
2026-01-05 22:24:16,193 - INFO -     Is text: True
2026-01-05 22:24:16,193 - INFO -     Sample: {"tools": [{"type": "function", "function": {"name": "check_valid_registration", "description": "Verifies whether a vehicle registration number is valid for a specific state and returns detailed infor...
2026-01-05 22:24:16,193 - INFO - 
Message Structure:
2026-01-05 22:24:16,193 - INFO -   Roles: ['assistant', 'tool', 'user']
2026-01-05 22:24:16,193 - INFO -   Message keys: ['content', 'role', 'tool_calls']
2026-01-05 22:24:16,193 - INFO -   Sample conversation length: 4 messages
2026-01-05 22:24:16,193 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,193 - INFO - 
================================================================================
2026-01-05 22:24:16,193 - INFO - Dataset Summary: nvidia/Nemotron-Post-Training-Dataset-v1
2026-01-05 22:24:16,193 - INFO -   Total rows: 25,659,642
2026-01-05 22:24:16,193 - INFO -   Number of splits: 5
2026-01-05 22:24:16,193 - INFO -   All columns: ['category', 'generator', 'license', 'messages', 'metadata', 'reasoning', 'uuid', 'version']
2026-01-05 22:24:16,193 - INFO -   Embedding columns: ['messages']
2026-01-05 22:24:16,193 - INFO - ================================================================================
2026-01-05 22:24:16,507 - INFO - 
[3/6] Processing: nvidia/Nemotron-Post-Training-Dataset-v2
2026-01-05 22:24:16,507 - INFO - 
================================================================================
2026-01-05 22:24:16,507 - INFO - Exploring: nvidia/Nemotron-Post-Training-Dataset-v2
2026-01-05 22:24:16,507 - INFO - ================================================================================
2026-01-05 22:24:16,679 - INFO - 
Split: stem (355,000 rows)
2026-01-05 22:24:16,679 - INFO - Columns (7): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,682 - INFO - 
Column Details:
2026-01-05 22:24:16,682 - INFO -   - uuid
2026-01-05 22:24:16,682 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,682 - INFO -     Value types: ['str']
2026-01-05 22:24:16,682 - INFO -     Is text: True
2026-01-05 22:24:16,682 - INFO -     Sample: 5ed129f9-8548-4cbd-abd4-7ff362f7facc
2026-01-05 22:24:16,682 - INFO -   - license
2026-01-05 22:24:16,682 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,682 - INFO -     Value types: ['str']
2026-01-05 22:24:16,682 - INFO -     Is text: True
2026-01-05 22:24:16,682 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,682 - INFO -   - generator
2026-01-05 22:24:16,682 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,682 - INFO -     Value types: ['str']
2026-01-05 22:24:16,682 - INFO -     Is text: True
2026-01-05 22:24:16,682 - INFO -     Sample: DeepSeek-R1-0528
2026-01-05 22:24:16,682 - INFO -   - version
2026-01-05 22:24:16,682 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,682 - INFO -     Value types: ['str']
2026-01-05 22:24:16,682 - INFO -     Is text: True
2026-01-05 22:24:16,682 - INFO -     Sample: v2
2026-01-05 22:24:16,682 - INFO -   - category
2026-01-05 22:24:16,682 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,682 - INFO -     Value types: ['str']
2026-01-05 22:24:16,682 - INFO -     Is text: True
2026-01-05 22:24:16,682 - INFO -     Sample: stem
2026-01-05 22:24:16,682 - INFO -   - reasoning
2026-01-05 22:24:16,682 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,682 - INFO -     Value types: ['str']
2026-01-05 22:24:16,682 - INFO -     Is text: True
2026-01-05 22:24:16,682 - INFO -     Sample: off
2026-01-05 22:24:16,682 - INFO -   - messages
2026-01-05 22:24:16,682 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,682 - INFO -     Value types: ['list']
2026-01-05 22:24:16,682 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,682 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:16,682 - INFO - 
Message Structure:
2026-01-05 22:24:16,682 - INFO -   Roles: ['assistant', 'system', 'user']
2026-01-05 22:24:16,682 - INFO -   Message keys: ['content', 'role']
2026-01-05 22:24:16,682 - INFO -   Sample conversation length: 3 messages
2026-01-05 22:24:16,682 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,682 - INFO - 
Split: chat (627,720 rows)
2026-01-05 22:24:16,682 - INFO - Columns (7): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,685 - INFO - 
Column Details:
2026-01-05 22:24:16,685 - INFO -   - uuid
2026-01-05 22:24:16,685 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,685 - INFO -     Value types: ['str']
2026-01-05 22:24:16,685 - INFO -     Is text: True
2026-01-05 22:24:16,685 - INFO -     Sample: 76242391-3c82-4471-a971-e51f57b2899e
2026-01-05 22:24:16,685 - INFO -   - license
2026-01-05 22:24:16,685 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,685 - INFO -     Value types: ['str']
2026-01-05 22:24:16,685 - INFO -     Is text: True
2026-01-05 22:24:16,685 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,685 - INFO -   - generator
2026-01-05 22:24:16,685 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,685 - INFO -     Value types: ['str']
2026-01-05 22:24:16,685 - INFO -     Is text: True
2026-01-05 22:24:16,685 - INFO -     Sample: Qwen3-235B-A22B, Qwen3-30B-A3B
2026-01-05 22:24:16,685 - INFO -   - version
2026-01-05 22:24:16,685 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,685 - INFO -     Value types: ['str']
2026-01-05 22:24:16,685 - INFO -     Is text: True
2026-01-05 22:24:16,685 - INFO -     Sample: v2
2026-01-05 22:24:16,685 - INFO -   - category
2026-01-05 22:24:16,685 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,685 - INFO -     Value types: ['str']
2026-01-05 22:24:16,685 - INFO -     Is text: True
2026-01-05 22:24:16,685 - INFO -     Sample: chat
2026-01-05 22:24:16,685 - INFO -   - reasoning
2026-01-05 22:24:16,685 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,685 - INFO -     Value types: ['str']
2026-01-05 22:24:16,685 - INFO -     Is text: True
2026-01-05 22:24:16,685 - INFO -     Sample: off
2026-01-05 22:24:16,685 - INFO -   - messages
2026-01-05 22:24:16,685 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,685 - INFO -     Value types: ['list']
2026-01-05 22:24:16,685 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,685 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:16,685 - INFO - 
Message Structure:
2026-01-05 22:24:16,685 - INFO -   Roles: ['assistant', 'system', 'user']
2026-01-05 22:24:16,685 - INFO -   Message keys: ['content', 'role']
2026-01-05 22:24:16,685 - INFO -   Sample conversation length: 3 messages
2026-01-05 22:24:16,685 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,685 - INFO - 
Split: math (239,467 rows)
2026-01-05 22:24:16,685 - INFO - Columns (7): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,687 - INFO - 
Column Details:
2026-01-05 22:24:16,687 - INFO -   - uuid
2026-01-05 22:24:16,687 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,687 - INFO -     Value types: ['str']
2026-01-05 22:24:16,687 - INFO -     Is text: True
2026-01-05 22:24:16,687 - INFO -     Sample: cda83850-fbb6-447c-97fb-59c111a5596a
2026-01-05 22:24:16,687 - INFO -   - license
2026-01-05 22:24:16,687 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,687 - INFO -     Value types: ['str']
2026-01-05 22:24:16,687 - INFO -     Is text: True
2026-01-05 22:24:16,687 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,687 - INFO -   - generator
2026-01-05 22:24:16,687 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,687 - INFO -     Value types: ['str']
2026-01-05 22:24:16,687 - INFO -     Is text: True
2026-01-05 22:24:16,687 - INFO -     Sample: DeepSeek-R1-0528
2026-01-05 22:24:16,687 - INFO -   - version
2026-01-05 22:24:16,687 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,687 - INFO -     Value types: ['str']
2026-01-05 22:24:16,687 - INFO -     Is text: True
2026-01-05 22:24:16,687 - INFO -     Sample: v2
2026-01-05 22:24:16,687 - INFO -   - category
2026-01-05 22:24:16,687 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,687 - INFO -     Value types: ['str']
2026-01-05 22:24:16,687 - INFO -     Is text: True
2026-01-05 22:24:16,687 - INFO -     Sample: math
2026-01-05 22:24:16,687 - INFO -   - reasoning
2026-01-05 22:24:16,687 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,687 - INFO -     Value types: ['str']
2026-01-05 22:24:16,687 - INFO -     Is text: True
2026-01-05 22:24:16,687 - INFO -     Sample: off
2026-01-05 22:24:16,687 - INFO -   - messages
2026-01-05 22:24:16,687 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,687 - INFO -     Value types: ['list']
2026-01-05 22:24:16,687 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,687 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:16,687 - INFO - 
Message Structure:
2026-01-05 22:24:16,687 - INFO -   Roles: ['assistant', 'system', 'user']
2026-01-05 22:24:16,688 - INFO -   Message keys: ['content', 'role']
2026-01-05 22:24:16,688 - INFO -   Sample conversation length: 3 messages
2026-01-05 22:24:16,688 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,688 - INFO - 
Split: code (175,000 rows)
2026-01-05 22:24:16,688 - INFO - Columns (7): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,690 - INFO - 
Column Details:
2026-01-05 22:24:16,690 - INFO -   - uuid
2026-01-05 22:24:16,690 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,690 - INFO -     Value types: ['str']
2026-01-05 22:24:16,690 - INFO -     Is text: True
2026-01-05 22:24:16,690 - INFO -     Sample: d375a7a7-6369-4b58-808f-ce0716325977
2026-01-05 22:24:16,690 - INFO -   - license
2026-01-05 22:24:16,690 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,690 - INFO -     Value types: ['str']
2026-01-05 22:24:16,690 - INFO -     Is text: True
2026-01-05 22:24:16,690 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,690 - INFO -   - generator
2026-01-05 22:24:16,690 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,690 - INFO -     Value types: ['str']
2026-01-05 22:24:16,690 - INFO -     Is text: True
2026-01-05 22:24:16,690 - INFO -     Sample: DeepSeek-R1-0528
2026-01-05 22:24:16,690 - INFO -   - version
2026-01-05 22:24:16,690 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,690 - INFO -     Value types: ['str']
2026-01-05 22:24:16,690 - INFO -     Is text: True
2026-01-05 22:24:16,690 - INFO -     Sample: v2
2026-01-05 22:24:16,690 - INFO -   - category
2026-01-05 22:24:16,690 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,690 - INFO -     Value types: ['str']
2026-01-05 22:24:16,690 - INFO -     Is text: True
2026-01-05 22:24:16,690 - INFO -     Sample: code
2026-01-05 22:24:16,690 - INFO -   - reasoning
2026-01-05 22:24:16,690 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,690 - INFO -     Value types: ['str']
2026-01-05 22:24:16,690 - INFO -     Is text: True
2026-01-05 22:24:16,690 - INFO -     Sample: off
2026-01-05 22:24:16,690 - INFO -   - messages
2026-01-05 22:24:16,690 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,690 - INFO -     Value types: ['list']
2026-01-05 22:24:16,690 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,690 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:16,690 - INFO - 
Message Structure:
2026-01-05 22:24:16,690 - INFO -   Roles: ['assistant', 'system', 'user']
2026-01-05 22:24:16,690 - INFO -   Message keys: ['content', 'role']
2026-01-05 22:24:16,690 - INFO -   Sample conversation length: 3 messages
2026-01-05 22:24:16,690 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,690 - INFO - 
Split: multilingual_ja (975,202 rows)
2026-01-05 22:24:16,690 - INFO - Columns (7): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,693 - INFO - 
Column Details:
2026-01-05 22:24:16,693 - INFO -   - uuid
2026-01-05 22:24:16,693 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,693 - INFO -     Value types: ['str']
2026-01-05 22:24:16,693 - INFO -     Is text: True
2026-01-05 22:24:16,693 - INFO -     Sample: cb113e8e-40a8-4cfa-9bbe-6e8f11fea462
2026-01-05 22:24:16,693 - INFO -   - license
2026-01-05 22:24:16,693 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,693 - INFO -     Value types: ['str']
2026-01-05 22:24:16,693 - INFO -     Is text: True
2026-01-05 22:24:16,693 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,693 - INFO -   - generator
2026-01-05 22:24:16,693 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,693 - INFO -     Value types: ['str']
2026-01-05 22:24:16,693 - INFO -     Is text: True
2026-01-05 22:24:16,693 - INFO -     Sample: DeepSeek-R1-0528, Qwen2.5-14B-Instruct
2026-01-05 22:24:16,693 - INFO -   - version
2026-01-05 22:24:16,693 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,693 - INFO -     Value types: ['str']
2026-01-05 22:24:16,693 - INFO -     Is text: True
2026-01-05 22:24:16,693 - INFO -     Sample: v2
2026-01-05 22:24:16,693 - INFO -   - category
2026-01-05 22:24:16,693 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,693 - INFO -     Value types: ['str']
2026-01-05 22:24:16,693 - INFO -     Is text: True
2026-01-05 22:24:16,693 - INFO -     Sample: multilingual_ja
2026-01-05 22:24:16,693 - INFO -   - reasoning
2026-01-05 22:24:16,693 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,693 - INFO -     Value types: ['str']
2026-01-05 22:24:16,693 - INFO -     Is text: True
2026-01-05 22:24:16,693 - INFO -     Sample: on
2026-01-05 22:24:16,693 - INFO -   - messages
2026-01-05 22:24:16,693 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,693 - INFO -     Value types: ['list']
2026-01-05 22:24:16,693 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,693 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:16,693 - INFO - 
Message Structure:
2026-01-05 22:24:16,693 - INFO -   Roles: ['assistant', 'system', 'user']
2026-01-05 22:24:16,693 - INFO -   Message keys: ['content', 'role']
2026-01-05 22:24:16,693 - INFO -   Sample conversation length: 3 messages
2026-01-05 22:24:16,693 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,693 - INFO - 
Split: multilingual_de (1,015,314 rows)
2026-01-05 22:24:16,693 - INFO - Columns (7): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,695 - INFO - 
Column Details:
2026-01-05 22:24:16,695 - INFO -   - uuid
2026-01-05 22:24:16,696 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,696 - INFO -     Value types: ['str']
2026-01-05 22:24:16,696 - INFO -     Is text: True
2026-01-05 22:24:16,696 - INFO -     Sample: ce80510c-c67b-43c6-a975-e85d576d417e
2026-01-05 22:24:16,696 - INFO -   - license
2026-01-05 22:24:16,696 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,696 - INFO -     Value types: ['str']
2026-01-05 22:24:16,696 - INFO -     Is text: True
2026-01-05 22:24:16,696 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,696 - INFO -   - generator
2026-01-05 22:24:16,696 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,696 - INFO -     Value types: ['str']
2026-01-05 22:24:16,696 - INFO -     Is text: True
2026-01-05 22:24:16,696 - INFO -     Sample: DeepSeek-R1-0528, Qwen2.5-32B-Instruct-AWQ
2026-01-05 22:24:16,696 - INFO -   - version
2026-01-05 22:24:16,696 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,696 - INFO -     Value types: ['str']
2026-01-05 22:24:16,696 - INFO -     Is text: True
2026-01-05 22:24:16,696 - INFO -     Sample: v2
2026-01-05 22:24:16,696 - INFO -   - category
2026-01-05 22:24:16,696 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,696 - INFO -     Value types: ['str']
2026-01-05 22:24:16,696 - INFO -     Is text: True
2026-01-05 22:24:16,696 - INFO -     Sample: multilingual_de
2026-01-05 22:24:16,696 - INFO -   - reasoning
2026-01-05 22:24:16,696 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,696 - INFO -     Value types: ['str']
2026-01-05 22:24:16,696 - INFO -     Is text: True
2026-01-05 22:24:16,696 - INFO -     Sample: on
2026-01-05 22:24:16,696 - INFO -   - messages
2026-01-05 22:24:16,696 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,696 - INFO -     Value types: ['list']
2026-01-05 22:24:16,696 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,696 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:16,696 - INFO - 
Message Structure:
2026-01-05 22:24:16,696 - INFO -   Roles: ['assistant', 'system', 'user']
2026-01-05 22:24:16,696 - INFO -   Message keys: ['content', 'role']
2026-01-05 22:24:16,696 - INFO -   Sample conversation length: 3 messages
2026-01-05 22:24:16,696 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,696 - INFO - 
Split: multilingual_it (1,016,503 rows)
2026-01-05 22:24:16,696 - INFO - Columns (7): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,698 - INFO - 
Column Details:
2026-01-05 22:24:16,698 - INFO -   - uuid
2026-01-05 22:24:16,698 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,698 - INFO -     Value types: ['str']
2026-01-05 22:24:16,698 - INFO -     Is text: True
2026-01-05 22:24:16,698 - INFO -     Sample: d1ade403-e627-4cfb-b49d-0ba3e30e48c1
2026-01-05 22:24:16,698 - INFO -   - license
2026-01-05 22:24:16,698 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,698 - INFO -     Value types: ['str']
2026-01-05 22:24:16,698 - INFO -     Is text: True
2026-01-05 22:24:16,698 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,698 - INFO -   - generator
2026-01-05 22:24:16,698 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,698 - INFO -     Value types: ['str']
2026-01-05 22:24:16,698 - INFO -     Is text: True
2026-01-05 22:24:16,699 - INFO -     Sample: DeepSeek-R1-0528, Qwen2.5-14B-Instruct
2026-01-05 22:24:16,699 - INFO -   - version
2026-01-05 22:24:16,699 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,699 - INFO -     Value types: ['str']
2026-01-05 22:24:16,699 - INFO -     Is text: True
2026-01-05 22:24:16,699 - INFO -     Sample: v2
2026-01-05 22:24:16,699 - INFO -   - category
2026-01-05 22:24:16,699 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,699 - INFO -     Value types: ['str']
2026-01-05 22:24:16,699 - INFO -     Is text: True
2026-01-05 22:24:16,699 - INFO -     Sample: multilingual_it
2026-01-05 22:24:16,699 - INFO -   - reasoning
2026-01-05 22:24:16,699 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,699 - INFO -     Value types: ['str']
2026-01-05 22:24:16,699 - INFO -     Is text: True
2026-01-05 22:24:16,699 - INFO -     Sample: on
2026-01-05 22:24:16,699 - INFO -   - messages
2026-01-05 22:24:16,699 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,699 - INFO -     Value types: ['list']
2026-01-05 22:24:16,699 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,699 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:16,699 - INFO - 
Message Structure:
2026-01-05 22:24:16,699 - INFO -   Roles: ['assistant', 'system', 'user']
2026-01-05 22:24:16,699 - INFO -   Message keys: ['content', 'role']
2026-01-05 22:24:16,699 - INFO -   Sample conversation length: 3 messages
2026-01-05 22:24:16,699 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,699 - INFO - 
Split: multilingual_es (935,704 rows)
2026-01-05 22:24:16,699 - INFO - Columns (7): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,701 - INFO - 
Column Details:
2026-01-05 22:24:16,701 - INFO -   - uuid
2026-01-05 22:24:16,701 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,701 - INFO -     Value types: ['str']
2026-01-05 22:24:16,701 - INFO -     Is text: True
2026-01-05 22:24:16,701 - INFO -     Sample: fe99662e-f5ed-4b5c-be09-41228474ab5d
2026-01-05 22:24:16,701 - INFO -   - license
2026-01-05 22:24:16,701 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,701 - INFO -     Value types: ['str']
2026-01-05 22:24:16,701 - INFO -     Is text: True
2026-01-05 22:24:16,701 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,701 - INFO -   - generator
2026-01-05 22:24:16,701 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,701 - INFO -     Value types: ['str']
2026-01-05 22:24:16,701 - INFO -     Is text: True
2026-01-05 22:24:16,701 - INFO -     Sample: DeepSeek-R1-0528, Qwen2.5-14B-Instruct
2026-01-05 22:24:16,701 - INFO -   - version
2026-01-05 22:24:16,701 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,701 - INFO -     Value types: ['str']
2026-01-05 22:24:16,701 - INFO -     Is text: True
2026-01-05 22:24:16,701 - INFO -     Sample: v2
2026-01-05 22:24:16,701 - INFO -   - category
2026-01-05 22:24:16,701 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,701 - INFO -     Value types: ['str']
2026-01-05 22:24:16,701 - INFO -     Is text: True
2026-01-05 22:24:16,701 - INFO -     Sample: multilingual_es
2026-01-05 22:24:16,701 - INFO -   - reasoning
2026-01-05 22:24:16,701 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,701 - INFO -     Value types: ['str']
2026-01-05 22:24:16,701 - INFO -     Is text: True
2026-01-05 22:24:16,701 - INFO -     Sample: on
2026-01-05 22:24:16,701 - INFO -   - messages
2026-01-05 22:24:16,701 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,701 - INFO -     Value types: ['list']
2026-01-05 22:24:16,701 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,701 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:16,701 - INFO - 
Message Structure:
2026-01-05 22:24:16,701 - INFO -   Roles: ['assistant', 'system', 'user']
2026-01-05 22:24:16,701 - INFO -   Message keys: ['content', 'role']
2026-01-05 22:24:16,701 - INFO -   Sample conversation length: 3 messages
2026-01-05 22:24:16,701 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,701 - INFO - 
Split: multilingual_fr (1,001,504 rows)
2026-01-05 22:24:16,701 - INFO - Columns (7): ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,704 - INFO - 
Column Details:
2026-01-05 22:24:16,704 - INFO -   - uuid
2026-01-05 22:24:16,704 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,704 - INFO -     Value types: ['str']
2026-01-05 22:24:16,704 - INFO -     Is text: True
2026-01-05 22:24:16,704 - INFO -     Sample: 86fd2488-790f-4551-9c68-7cfc97c53b86
2026-01-05 22:24:16,704 - INFO -   - license
2026-01-05 22:24:16,704 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,704 - INFO -     Value types: ['str']
2026-01-05 22:24:16,704 - INFO -     Is text: True
2026-01-05 22:24:16,704 - INFO -     Sample: CC BY 4.0
2026-01-05 22:24:16,704 - INFO -   - generator
2026-01-05 22:24:16,704 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,704 - INFO -     Value types: ['str']
2026-01-05 22:24:16,704 - INFO -     Is text: True
2026-01-05 22:24:16,704 - INFO -     Sample: DeepSeek-R1-0528, Qwen2.5-14B-Instruct
2026-01-05 22:24:16,704 - INFO -   - version
2026-01-05 22:24:16,704 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,704 - INFO -     Value types: ['str']
2026-01-05 22:24:16,704 - INFO -     Is text: True
2026-01-05 22:24:16,704 - INFO -     Sample: v2
2026-01-05 22:24:16,704 - INFO -   - category
2026-01-05 22:24:16,704 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,704 - INFO -     Value types: ['str']
2026-01-05 22:24:16,704 - INFO -     Is text: True
2026-01-05 22:24:16,704 - INFO -     Sample: multilingual_fr
2026-01-05 22:24:16,704 - INFO -   - reasoning
2026-01-05 22:24:16,704 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,704 - INFO -     Value types: ['str']
2026-01-05 22:24:16,704 - INFO -     Is text: True
2026-01-05 22:24:16,704 - INFO -     Sample: on
2026-01-05 22:24:16,704 - INFO -   - messages
2026-01-05 22:24:16,704 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,704 - INFO -     Value types: ['list']
2026-01-05 22:24:16,704 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,704 - INFO -     Sample structure: ['role', 'content']
2026-01-05 22:24:16,704 - INFO - 
Message Structure:
2026-01-05 22:24:16,704 - INFO -   Roles: ['assistant', 'system', 'user']
2026-01-05 22:24:16,704 - INFO -   Message keys: ['content', 'role']
2026-01-05 22:24:16,704 - INFO -   Sample conversation length: 3 messages
2026-01-05 22:24:16,704 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,704 - INFO - 
================================================================================
2026-01-05 22:24:16,704 - INFO - Dataset Summary: nvidia/Nemotron-Post-Training-Dataset-v2
2026-01-05 22:24:16,704 - INFO -   Total rows: 6,341,414
2026-01-05 22:24:16,704 - INFO -   Number of splits: 9
2026-01-05 22:24:16,704 - INFO -   All columns: ['category', 'generator', 'license', 'messages', 'reasoning', 'uuid', 'version']
2026-01-05 22:24:16,704 - INFO -   Embedding columns: ['messages']
2026-01-05 22:24:16,704 - INFO - ================================================================================
2026-01-05 22:24:16,743 - INFO - 
[4/6] Processing: nvidia/Nemotron-Science-v1
2026-01-05 22:24:16,744 - INFO - 
================================================================================
2026-01-05 22:24:16,744 - INFO - Exploring: nvidia/Nemotron-Science-v1
2026-01-05 22:24:16,744 - INFO - ================================================================================
2026-01-05 22:24:16,755 - INFO - 
Split: MCQ (174,155 rows)
2026-01-05 22:24:16,755 - INFO - Columns (5): ['uuid', 'messages', 'license', 'used_in', 'tools']
2026-01-05 22:24:16,756 - INFO - 
Column Details:
2026-01-05 22:24:16,756 - INFO -   - uuid
2026-01-05 22:24:16,756 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,756 - INFO -     Value types: ['str']
2026-01-05 22:24:16,757 - INFO -     Is text: True
2026-01-05 22:24:16,757 - INFO -     Sample: 4ee55134-2c04-4dc8-8535-664e6572c32f
2026-01-05 22:24:16,757 - INFO -   - messages
2026-01-05 22:24:16,757 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None), 'reasoning_content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,757 - INFO -     Value types: ['list']
2026-01-05 22:24:16,757 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,757 - INFO -     Sample structure: ['role', 'content', 'reasoning_content']
2026-01-05 22:24:16,757 - INFO -   - license
2026-01-05 22:24:16,757 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,757 - INFO -     Value types: ['str']
2026-01-05 22:24:16,757 - INFO -     Is text: True
2026-01-05 22:24:16,757 - INFO -     Sample: cc-by-4.0
2026-01-05 22:24:16,757 - INFO -   - used_in
2026-01-05 22:24:16,757 - INFO -     Type: [Value(dtype='string', id=None)]
2026-01-05 22:24:16,757 - INFO -     Value types: ['list']
2026-01-05 22:24:16,757 - INFO -   - tools
2026-01-05 22:24:16,757 - INFO -     Type: [Value(dtype='null', id=None)]
2026-01-05 22:24:16,757 - INFO -     Value types: ['list']
2026-01-05 22:24:16,757 - INFO - 
Message Structure:
2026-01-05 22:24:16,757 - INFO -   Roles: ['assistant', 'user']
2026-01-05 22:24:16,757 - INFO -   Message keys: ['content', 'reasoning_content', 'role']
2026-01-05 22:24:16,757 - INFO -   Sample conversation length: 2 messages
2026-01-05 22:24:16,757 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,757 - INFO - 
Split: RQA (52,179 rows)
2026-01-05 22:24:16,757 - INFO - Columns (5): ['uuid', 'messages', 'license', 'used_in', 'tools']
2026-01-05 22:24:16,758 - INFO - 
Column Details:
2026-01-05 22:24:16,759 - INFO -   - uuid
2026-01-05 22:24:16,759 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,759 - INFO -     Value types: ['str']
2026-01-05 22:24:16,759 - INFO -     Is text: True
2026-01-05 22:24:16,759 - INFO -     Sample: aa23c0c1-10f8-4b25-9cfd-6f2cf6c046d0
2026-01-05 22:24:16,759 - INFO -   - messages
2026-01-05 22:24:16,759 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None), 'reasoning_content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,759 - INFO -     Value types: ['list']
2026-01-05 22:24:16,759 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,759 - INFO -     Sample structure: ['role', 'content', 'reasoning_content']
2026-01-05 22:24:16,759 - INFO -   - license
2026-01-05 22:24:16,759 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,759 - INFO -     Value types: ['str']
2026-01-05 22:24:16,759 - INFO -     Is text: True
2026-01-05 22:24:16,759 - INFO -     Sample: cc-by-4.0
2026-01-05 22:24:16,759 - INFO -   - used_in
2026-01-05 22:24:16,759 - INFO -     Type: [Value(dtype='string', id=None)]
2026-01-05 22:24:16,759 - INFO -     Value types: ['list']
2026-01-05 22:24:16,759 - INFO -   - tools
2026-01-05 22:24:16,759 - INFO -     Type: [Value(dtype='null', id=None)]
2026-01-05 22:24:16,759 - INFO -     Value types: ['list']
2026-01-05 22:24:16,759 - INFO - 
Message Structure:
2026-01-05 22:24:16,759 - INFO -   Roles: ['assistant', 'user']
2026-01-05 22:24:16,759 - INFO -   Message keys: ['content', 'reasoning_content', 'role']
2026-01-05 22:24:16,759 - INFO -   Sample conversation length: 2 messages
2026-01-05 22:24:16,759 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,759 - INFO - 
================================================================================
2026-01-05 22:24:16,759 - INFO - Dataset Summary: nvidia/Nemotron-Science-v1
2026-01-05 22:24:16,759 - INFO -   Total rows: 226,334
2026-01-05 22:24:16,759 - INFO -   Number of splits: 2
2026-01-05 22:24:16,759 - INFO -   All columns: ['license', 'messages', 'tools', 'used_in', 'uuid']
2026-01-05 22:24:16,759 - INFO -   Embedding columns: ['messages']
2026-01-05 22:24:16,759 - INFO - ================================================================================
2026-01-05 22:24:16,760 - INFO - 
[5/6] Processing: nvidia/Nemotron-Instruction-Following-Chat-v1
2026-01-05 22:24:16,760 - INFO - 
================================================================================
2026-01-05 22:24:16,760 - INFO - Exploring: nvidia/Nemotron-Instruction-Following-Chat-v1
2026-01-05 22:24:16,760 - INFO - ================================================================================
2026-01-05 22:24:16,772 - INFO - 
Split: chat_if (426,009 rows)
2026-01-05 22:24:16,772 - INFO - Columns (7): ['uuid', 'messages', 'license', 'used_in', 'tools', 'reasoning', 'capability_target']
2026-01-05 22:24:16,776 - INFO - 
Column Details:
2026-01-05 22:24:16,776 - INFO -   - uuid
2026-01-05 22:24:16,776 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,776 - INFO -     Value types: ['str']
2026-01-05 22:24:16,776 - INFO -     Is text: True
2026-01-05 22:24:16,776 - INFO -     Sample: 67924276-d312-47a7-84c8-bb2bd3461be9
2026-01-05 22:24:16,776 - INFO -   - messages
2026-01-05 22:24:16,776 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None), 'reasoning_content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,776 - INFO -     Value types: ['list']
2026-01-05 22:24:16,776 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,776 - INFO -     Sample structure: ['role', 'content', 'reasoning_content']
2026-01-05 22:24:16,776 - INFO -   - license
2026-01-05 22:24:16,776 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,776 - INFO -     Value types: ['str']
2026-01-05 22:24:16,776 - INFO -     Is text: True
2026-01-05 22:24:16,776 - INFO -     Sample: odc-by-1.0
2026-01-05 22:24:16,776 - INFO -   - used_in
2026-01-05 22:24:16,776 - INFO -     Type: Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)
2026-01-05 22:24:16,776 - INFO -     Value types: ['list']
2026-01-05 22:24:16,776 - INFO -   - tools
2026-01-05 22:24:16,776 - INFO -     Type: Sequence(feature=Value(dtype='null', id=None), length=-1, id=None)
2026-01-05 22:24:16,776 - INFO -     Value types: ['list']
2026-01-05 22:24:16,776 - INFO -   - reasoning
2026-01-05 22:24:16,776 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,776 - INFO -     Value types: ['str']
2026-01-05 22:24:16,776 - INFO -     Is text: True
2026-01-05 22:24:16,776 - INFO -     Sample: off
2026-01-05 22:24:16,776 - INFO -   - capability_target
2026-01-05 22:24:16,776 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,776 - INFO -     Value types: ['str']
2026-01-05 22:24:16,776 - INFO -     Is text: True
2026-01-05 22:24:16,776 - INFO -     Sample: chat
2026-01-05 22:24:16,776 - INFO - 
Message Structure:
2026-01-05 22:24:16,776 - INFO -   Roles: ['assistant', 'system', 'user']
2026-01-05 22:24:16,776 - INFO -   Message keys: ['content', 'reasoning_content', 'role']
2026-01-05 22:24:16,776 - INFO -   Sample conversation length: 19 messages
2026-01-05 22:24:16,776 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,776 - INFO - 
Split: structured_outputs (4,969 rows)
2026-01-05 22:24:16,776 - INFO - Columns (7): ['uuid', 'messages', 'license', 'used_in', 'tools', 'reasoning', 'capability_target']
2026-01-05 22:24:16,779 - INFO - 
Column Details:
2026-01-05 22:24:16,779 - INFO -   - uuid
2026-01-05 22:24:16,779 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,779 - INFO -     Value types: ['str']
2026-01-05 22:24:16,779 - INFO -     Is text: True
2026-01-05 22:24:16,779 - INFO -     Sample: cd84e0c7-073c-4e6e-b28f-f1ee9c6172f0
2026-01-05 22:24:16,779 - INFO -   - messages
2026-01-05 22:24:16,779 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None), 'reasoning_content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,779 - INFO -     Value types: ['list']
2026-01-05 22:24:16,779 - INFO -     Is list of dicts: True
2026-01-05 22:24:16,779 - INFO -     Sample structure: ['role', 'content', 'reasoning_content']
2026-01-05 22:24:16,779 - INFO -   - license
2026-01-05 22:24:16,779 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,779 - INFO -     Value types: ['str']
2026-01-05 22:24:16,779 - INFO -     Is text: True
2026-01-05 22:24:16,779 - INFO -     Sample: cc-by-4.0
2026-01-05 22:24:16,779 - INFO -   - used_in
2026-01-05 22:24:16,779 - INFO -     Type: Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)
2026-01-05 22:24:16,779 - INFO -     Value types: ['list']
2026-01-05 22:24:16,779 - INFO -   - tools
2026-01-05 22:24:16,779 - INFO -     Type: Sequence(feature=Value(dtype='null', id=None), length=-1, id=None)
2026-01-05 22:24:16,779 - INFO -     Value types: ['list']
2026-01-05 22:24:16,779 - INFO -   - reasoning
2026-01-05 22:24:16,779 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,779 - INFO -     Value types: []
2026-01-05 22:24:16,779 - INFO -   - capability_target
2026-01-05 22:24:16,779 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,779 - INFO -     Value types: []
2026-01-05 22:24:16,779 - INFO - 
Message Structure:
2026-01-05 22:24:16,779 - INFO -   Roles: ['assistant', 'system', 'user']
2026-01-05 22:24:16,779 - INFO -   Message keys: ['content', 'reasoning_content', 'role']
2026-01-05 22:24:16,779 - INFO -   Sample conversation length: 3 messages
2026-01-05 22:24:16,779 - INFO - 
Recommended Embedding Columns: ['messages']
2026-01-05 22:24:16,779 - INFO - 
================================================================================
2026-01-05 22:24:16,779 - INFO - Dataset Summary: nvidia/Nemotron-Instruction-Following-Chat-v1
2026-01-05 22:24:16,779 - INFO -   Total rows: 430,978
2026-01-05 22:24:16,779 - INFO -   Number of splits: 2
2026-01-05 22:24:16,779 - INFO -   All columns: ['capability_target', 'license', 'messages', 'reasoning', 'tools', 'used_in', 'uuid']
2026-01-05 22:24:16,779 - INFO -   Embedding columns: ['messages']
2026-01-05 22:24:16,779 - INFO - ================================================================================
2026-01-05 22:24:16,784 - INFO - 
[6/6] Processing: nvidia/Nemotron-Math-Proofs-v1
2026-01-05 22:24:16,784 - INFO - 
================================================================================
2026-01-05 22:24:16,784 - INFO - Exploring: nvidia/Nemotron-Math-Proofs-v1
2026-01-05 22:24:16,784 - INFO - ================================================================================
2026-01-05 22:24:16,824 - INFO - 
Split: lean (1,376,663 rows)
2026-01-05 22:24:16,824 - INFO - Columns (13): ['problem', 'source', 'formal_statement', 'lean_header', 'url', 'user_name', 'user_url', 'sft_line_number', 'messages', 'uuid', 'used_in', 'tools', 'license']
2026-01-05 22:24:16,829 - INFO - 
Column Details:
2026-01-05 22:24:16,829 - INFO -   - problem
2026-01-05 22:24:16,829 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,829 - INFO -     Value types: ['str']
2026-01-05 22:24:16,829 - INFO -     Is text: True
2026-01-05 22:24:16,829 - INFO -     Sample: Prove that there exists a polynomial $ P \in \mathbb{Z}[X] $ such that the number 
$$
\sqrt[2003]{5 - 2\sqrt{6}} + \sqrt[2003]{5 + 2\sqrt{6}}
$$
is a root of $ P $.
2026-01-05 22:24:16,829 - INFO -   - source
2026-01-05 22:24:16,829 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,829 - INFO -     Value types: ['str']
2026-01-05 22:24:16,829 - INFO -     Is text: True
2026-01-05 22:24:16,829 - INFO -     Sample: aops
2026-01-05 22:24:16,829 - INFO -   - formal_statement
2026-01-05 22:24:16,829 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,829 - INFO -     Value types: ['str']
2026-01-05 22:24:16,829 - INFO -     Is text: True
2026-01-05 22:24:16,829 - INFO -     Sample: theorem problem_568570 :
    ∃ P : Polynomial ℤ,
      IsRoot (P.map (Int.castRingHom ℝ))
        ((5 - 2 * Real.sqrt (6 : ℝ)) ^ (1 / (2003 : ℝ)) +
          (5 + 2 * Real.sqrt (6 : ℝ)) ^ (1 / (2003 :...
2026-01-05 22:24:16,829 - INFO -   - lean_header
2026-01-05 22:24:16,829 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,829 - INFO -     Value types: ['str']
2026-01-05 22:24:16,829 - INFO -     Is text: True
2026-01-05 22:24:16,829 - INFO -     Sample: import Mathlib
import Aesop
import Mathlib
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat
open Polynomial
2026-01-05 22:24:16,829 - INFO -   - url
2026-01-05 22:24:16,829 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,829 - INFO -     Value types: ['str']
2026-01-05 22:24:16,829 - INFO -     Is text: True
2026-01-05 22:24:16,829 - INFO -     Sample: None
2026-01-05 22:24:16,829 - INFO -   - user_name
2026-01-05 22:24:16,829 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,829 - INFO -     Value types: ['str']
2026-01-05 22:24:16,829 - INFO -     Is text: True
2026-01-05 22:24:16,829 - INFO -     Sample: None
2026-01-05 22:24:16,829 - INFO -   - user_url
2026-01-05 22:24:16,829 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,829 - INFO -     Value types: ['str']
2026-01-05 22:24:16,829 - INFO -     Is text: True
2026-01-05 22:24:16,829 - INFO -     Sample: None
2026-01-05 22:24:16,829 - INFO -   - sft_line_number
2026-01-05 22:24:16,829 - INFO -     Type: Value(dtype='int64', id=None)
2026-01-05 22:24:16,829 - INFO -     Value types: []
2026-01-05 22:24:16,829 - INFO -   - messages
2026-01-05 22:24:16,829 - INFO -     Type: [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None), 'reasoning_content': Value(dtype='string', id=None)}]
2026-01-05 22:24:16,829 - INFO -     Value types: ['list']
2026-01-05 22:24:16,829 - INFO -   - uuid
2026-01-05 22:24:16,830 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,830 - INFO -     Value types: ['str']
2026-01-05 22:24:16,830 - INFO -     Is text: True
2026-01-05 22:24:16,830 - INFO -     Sample: b2ee7144-44aa-5d7f-8acc-812fae259c90
2026-01-05 22:24:16,830 - INFO -   - used_in
2026-01-05 22:24:16,830 - INFO -     Type: Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)
2026-01-05 22:24:16,830 - INFO -     Value types: ['list']
2026-01-05 22:24:16,830 - INFO -   - tools
2026-01-05 22:24:16,830 - INFO -     Type: Sequence(feature=Value(dtype='null', id=None), length=-1, id=None)
2026-01-05 22:24:16,830 - INFO -     Value types: ['list']
2026-01-05 22:24:16,830 - INFO -   - license
2026-01-05 22:24:16,830 - INFO -     Type: Value(dtype='string', id=None)
2026-01-05 22:24:16,830 - INFO -     Value types: ['str']
2026-01-05 22:24:16,830 - INFO -     Is text: True
2026-01-05 22:24:16,830 - INFO -     Sample: cc-by-4.0
2026-01-05 22:24:16,830 - INFO - 
Message Structure:
2026-01-05 22:24:16,830 - INFO -   Roles: []
2026-01-05 22:24:16,830 - INFO -   Message keys: []
2026-01-05 22:24:16,830 - INFO - 
Recommended Embedding Columns: ['problem', 'formal_statement']
2026-01-05 22:24:16,830 - INFO - 
================================================================================
2026-01-05 22:24:16,830 - INFO - Dataset Summary: nvidia/Nemotron-Math-Proofs-v1
2026-01-05 22:24:16,830 - INFO -   Total rows: 1,376,663
2026-01-05 22:24:16,830 - INFO -   Number of splits: 1
2026-01-05 22:24:16,830 - INFO -   All columns: ['formal_statement', 'lean_header', 'license', 'messages', 'problem', 'sft_line_number', 'source', 'tools', 'url', 'used_in', 'user_name', 'user_url', 'uuid']
2026-01-05 22:24:16,830 - INFO -   Embedding columns: ['formal_statement', 'problem']
2026-01-05 22:24:16,830 - INFO - ================================================================================
2026-01-05 22:24:16,838 - INFO - 
========================================================================================================================
2026-01-05 22:24:16,838 - INFO - SUMMARY TABLE: ALL DATASETS
2026-01-05 22:24:16,838 - INFO - ========================================================================================================================
2026-01-05 22:24:16,838 - INFO - Dataset                                                    Rows   Splits  All Columns  Embedding Cols
2026-01-05 22:24:16,838 - INFO - ------------------------------------------------------------------------------------------------------------------------
2026-01-05 22:24:16,838 - INFO - Llama-Nemotron-Post-Training-Dataset                 32,955,418        5            9               2
2026-01-05 22:24:16,838 - INFO - Nemotron-Post-Training-Dataset-v1                    25,659,642        5            8               1
2026-01-05 22:24:16,838 - INFO - Nemotron-Post-Training-Dataset-v2                     6,341,414        9            7               1
2026-01-05 22:24:16,838 - INFO - Nemotron-Science-v1                                     226,334        2            5               1
2026-01-05 22:24:16,838 - INFO - Nemotron-Instruction-Following-Chat-v1                  430,978        2            7               1
2026-01-05 22:24:16,838 - INFO - Nemotron-Math-Proofs-v1                               1,376,663        1           13               2
2026-01-05 22:24:16,838 - INFO - ========================================================================================================================
2026-01-05 22:24:16,838 - INFO - 
========================================================================================================================
2026-01-05 22:24:16,838 - INFO - DETAILED COLUMN INFORMATION
2026-01-05 22:24:16,838 - INFO - ========================================================================================================================
2026-01-05 22:24:16,838 - INFO - 
nvidia/Llama-Nemotron-Post-Training-Dataset:
2026-01-05 22:24:16,838 - INFO -   All columns: ['category', 'generator', 'input', 'license', 'output', 'reasoning', 'system_prompt', 'used_in_training', 'version']
2026-01-05 22:24:16,838 - INFO -   Embedding columns: ['output', 'system_prompt']
2026-01-05 22:24:16,838 - INFO -   Split column details:
2026-01-05 22:24:16,838 - INFO -     code: ['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt']
2026-01-05 22:24:16,838 - INFO -     math: ['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt']
2026-01-05 22:24:16,838 - INFO -     science: ['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt']
2026-01-05 22:24:16,838 - INFO -     chat: ['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt']
2026-01-05 22:24:16,838 - INFO -     safety: ['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt']
2026-01-05 22:24:16,838 - INFO - 
nvidia/Nemotron-Post-Training-Dataset-v1:
2026-01-05 22:24:16,838 - INFO -   All columns: ['category', 'generator', 'license', 'messages', 'metadata', 'reasoning', 'uuid', 'version']
2026-01-05 22:24:16,838 - INFO -   Embedding columns: ['messages']
2026-01-05 22:24:16,838 - INFO -   Split column details:
2026-01-05 22:24:16,838 - INFO -     chat: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages', 'metadata']
2026-01-05 22:24:16,838 - INFO -     code: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages', 'metadata']
2026-01-05 22:24:16,838 - INFO -     math: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages', 'metadata']
2026-01-05 22:24:16,838 - INFO -     stem: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages', 'metadata']
2026-01-05 22:24:16,839 - INFO -     tool_calling: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages', 'metadata']
2026-01-05 22:24:16,839 - INFO - 
nvidia/Nemotron-Post-Training-Dataset-v2:
2026-01-05 22:24:16,839 - INFO -   All columns: ['category', 'generator', 'license', 'messages', 'reasoning', 'uuid', 'version']
2026-01-05 22:24:16,839 - INFO -   Embedding columns: ['messages']
2026-01-05 22:24:16,839 - INFO -   Split column details:
2026-01-05 22:24:16,839 - INFO -     stem: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,839 - INFO -     chat: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,839 - INFO -     math: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,839 - INFO -     code: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,839 - INFO -     multilingual_ja: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,839 - INFO -     multilingual_de: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,839 - INFO -     multilingual_it: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,839 - INFO -     multilingual_es: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,839 - INFO -     multilingual_fr: ['uuid', 'license', 'generator', 'version', 'category', 'reasoning', 'messages']
2026-01-05 22:24:16,839 - INFO - 
nvidia/Nemotron-Science-v1:
2026-01-05 22:24:16,839 - INFO -   All columns: ['license', 'messages', 'tools', 'used_in', 'uuid']
2026-01-05 22:24:16,839 - INFO -   Embedding columns: ['messages']
2026-01-05 22:24:16,839 - INFO -   Split column details:
2026-01-05 22:24:16,839 - INFO -     MCQ: ['uuid', 'messages', 'license', 'used_in', 'tools']
2026-01-05 22:24:16,839 - INFO -     RQA: ['uuid', 'messages', 'license', 'used_in', 'tools']
2026-01-05 22:24:16,839 - INFO - 
nvidia/Nemotron-Instruction-Following-Chat-v1:
2026-01-05 22:24:16,839 - INFO -   All columns: ['capability_target', 'license', 'messages', 'reasoning', 'tools', 'used_in', 'uuid']
2026-01-05 22:24:16,839 - INFO -   Embedding columns: ['messages']
2026-01-05 22:24:16,839 - INFO -   Split column details:
2026-01-05 22:24:16,839 - INFO -     chat_if: ['uuid', 'messages', 'license', 'used_in', 'tools', 'reasoning', 'capability_target']
2026-01-05 22:24:16,839 - INFO -     structured_outputs: ['uuid', 'messages', 'license', 'used_in', 'tools', 'reasoning', 'capability_target']
2026-01-05 22:24:16,839 - INFO - 
nvidia/Nemotron-Math-Proofs-v1:
2026-01-05 22:24:16,839 - INFO -   All columns: ['formal_statement', 'lean_header', 'license', 'messages', 'problem', 'sft_line_number', 'source', 'tools', 'url', 'used_in', 'user_name', 'user_url', 'uuid']
2026-01-05 22:24:16,839 - INFO -   Embedding columns: ['formal_statement', 'problem']
2026-01-05 22:24:16,839 - INFO - 
Generated embedding extraction functions: /localhome/local-tranminhq/embeddings/embedding_extraction_functions.py
2026-01-05 22:24:16,839 - INFO - 
Detailed results saved to: /localhome/local-tranminhq/embeddings/dataset_exploration_detailed.json
2026-01-05 22:24:16,839 - INFO - 
========================================================================================================================
2026-01-05 22:24:16,839 - INFO - EXPLORATION COMPLETE
2026-01-05 22:24:16,839 - INFO - ========================================================================================================================
