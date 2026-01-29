# Activation cache (residual stream)

When you run inference with `--cache_residual_stream`, we save the **last layer** last-token representation only (single tensor `[1, hidden_size]`), plus **question**, **oracle answer**, and **model prediction** in a clean format for modeling (probing, interventions, etc.).

## What to do next

1. **Dry run** (already works):  
   `python run_all_tasks.py --dry_run [--first_k N]`
2. **Quick test**:  
   `python run_all_tasks.py --first_k 5 --cache_residual_stream`  
   Then inspect `outputs/residual_stream/.../*.pt` and load one with `torch.load(..., weights_only=True)`.
3. **Full run**:  
   `python run_all_tasks.py --cache_residual_stream`  
   Use `--lm_only` or `--vlm_only` if you want to restrict to LMs or VLMs.

## Where activations are stored

```
{cache_dir} / {dataset_id} / {model_id} / {mode} / {task} / {example_key}.pt
```

- **cache_dir**: `--cache_dir` if set, else `<output_folder>/residual_stream`.
- **dataset_id**: e.g. `MilaWang__SpatialEval` (slashes → `__`).
- **model_id**: e.g. `meta-llama__Meta-Llama-3-8B-Instruct`.
- **mode**: `tqa`, `vqa`, or `vtqa`.
- **task**: `spatialmap`, `mazenav`, `spatialgrid`, or `spatialreal` (from example id).
- **example_key**: `{example_id}_{index}` with dots replaced by underscores (e.g. `spatialmap_0_42`).

Example:

```
outputs/residual_stream/MilaWang__SpatialEval/meta-llama__Meta-Llama-3-8B-Instruct/tqa/spatialmap/spatialmap_0_0.pt
```

## Format of each `.pt` file

Each file is a `torch.save` dict. Load with `torch.load(path, weights_only=True)` (or `weights_only=False` if you need legacy pickle).

| Key | Type | Description |
|-----|------|-------------|
| `question` | `str` | Raw SpatialEval question (`item['text']`). |
| `oracle_answer` | `str` | Ground-truth answer. |
| `model_prediction` | `str` | Model-generated answer. |
| `example_id` | `str` | e.g. `spatialmap.0`. |
| `index` | `int` | Index within the question group. |
| `task` | `str` | `spatialmap`, `mazenav`, `spatialgrid`, `spatialreal`. |
| `mode` | `str` | `tqa`, `vqa`, or `vtqa`. |
| `hidden_states_last` | `Tensor` | Last layer, last-token representation only. Shape `[1, hidden_size]`. |
| `hidden_size` | `int` | Model hidden size. |

We store only the **last token** of the **last layer** (pre-generation), not the full sequence or other layers.

## Example: load and use for modeling

```python
import torch

path = "outputs/residual_stream/.../spatialmap_0_0.pt"
data = torch.load(path, weights_only=True)

question = data["question"]
oracle = data["oracle_answer"]
pred = data["model_prediction"]
# [1, hidden_size] — last layer, last token only
hidden_last = data["hidden_states_last"]
```

## Supported when caching

- **LM (text-only)**: Llama, Qwen, etc. — all paths that use `inference_lm.py`.
- **VLM**: Bunny HF path only (`inference_vlm.py`). LLaVA and Qwen-VL use chat/generate APIs; cache not implemented there.
