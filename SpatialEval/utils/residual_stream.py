"""
Utilities for caching the residual stream during inference.
Used for further modeling (probing, interventions, etc.).

We store only the *last layer* last-token representation (shape [1, hidden_size]).
Each .pt file also contains: question, oracle_answer, model_prediction, plus metadata.

Cache layout:
  {cache_dir} / {dataset_id} / {model_id} / {mode} / {task} / {example_key}.pt

Example key: e.g. spatialmap_0_42 (from example_id + index).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def cache_dir_for_model_mode_task(
    cache_root: str,
    model_id: str,
    mode: str,
    task: str,
    dataset_id: str = "MilaWang/SpatialEval",
) -> Path:
    """Base cache directory: {cache_root}/{dataset_id}/{model_id}/{mode}/{task}/."""
    safe_model = model_id.replace("/", "__")
    safe_ds = dataset_id.replace("/", "__")
    return Path(cache_root) / safe_ds / safe_model / mode / task


def example_key(example_id: str, index: int) -> str:
    """Filename-safe unique key per example (e.g. spatialmap_0_42)."""
    safe_id = example_id.replace(".", "_")
    return f"{safe_id}_{index}"


def _last_layer_last_token(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Extract last-token representation of the last layer. Shape [1, hidden_size]."""
    h = hidden_states[-1]  # [batch, seq_len, hidden_size]
    return h[:, -1:, :].detach().float().cpu()


def save_activations_for_modeling(
    cache_path: Path,
    *,
    question: str,
    oracle_answer: str | int | float,
    model_prediction: str,
    hidden_states: tuple[torch.Tensor, ...],
    example_id: str,
    index: int,
    task: str,
    mode: str,
) -> None:
    """
    Save activations for modeling: last layer, last-token representation only, plus
    question, oracle answer, and model prediction.

    hidden_states: tuple (embedding_out, layer_0_out, ..., layer_L_out), each [B, T, D].
    We store only the last token of the last layer, shape [1, D].
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    last = _last_layer_last_token(hidden_states)
    _, _, hidden_size = hidden_states[0].shape

    payload: dict[str, Any] = {
        "question": question,
        "oracle_answer": str(oracle_answer),
        "model_prediction": model_prediction,
        "example_id": example_id,
        "index": index,
        "task": task,
        "mode": mode,
        "hidden_states_last": last,
        "hidden_size": int(hidden_size),
    }
    torch.save(payload, cache_path)


def run_forward_return_hidden_states(model, inputs: dict) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """
    Run forward with output_hidden_states=True.
    Returns (logits_or_used, hidden_states).
    """
    model_device = next(model.parameters()).device
    inp = {k: v.to(model_device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inp, output_hidden_states=True)
    return out.logits, out.hidden_states


def run_forward_return_hidden_states_vlm(model, inputs_dict: dict) -> tuple[Any, tuple[torch.Tensor, ...]]:
    """VLM forward with output_hidden_states. Returns (logits_or_used, hidden_states)."""
    model_device = next(model.parameters()).device
    inp = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs_dict.items()}
    with torch.inference_mode():
        out = model(**inp, output_hidden_states=True)
    return out.logits, out.hidden_states
