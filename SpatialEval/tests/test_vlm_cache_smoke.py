#!/usr/bin/env python3
"""
Small smoke tests for VLM inference and activation caching (Llama 3.2 Vision, Qwen2.5-VL).
Run from SpatialEval: python test_vlm_cache_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure we can import from parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_model_detection():
    """Test 1: Model path detection (no heavy deps)."""
    def _is_llama32_vision(model_path: str) -> bool:
        p = model_path.lower()
        return ("llama-3.2" in p or "llama_3.2" in p) and "vision" in p

    def _is_qwen25_vl(model_path: str) -> bool:
        p = model_path.lower()
        return "qwen2.5-vl" in p or "qwen2_5_vl" in p

    assert _is_llama32_vision("meta-llama/Llama-3.2-11B-Vision-Instruct") is True
    assert _is_llama32_vision("other/model") is False
    assert _is_qwen25_vl("Qwen/Qwen2.5-VL-7B-Instruct") is True
    assert _is_qwen25_vl("Qwen/Qwen2.5-VL-3B-Instruct") is True
    assert _is_qwen25_vl("Qwen/Qwen2-VL-7B-Instruct") is False  # legacy, not 2.5
    print("  [PASS] Model detection (_is_llama32_vision, _is_qwen25_vl)")


def test_residual_stream_helpers():
    """Test 2: residual_stream cache helpers (needs torch)."""
    try:
        import torch
    except ImportError:
        print("  [SKIP] residual_stream (torch not installed)")
        return
    from utils.residual_stream import (
        cache_dir_for_model_mode_task,
        example_key,
        _last_layer_last_token,
        save_activations_for_modeling,
    )
    # Cache dir layout
    base = cache_dir_for_model_mode_task(
        "out", "Qwen/Qwen2.5-VL-3B-Instruct", "vqa", "spatialmap", "MilaWang/SpatialEval"
    )
    assert base == Path("out/MilaWang__SpatialEval/Qwen__Qwen2.5-VL-3B-Instruct/vqa/spatialmap")
    assert example_key("spatialmap.0", 1) == "spatialmap_0_1"
    # Last-token extraction
    B, T, D = 1, 10, 64
    fake_states = (torch.randn(B, T, D), torch.randn(B, T, D))
    last = _last_layer_last_token(fake_states)
    assert last.shape == (1, 1, D)
    assert last.dtype == torch.float32
    # Save (to temp file)
    tmp = Path("/tmp/test_vlm_cache_smoke.pt")
    save_activations_for_modeling(
        tmp,
        question="Q?",
        oracle_answer="A",
        model_prediction="P",
        hidden_states=fake_states,
        example_id="x.0",
        index=0,
        task="spatialmap",
        mode="vqa",
    )
    assert tmp.exists()
    data = torch.load(tmp, weights_only=True)
    assert data["question"] == "Q?"
    assert data["hidden_states_last"].shape == (1, 1, D)
    tmp.unlink(missing_ok=True)
    print("  [PASS] residual_stream (cache_dir, example_key, save/load .pt)")


def test_inference_vlm_import_and_cli():
    """Test 3: inference_vlm.py parses CLI and exposes detection (needs full deps)."""
    try:
        from inference_vlm import _is_llama32_vision, _is_qwen25_vl
    except ImportError as e:
        print("  [SKIP] inference_vlm import:", e)
        return
    assert _is_llama32_vision("meta-llama/Llama-3.2-11B-Vision-Instruct")
    assert _is_qwen25_vl("Qwen/Qwen2.5-VL-3B-Instruct")
    print("  [PASS] inference_vlm import and detection")


def test_run_one_example_qwen25vl_3b():
    """Test 4: Run 1 example with Qwen2.5-VL-3B (tqa, no image) + cache. Needs HF/transformers/torch."""
    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from datasets import load_dataset
    except ImportError as e:
        print("  [SKIP] Run 1 example (missing deps):", e)
        return
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading {model_id} on {device}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    # One tqa example (text only)
    ds = load_dataset("MilaWang/SpatialEval", "tqa", split="test")
    ds = ds.filter(lambda x: "spatialmap" in x["id"])
    item = next(iter(ds))
    from inference_vlm import _build_hf_vlm_messages, _prepare_hf_vlm_inputs
    prompt = item["text"]
    messages = _build_hf_vlm_messages(prompt, image=None)
    model_device = next(model.parameters()).device
    inputs = _prepare_hf_vlm_inputs(processor, messages, model_device)
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)
        hidden_states = out.hidden_states
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    answer_text = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    from utils.residual_stream import _last_layer_last_token
    last = _last_layer_last_token(hidden_states)
    assert last.shape[0] == 1 and last.shape[-1] == model.config.text_config.hidden_size
    print("  [PASS] One example: forward + generate + last-token shape OK; answer length =", len(answer_text))


def main():
    print("Test 1: Model detection")
    test_model_detection()
    print("Test 2: residual_stream helpers")
    test_residual_stream_helpers()
    print("Test 3: inference_vlm import + CLI detection")
    test_inference_vlm_import_and_cli()
    print("Test 4: One example (Qwen2.5-VL-3B, tqa, with cache)")
    test_run_one_example_qwen25vl_3b()
    print("\nAll requested tests finished.")


if __name__ == "__main__":
    main()
