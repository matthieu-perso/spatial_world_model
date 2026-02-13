#!/usr/bin/env python3
"""
Run SpatialEval on all tasks with Llama + Qwen (text-only and multimodal), optionally cache residual stream.

- LM (text-only): Llama, Qwen. Modes: tqa only. Tasks: all (spatialmap, mazenav, spatialgrid, spatialreal).
- VLM (multimodal): Llama 3.2 Vision, Qwen2.5-VL (7B, 3B). Modes: tqa, vqa, vtqa. Tasks: all.

Residual-stream cache: use --cache_residual_stream. Saves last-layer last-token activations to --cache_dir
(default <output_folder>/residual_stream). LM and the three VLMs above all support caching.

Usage:
  cd SpatialEval
  python run_all_tasks.py [--cache_residual_stream] [--cache_dir DIR] [--first_k N] [--output_folder DIR] ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Default HF model IDs: Llama + Qwen, modal and multimodal
MODELS_LM = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
]
MODELS_VLM = [
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
]


def _run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--output_folder", type=str, default="outputs")
    ap.add_argument("--dataset_id", type=str, default="MilaWang/SpatialEval")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--task", type=str, default="all", choices=["all", "spatialmap", "mazenav", "spatialgrid", "spatialreal"])
    ap.add_argument("--first_k", type=int, default=None, help="Limit examples per question type (for quick runs)")
    ap.add_argument("--cache_residual_stream", action="store_true")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--lm_only", action="store_true", help="Run only LMs (Llama, Qwen text)")
    ap.add_argument("--vlm_only", action="store_true", help="Run only VLMs (LLaVA, Qwen-VL)")
    ap.add_argument("--dry_run", action="store_true", help="Print commands only, do not run")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    dry = args.dry_run
    base = [
        "--output_folder", args.output_folder,
        "--dataset_id", args.dataset_id,
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--repetition_penalty", str(args.repetition_penalty),
        "--max_new_tokens", str(args.max_new_tokens),
        "--task", args.task,
        "--w_reason",
    ]
    if args.first_k is not None:
        base += ["--first_k", str(args.first_k)]
    if args.cache_residual_stream:
        base += ["--cache_residual_stream"]
    if args.cache_dir:
        base += ["--cache_dir", args.cache_dir]

    # LMs: tqa only
    if not args.vlm_only:
        for model in MODELS_LM:
            cmd = [
                sys.executable, "inference_lm.py",
                "--model-path", model,
                "--device", args.device,
                "--mode", "tqa",
            ] + base
            _run(cmd, root, dry_run=dry)

    # VLMs: tqa, vqa, vtqa
    if not args.lm_only:
        for model in MODELS_VLM:
            for mode in ["tqa", "vqa", "vtqa"]:
                cmd = [
                    sys.executable, "inference_vlm.py",
                    "--model_path", model,
                    "--device", args.device,
                    "--mode", mode,
                ] + base
                _run(cmd, root, dry_run=dry)

    print("run_all_tasks done.")


if __name__ == "__main__":
    main()
