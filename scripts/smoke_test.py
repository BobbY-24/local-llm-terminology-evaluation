#!/usr/bin/env python3
"""Short, offline completion test for the installed base model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# The default model is a local directory. These flags make accidental Hub use fail.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_DIR / "models" / "Qwen2.5-3B-4bit"
DEFAULT_PROMPT = "Gaming glossary\n\nTerm: stat stick\nDefinition:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-kv-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Local model not found: {model_path}")

    config = json.loads((model_path / "config.json").read_text())
    mx.random.seed(args.seed)
    model, tokenizer = load(str(model_path))
    sampler = make_sampler(temp=args.temperature)

    text = ""
    last = None
    for response in stream_generate(
        model,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        max_kv_size=args.max_kv_size,
    ):
        text += response.text
        last = response

    quant = config.get("quantization", {})
    print(f"Model: {model_path}")
    print(f"Quantization: {quant.get('bits', 'unknown')}-bit, group size {quant.get('group_size', 'unknown')}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Output: {text!r}")
    if last is not None:
        print(
            "Metrics: "
            f"prompt_tokens={last.prompt_tokens}, "
            f"generation_tokens={last.generation_tokens}, "
            f"generation_tps={last.generation_tps:.2f}, "
            f"peak_memory_gb={last.peak_memory:.3f}, "
            f"finish_reason={last.finish_reason}"
        )


if __name__ == "__main__":
    main()
