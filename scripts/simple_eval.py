#!/usr/bin/env python3
"""Run a tiny, reproducible local terminology evaluation and write JSONL."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Fail instead of reaching the Hub when a local path is missing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx
import mlx.core as mx
import mlx_lm
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_DIR / "models" / "Qwen2.5-3B-4bit"
DEFAULT_DATA = PROJECT_DIR / "data" / "pilot_terms.jsonl"
DEFAULT_RESULTS = PROJECT_DIR / "results"
DEFAULT_REPOSITORY_ID = "mlx-community/Qwen2.5-3B-4bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--repository-id", default=DEFAULT_REPOSITORY_ID)
    parser.add_argument("--model-stage", choices=("base", "instruct"), default="base")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-kv-size", type=int, default=512)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def make_prompt(item: dict[str, Any]) -> str:
    template = item.get("template", "definition_completion")
    term = item.get("term", "")
    domain = item.get("domain", "gaming")
    if template == "definition_completion":
        heading = "Gaming" if domain == "gaming" else domain.title()
        return f"{heading} glossary\n\nTerm: {term}\nDefinition:"
    if template == "natural_continuation":
        return f'In {domain}, "{term}" refers to'
    if template == "mcq":
        question = item.get("question")
        options = item.get("options")
        if not question or not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"MCQ item {item.get('id', '<unknown>')} needs question and options")
        labeled = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))
        return f"Question: {question}\n{labeled}\nAnswer:"
    raise ValueError(f"Unknown template {template!r} in item {item.get('id', '<unknown>')}")


def discover_revision(model_path: Path) -> str | None:
    tree_dir = model_path / ".cache" / "huggingface" / "trees"
    trees = sorted(tree_dir.glob("*.json")) if tree_dir.is_dir() else []
    return trees[-1].stem if trees else None


def unique_result_path(results_dir: Path, model_name: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")
    candidate = results_dir / f"pilot_{slug}_{stamp}.jsonl"
    counter = 1
    while candidate.exists():
        candidate = results_dir / f"pilot_{slug}_{stamp}_{counter}.jsonl"
        counter += 1
    return candidate


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Local model not found: {model_path}")
    if not data_path.is_file():
        raise SystemExit(f"Evaluation data not found: {data_path}")

    config = json.loads((model_path / "config.json").read_text())
    items = read_jsonl(data_path)
    revision = discover_revision(model_path)
    quantization = config.get("quantization") or config.get("quantization_config") or {}
    run_started = datetime.now(timezone.utc).isoformat()
    output_path = unique_result_path(results_dir, model_path.name)

    mx.random.seed(args.seed)
    model, tokenizer = load(str(model_path))
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p, top_k=args.top_k)

    with output_path.open("x", encoding="utf-8") as output:
        for index, item in enumerate(items):
            prompt = make_prompt(item)
            generated = ""
            last = None
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                max_kv_size=args.max_kv_size,
            ):
                generated += response.text
                last = response

            record = {
                "run_started_utc": run_started,
                "item_index": index,
                "item": item,
                "model": {
                    "family": "Qwen2.5",
                    "repository_id": args.repository_id,
                    "revision": revision,
                    "local_path": str(model_path),
                    "stage": args.model_stage,
                    "parameter_count": "3.09B",
                    "quantization": quantization,
                    "architecture": config.get("model_type"),
                },
                "runtime": {
                    "python": platform.python_version(),
                    "mlx": importlib.metadata.version("mlx"),
                    "mlx_lm": importlib.metadata.version("mlx-lm"),
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                },
                "decoding": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "seed": args.seed,
                    "max_tokens": args.max_tokens,
                    "max_kv_size": args.max_kv_size,
                    "chat_template_applied": False,
                },
                "prompt": prompt,
                "generated_output": generated,
                "generation_metrics": None
                if last is None
                else {
                    "prompt_tokens": last.prompt_tokens,
                    "generation_tokens": last.generation_tokens,
                    "prompt_tokens_per_second": last.prompt_tps,
                    "generation_tokens_per_second": last.generation_tps,
                    "peak_memory_gb": last.peak_memory,
                    "finish_reason": last.finish_reason,
                },
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{index + 1}/{len(items)}] {item.get('id', '<unnamed>')}: {generated!r}")

    print(f"Results: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; any completed JSONL rows remain in the timestamped file.", file=sys.stderr)
        raise SystemExit(130)
