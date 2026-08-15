#!/usr/bin/env python3
"""Run resumable, sequential local MLX terminology experiments."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from terminology_eval_common import (
    load_completed_ids,
    load_config,
    project_root,
    read_jsonl,
    resolve_project_path,
    sha256_file,
)


EXPERIMENT_FILES = {
    "masked_recovery": "masked_recovery.jsonl",
    "definition_ablation": "definition_ablation.jsonl",
    "context_compatibility": "context_compatibility.jsonl",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generation_limit(config: dict[str, Any], experiment: str) -> int:
    key = {
        "masked_recovery": "max_new_tokens_mcq",
        "definition_ablation": "max_new_tokens_definition",
        "context_compatibility": "max_new_tokens_compatibility",
    }[experiment]
    return int(config["generation"][key])


def generate_one(model: Any, tokenizer: Any, prompt: str, config: dict[str, Any], max_tokens: int) -> tuple[str, dict[str, Any] | None]:
    generation = config["generation"]
    sampler = make_sampler(
        temp=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        top_k=int(generation["top_k"]),
    )
    text = ""
    last = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        max_kv_size=int(config["model"]["max_kv_size"]),
    ):
        text += response.text
        last = response
    if last is None:
        return text, None
    return text, {
        "prompt_tokens": last.prompt_tokens,
        "generation_tokens": last.generation_tokens,
        "prompt_tokens_per_second": last.prompt_tps,
        "generation_tokens_per_second": last.generation_tps,
        "peak_memory_gb": last.peak_memory,
        "finish_reason": last.finish_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/terminology_eval.yaml")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENT_FILES), required=True)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only: stop after this many selected items")
    parser.add_argument("--run-id", default=None, help="Override the configured run ID (useful for smoke tests)")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    root = project_root()
    config, config_path = load_config(resolve_project_path(root, args.config))
    processed_dir = resolve_project_path(root, config["paths"]["processed_dir"])
    data_path = processed_dir / EXPERIMENT_FILES[args.experiment]
    if not data_path.is_file():
        raise SystemExit(f"Processed data missing: {data_path}. Run build_eval_dataset.py first.")
    items = read_jsonl(data_path)
    if args.limit is not None:
        items = items[: args.limit]

    run_id = args.run_id or str(config["run_id"])
    results_root = resolve_project_path(root, config["paths"]["results_root"])
    raw_dir = results_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{run_id}_{args.experiment}.jsonl"
    metadata_path = raw_dir / f"{run_id}_{args.experiment}.metadata.json"
    completed = load_completed_ids(output_path)
    pending = [item for item in items if item["experiment_id"] not in completed]

    model_path = resolve_project_path(root, config["model"]["path"])
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Local model missing: {model_path}")
    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    config_hash = sha256_file(config_path)
    data_hash = sha256_file(data_path)
    metadata = {
        "run_id": run_id,
        "experiment": args.experiment,
        "started_utc": utc_now(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "data_path": str(data_path),
        "data_sha256": data_hash,
        "selected_items": len(items),
        "already_completed": len(completed & {item["experiment_id"] for item in items}),
        "model": {
            **config["model"],
            "resolved_path": str(model_path),
            "architecture": model_config.get("model_type"),
            "quantization": model_config.get("quantization") or model_config.get("quantization_config"),
        },
        "generation": config["generation"],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "transformers": importlib.metadata.version("transformers"),
        },
        "offline_flags": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != config_hash or existing.get("data_sha256") != data_hash:
            raise SystemExit(f"Refusing to resume {run_id}: config or processed dataset hash changed.")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "experiment": args.experiment,
                "selected": len(items),
                "completed": len(items) - len(pending),
                "pending": len(pending),
                "output": str(output_path),
            }
        ),
        flush=True,
    )
    if not pending:
        print("Nothing to do; all selected experiment IDs are already present.", flush=True)
        return

    mx.random.seed(int(config["generation"]["seed"]))
    model, tokenizer = load(str(model_path))
    max_tokens = generation_limit(config, args.experiment)
    started = time.monotonic()
    output_mode = "a" if output_path.exists() else "x"
    with output_path.open(output_mode, encoding="utf-8") as handle:
        for index, item in enumerate(pending, 1):
            raw_output, metrics = generate_one(model, tokenizer, item["prompt"], config, max_tokens)
            record = {
                **item,
                "run_id": run_id,
                "run_timestamp_utc": utc_now(),
                "raw_output": raw_output,
                "model": metadata["model"],
                "runtime": metadata["runtime"],
                "decoding": {
                    "do_sample": False,
                    "temperature": float(config["generation"]["temperature"]),
                    "top_p": float(config["generation"]["top_p"]),
                    "top_k": int(config["generation"]["top_k"]),
                    "seed": int(config["generation"]["seed"]),
                    "max_new_tokens": max_tokens,
                    "max_kv_size": int(config["model"]["max_kv_size"]),
                    "chat_template_applied": False,
                },
                "generation_metrics": metrics,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index % args.progress_every == 0 or index == len(pending):
                elapsed = time.monotonic() - started
                rate = index / elapsed if elapsed else 0.0
                eta = (len(pending) - index) / rate if rate else 0.0
                print(
                    f"[{index}/{len(pending)}] {item['experiment_id']} elapsed={elapsed:.1f}s eta={eta:.1f}s output={raw_output!r}",
                    flush=True,
                )
            if index % 50 == 0:
                gc.collect()
                mx.clear_cache()

    print(f"Completed {len(pending)} new items. Results: {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted safely; completed JSONL records were flushed and the run can be resumed.", flush=True)
        raise SystemExit(130)
