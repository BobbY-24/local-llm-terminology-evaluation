#!/usr/bin/env python3
"""Score objective tasks and create a blank human-scoring artifact for definitions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from terminology_eval_common import FAILURE_TAXONOMY, load_config, parse_choice, parse_yes_no, project_root, read_jsonl, resolve_project_path


def write_scored(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"Preserving existing scored file: {path}")
        return
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(records)} records: {path}")


def score_mcq(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for record in records:
        parsed = parse_choice(record.get("raw_output", ""))
        scored.append(
            {
                **record,
                **parsed,
                "correct": parsed["parsed_choice"] == record["gold_choice"],
                "scoring_rule": "first unambiguous generated A-D choice; unparseable outputs count as incorrect",
            }
        )
    return scored


def score_compatibility(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for record in records:
        parsed = parse_yes_no(record.get("raw_output", ""))
        scored.append(
            {
                **record,
                **parsed,
                "correct": parsed["parsed_answer"] == record["gold_answer"],
                "scoring_rule": "first unambiguous YES/NO answer; unparseable outputs count as incorrect",
            }
        )
    return scored


def write_human_scoring(path: Path, definitions: list[dict[str, Any]]) -> None:
    if path.exists():
        print(f"Preserving existing human-scoring artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_id",
        "item_id",
        "term",
        "domain",
        "linguistic_construction",
        "community",
        "condition",
        "model_output",
        "semantic_score",
        "failure_category",
        "notes",
    ]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in definitions:
            writer.writerow(
                {
                    "experiment_id": record["experiment_id"],
                    "item_id": record["item_id"],
                    "term": record["term"],
                    "domain": record["domain"],
                    "linguistic_construction": record["linguistic_construction"],
                    "community": record["community"],
                    "condition": record["condition"],
                    "model_output": record.get("raw_output", ""),
                    "semantic_score": "",
                    "failure_category": "",
                    "notes": "",
                }
            )
    taxonomy_path = path.with_name("failure_taxonomy.json")
    if not taxonomy_path.exists():
        taxonomy_path.write_text(
            json.dumps(
                {
                    "semantic_rubric": {
                        "2": "Core community-specific meaning correctly recovered",
                        "1": "Partially correct or relevant, but an important semantic feature is missing",
                        "0": "Incorrect meaning",
                    },
                    "failure_categories": list(FAILURE_TAXONOMY),
                    "instruction": "Assign semantic_score and nuanced failure_category manually; do not infer them from keywords.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {len(definitions)} blank human-scoring rows: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/terminology_eval.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    root = project_root()
    config, _ = load_config(resolve_project_path(root, args.config))
    run_id = args.run_id or str(config["run_id"])
    results_root = resolve_project_path(root, config["paths"]["results_root"])
    raw_dir = results_root / "raw"
    scored_dir = results_root / "scored"

    masked_path = raw_dir / f"{run_id}_masked_recovery.jsonl"
    compatibility_path = raw_dir / f"{run_id}_context_compatibility.jsonl"
    definitions_path = raw_dir / f"{run_id}_definition_ablation.jsonl"

    if masked_path.exists():
        write_scored(scored_dir / f"{run_id}_masked_recovery_scored.jsonl", score_mcq(read_jsonl(masked_path)))
    if compatibility_path.exists():
        write_scored(
            scored_dir / f"{run_id}_context_compatibility_scored.jsonl",
            score_compatibility(read_jsonl(compatibility_path)),
        )
    if definitions_path.exists():
        write_human_scoring(scored_dir / f"{run_id}_definition_human_scoring.csv", read_jsonl(definitions_path))


if __name__ == "__main__":
    main()
