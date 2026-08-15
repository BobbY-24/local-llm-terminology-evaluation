#!/usr/bin/env python3
"""Shared utilities for the community-terminology evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


CHOICE_LABELS = ("A", "B", "C", "D")
FAILURE_TAXONOMY = (
    "CORRECT",
    "PARTIAL_CORE_MEANING",
    "COMPOSITIONAL_LITERALIZATION",
    "WRONG_SENSE",
    "WRONG_COMMUNITY",
    "RELATED_BUT_DISTINCT_CONCEPT",
    "OVERGENERALIZATION",
    "HALLUCINATED_MECHANISM",
    "NO_KNOWLEDGE",
    "REFUSAL_OR_NO_ANSWER",
    "INSTRUCTION_FORMAT_FAILURE",
    "OTHER",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return config, config_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def write_jsonl_exclusive(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_completed_ids(path: str | Path) -> set[str]:
    output_path = Path(path)
    if not output_path.exists():
        return set()
    return {record["experiment_id"] for record in read_jsonl(output_path)}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def parse_choice(raw_output: str) -> dict[str, Any]:
    """Parse one unambiguous first A-D answer and preserve format diagnostics."""
    text = raw_output.strip()
    if not text:
        return {"parsed_choice": None, "format_error": True, "extra_generation": False, "unparseable": True}

    leading = re.match(
        r"^(?:(?:the\s+)?(?:answer|correct\s+option|option)\s*(?:is|:)?\s*)?[\(\[]?([ABCD])[\)\].:]?(?=\s|$)",
        text,
        flags=re.IGNORECASE,
    )
    if leading:
        choice = leading.group(1).upper()
        remainder = text[leading.end() :].strip()
        extra = bool(re.search(r"[A-Za-z0-9]", remainder))
        return {"parsed_choice": choice, "format_error": False, "extra_generation": extra, "unparseable": False}

    explicit = re.findall(
        r"(?:answer|correct\s+option|option)\s*(?:is|:)?\s*[\(\[]?([ABCD])[\)\]]?",
        text,
        flags=re.IGNORECASE,
    )
    distinct = {value.upper() for value in explicit}
    if len(distinct) == 1:
        return {"parsed_choice": distinct.pop(), "format_error": False, "extra_generation": True, "unparseable": False}

    first_line = text.splitlines()[0].strip()
    standalone = {value.upper() for value in re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", first_line, flags=re.IGNORECASE)}
    if len(standalone) == 1:
        return {"parsed_choice": standalone.pop(), "format_error": False, "extra_generation": True, "unparseable": False}
    return {"parsed_choice": None, "format_error": True, "extra_generation": bool(text), "unparseable": True}


def parse_yes_no(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    if not text:
        return {"parsed_answer": None, "format_error": True, "extra_generation": False, "unparseable": True}
    leading = re.match(r"^(?:(?:the\s+)?answer\s*(?:is|:)?\s*)?(YES|NO)(?=\W|$)", text, flags=re.IGNORECASE)
    if leading:
        answer = leading.group(1).upper()
        remainder = text[leading.end() :].strip()
        return {
            "parsed_answer": answer,
            "format_error": False,
            "extra_generation": bool(re.search(r"[A-Za-z0-9]", remainder)),
            "unparseable": False,
        }
    found = {value.upper() for value in re.findall(r"\b(YES|NO)\b", text, flags=re.IGNORECASE)}
    if len(found) == 1:
        return {"parsed_answer": found.pop(), "format_error": False, "extra_generation": True, "unparseable": False}
    return {"parsed_answer": None, "format_error": True, "extra_generation": bool(text), "unparseable": True}

