#!/usr/bin/env python3
"""Validate the source workbook and build deterministic terminology tasks."""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from terminology_eval_common import CHOICE_LABELS, load_config, project_root, resolve_project_path, stable_int


SHEETS = ("Cybersecurity", "Gaming")
EXPECTED_HEADERS = ["Term", "Linguistic Construction", "Community / Sub-community"] + [
    value for number in range(1, 9) for value in (f"Real Usage {number}", f"Source Link {number}")
]
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).strip().casefold().split())


def valid_url(value: str) -> bool:
    try:
        parsed = urlparse(value.strip())
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and " " not in value


def column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference.upper())
    if not letters:
        raise ValueError(f"Invalid cell reference: {cell_reference}")
    result = 0
    for char in letters.group(0):
        result = result * 26 + ord(char) - 64
    return result - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in item.findall(".//m:t", NS)) for item in root.findall("m:si", NS)]


def read_xlsx(path: str | Path) -> dict[str, list[list[Any]]]:
    """Read displayed cell values from a static XLSX using only the standard library."""
    workbook_path = Path(path)
    with zipfile.ZipFile(workbook_path) as archive:
        shared = _shared_strings(archive)
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rels = {node.attrib["Id"]: node.attrib["Target"] for node in rel_root.findall("r:Relationship", REL_NS)}
        result: dict[str, list[list[Any]]] = {}
        for sheet_node in workbook_root.findall("m:sheets/m:sheet", NS):
            name = sheet_node.attrib["name"]
            rel_id = sheet_node.attrib[f"{{{OFFICE_REL}}}id"]
            target = rels[rel_id].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            sheet_root = ET.fromstring(archive.read(target))
            rows: list[list[Any]] = []
            for row_node in sheet_root.findall(".//m:sheetData/m:row", NS):
                row_number = int(row_node.attrib.get("r", len(rows) + 1))
                while len(rows) < row_number:
                    rows.append([])
                values = rows[row_number - 1]
                for cell in row_node.findall("m:c", NS):
                    index = column_index(cell.attrib["r"])
                    while len(values) <= index:
                        values.append(None)
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find("m:v", NS)
                    if cell_type == "inlineStr":
                        value: Any = "".join(node.text or "" for node in cell.findall(".//m:t", NS))
                    elif value_node is None:
                        value = None
                    elif cell_type == "s":
                        value = shared[int(value_node.text or 0)]
                    elif cell_type == "b":
                        value = value_node.text == "1"
                    elif cell_type in {"str", "e"}:
                        value = value_node.text or ""
                    else:
                        raw = value_node.text or ""
                        try:
                            number = float(raw)
                            value = int(number) if number.is_integer() else number
                        except ValueError:
                            value = raw
                    values[index] = value
            result[name] = rows
    return result


def _term_patterns(term: str) -> list[tuple[str, re.Pattern[str]]]:
    prefix = r"(?<!\w)" if term and term[0].isalnum() else ""
    suffix = r"(?!\w)" if term and term[-1].isalnum() else ""
    patterns = [("exact", re.compile(prefix + re.escape(term) + suffix, re.IGNORECASE))]
    tokens = [token for token in re.split(r"[\s-]+", term.strip()) if token]
    if len(tokens) > 1:
        patterns.append(("hyphen_space_variant", re.compile(prefix + r"[\s-]+".join(map(re.escape, tokens)) + suffix, re.IGNORECASE)))
    if term and term[-1].isalpha() and not term.casefold().endswith("s"):
        patterns.append(("simple_plural_s", re.compile(prefix + re.escape(term) + r"s(?!\w)", re.IGNORECASE)))
    return patterns


def safe_mask_term(text: str, term: str) -> tuple[str | None, str | None, int]:
    for rule, pattern in _term_patterns(term):
        matches = list(pattern.finditer(text))
        if matches:
            masked, count = pattern.subn("[TERM]", text)
            if any(candidate.search(masked) for _, candidate in _term_patterns(term)):
                return None, "residual_target_after_masking", 0
            return masked, rule, count
    return None, "canonical_term_absent", 0


def _candidate_allowed(target: str, candidate: str) -> bool:
    left = re.sub(r"\W+", "", normalize_text(target))
    right = re.sub(r"\W+", "", normalize_text(candidate))
    if not left or not right or left == right:
        return False
    return left not in right and right not in left


def choose_distractors(target: dict[str, Any], terms: list[dict[str, Any]], seed: int, count: int = 3) -> list[dict[str, str]]:
    buckets = [
        ("same_community", lambda row: normalize_text(row["community"]) == normalize_text(target["community"])),
        (
            "same_domain_construction",
            lambda row: row["domain"] == target["domain"]
            and normalize_text(row["linguistic_construction"]) == normalize_text(target["linguistic_construction"]),
        ),
        ("same_domain", lambda row: row["domain"] == target["domain"]),
        ("cross_domain_fallback", lambda row: True),
    ]
    rng = random.Random(seed ^ stable_int(target["item_id"]))
    selected: list[dict[str, str]] = []
    selected_keys: set[str] = {normalize_text(target["term"])}
    for priority, predicate in buckets:
        candidates = [
            row
            for row in terms
            if predicate(row)
            and normalize_text(row["term"]) not in selected_keys
            and _candidate_allowed(target["term"], row["term"])
        ]
        candidates.sort(key=lambda row: row["item_id"])
        rng.shuffle(candidates)
        for row in candidates:
            key = normalize_text(row["term"])
            if key in selected_keys:
                continue
            selected.append({"term": row["term"], "item_id": row["item_id"], "priority": priority})
            selected_keys.add(key)
            if len(selected) == count:
                return selected
    raise ValueError(f"Could not construct {count} distinct distractors for {target['item_id']}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_terms(workbook: dict[str, list[list[Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    missing_cells: list[dict[str, Any]] = []
    header_problems: list[dict[str, Any]] = []
    for sheet_name in SHEETS:
        rows = workbook.get(sheet_name)
        if not rows:
            raise ValueError(f"Required sheet missing: {sheet_name}")
        headers = [(str(value).strip() if value is not None else "") for value in rows[0][: len(EXPECTED_HEADERS)]]
        if headers != EXPECTED_HEADERS:
            header_problems.append({"sheet": sheet_name, "expected": EXPECTED_HEADERS, "observed": headers})
        prefix = "CYB" if sheet_name == "Cybersecurity" else "GAM"
        record_number = 0
        for index, row in enumerate(rows[1:], 1):
            padded = list(row) + [None] * (len(EXPECTED_HEADERS) - len(row))
            if all(value is None or (isinstance(value, str) and not value.strip()) for value in padded[: len(EXPECTED_HEADERS)]):
                continue
            record_number += 1
            for column_index_, header in enumerate(EXPECTED_HEADERS):
                value = padded[column_index_]
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing_cells.append({"sheet": sheet_name, "source_row": index + 1, "column": header})
            term = str(padded[0] or "").strip()
            construction = str(padded[1] or "").strip()
            community = str(padded[2] or "").strip()
            usages = []
            for usage_number in range(1, 9):
                text = str(padded[3 + (usage_number - 1) * 2] or "").strip()
                url = str(padded[4 + (usage_number - 1) * 2] or "").strip()
                usages.append(
                    {
                        "usage_number": usage_number,
                        "text": text,
                        "source_url": url,
                        "source_sheet": sheet_name,
                        "source_row": index + 1,
                        "usage_column": f"Real Usage {usage_number}",
                        "source_column": f"Source Link {usage_number}",
                        "url_valid": valid_url(url) if url else False,
                    }
                )
            terms.append(
                {
                    "item_id": f"{prefix}-{record_number:04d}",
                    "term": term,
                    "domain": sheet_name,
                    "linguistic_construction": construction,
                    "community": community,
                    "source_sheet": sheet_name,
                    "source_row": index + 1,
                    "usages": usages,
                }
            )
    return terms, {"missing_cells": missing_cells, "header_problems": header_problems}


def audit_dataset(workbook: dict[str, list[list[Any]]], terms: list[dict[str, Any]], parse_audit: dict[str, Any]) -> dict[str, Any]:
    term_occurrences: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    cross_term_occurrences: dict[str, list[str]] = collections.defaultdict(list)
    usage_occurrences: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    url_occurrences: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    malformed_urls: list[dict[str, Any]] = []
    unmatched_pairs: list[dict[str, Any]] = []
    valid_counts: dict[str, int] = {}
    construction_by_domain: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    community_by_domain: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    usage_records = 0
    valid_usage_records = 0

    for term in terms:
        normalized = normalize_text(term["term"])
        term_occurrences[(term["domain"], normalized)].append(term["item_id"])
        cross_term_occurrences[normalized].append(term["item_id"])
        construction_by_domain[term["domain"]][term["linguistic_construction"]] += 1
        community_by_domain[term["domain"]][term["community"]] += 1
        count = 0
        for usage in term["usages"]:
            text_present = bool(usage["text"])
            url_present = bool(usage["source_url"])
            if text_present:
                usage_records += 1
                usage_occurrences[normalize_text(usage["text"])].append(
                    {"item_id": term["item_id"], "usage_number": usage["usage_number"]}
                )
            if url_present:
                url_occurrences[usage["source_url"].strip()].append(
                    {"item_id": term["item_id"], "usage_number": usage["usage_number"]}
                )
                if not usage["url_valid"]:
                    malformed_urls.append(
                        {"item_id": term["item_id"], "usage_number": usage["usage_number"], "source_url": usage["source_url"]}
                    )
            if text_present != url_present:
                unmatched_pairs.append(
                    {
                        "item_id": term["item_id"],
                        "usage_number": usage["usage_number"],
                        "usage_present": text_present,
                        "url_present": url_present,
                    }
                )
            if text_present and url_present and usage["url_valid"]:
                count += 1
                valid_usage_records += 1
        valid_counts[term["item_id"]] = count

    source_audit_rows = workbook.get("Source Audit", [])
    detail_header_index = next(
        (
            index
            for index, row in enumerate(source_audit_rows)
            if [str(value or "").strip() for value in (list(row) + [None] * 5)[:5]]
            == ["Domain", "Source Family", "Term", "Usage", "Source Link"]
        ),
        None,
    )
    detailed_rows = source_audit_rows[detail_header_index + 1 :] if detail_header_index is not None else []
    audit_keys = {
        (str(row[0] or "").strip(), normalize_text(str(row[2] or "")), str(row[3] or "").strip(), str(row[4] or "").strip())
        for row in detailed_rows
        if len(row) >= 5
    }
    source_keys = {
        (term["domain"], normalize_text(term["term"]), f"Usage {usage['usage_number']}", usage["source_url"])
        for term in terms
        for usage in term["usages"]
        if usage["source_url"]
    }

    domain_stats = {}
    for domain in SHEETS:
        selected = [term for term in terms if term["domain"] == domain]
        domain_stats[domain] = {
            "terms": len(selected),
            "usage_examples_present": sum(bool(usage["text"]) for term in selected for usage in term["usages"]),
            "valid_usage_records": sum(valid_counts[term["item_id"]] for term in selected),
            "communities": len({term["community"] for term in selected}),
            "construction_types": len({term["linguistic_construction"] for term in selected}),
        }

    return {
        "dataset_counts": {
            "terms_total": len(terms),
            "terms_by_domain": {domain: sum(term["domain"] == domain for term in terms) for domain in SHEETS},
            "usage_examples_present": usage_records,
            "valid_usage_records": valid_usage_records,
            "maximum_usage_slots": len(terms) * 8,
            "domain_statistics": domain_stats,
        },
        "missing_cells": {
            "count": len(parse_audit["missing_cells"]),
            "by_column": dict(collections.Counter(item["column"] for item in parse_audit["missing_cells"])),
            "locations": parse_audit["missing_cells"],
        },
        "header_problems": parse_audit["header_problems"],
        "duplicate_terms_within_domain": [
            {"domain": domain, "normalized_term": term, "item_ids": ids}
            for (domain, term), ids in sorted(term_occurrences.items())
            if len(ids) > 1
        ],
        "duplicate_terms_across_domains": [
            {"normalized_term": term, "item_ids": ids}
            for term, ids in sorted(cross_term_occurrences.items())
            if len({item.split("-")[0] for item in ids}) > 1
        ],
        "duplicate_usage_examples": [
            {"normalized_usage": text, "occurrences": occurrences}
            for text, occurrences in usage_occurrences.items()
            if text and len(occurrences) > 1
        ],
        "duplicate_urls": [
            {"source_url": url, "occurrences": occurrences}
            for url, occurrences in url_occurrences.items()
            if url and len(occurrences) > 1
        ],
        "malformed_urls": malformed_urls,
        "unmatched_usage_url_pairs": unmatched_pairs,
        "valid_usage_examples_per_term": valid_counts,
        "valid_usage_examples_per_term_distribution": dict(sorted(collections.Counter(valid_counts.values()).items())),
        "linguistic_construction_distribution": {
            domain: dict(counter.most_common()) for domain, counter in construction_by_domain.items()
        },
        "community_distribution": {domain: dict(counter.most_common()) for domain, counter in community_by_domain.items()},
        "source_audit_reconciliation": {
            "detail_header_row": detail_header_index + 1 if detail_header_index is not None else None,
            "detailed_records": len(detailed_rows),
            "source_records": len(source_keys),
            "records_matching": len(audit_keys & source_keys),
            "only_in_source_audit": [list(value) for value in sorted(audit_keys - source_keys)],
            "only_in_domain_sheets": [list(value) for value in sorted(source_keys - audit_keys)],
        },
        "gold_resource_audit": {
            "template_path": "/Users/bobyan/Desktop/community-term-comprehension-experiment/gold_annotations_template.xlsx",
            "canonical_meanings_populated": 0,
            "canonical_meanings_total": 181,
            "reviewer_status": "unreviewed",
            "decision": "No validated term-level semantic gold; definition outputs require human scoring.",
            "methodology_manual_path": "/Users/bobyan/Desktop/Final Report/Community_Term_Evaluation_Metadata_Annotation_Manual_v2.docx",
            "illustrative_gold_meanings": 12,
            "manual_scope": "Illustrative examples only; not treated as a complete adjudicated gold dataset.",
        },
    }


def definition_prompt(term: dict[str, Any], condition: str, contexts: list[dict[str, Any]]) -> str:
    lines = ["Terminology glossary"]
    if condition == "A1_domain":
        lines.append(f"Domain: {term['domain']}")
    elif condition in {"A2_community", "A3_one_context", "A4_three_contexts"}:
        lines.append(f"Community: {term['community']}")
    lines.append(f"Term: {term['term']}")
    if condition == "A3_one_context":
        lines.extend(["Authentic usage:", contexts[0]["text"]])
    elif condition == "A4_three_contexts":
        lines.append("Authentic usages:")
        lines.extend(f"{index}. {context['text']}" for index, context in enumerate(contexts, 1))
    lines.append("Meaning:")
    return "\n".join(lines)


def build_items(terms: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    masking_audit: list[dict[str, Any]] = []
    eligible_by_term: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for term in terms:
        for usage in term["usages"]:
            if not usage["text"] or not usage["source_url"] or not usage["url_valid"]:
                masking_audit.append(
                    {
                        "item_id": term["item_id"],
                        "usage_number": usage["usage_number"],
                        "eligible": False,
                        "reason": "missing_usage_or_valid_url",
                    }
                )
                continue
            masked, rule, count = safe_mask_term(usage["text"], term["term"])
            if masked is None:
                masking_audit.append(
                    {"item_id": term["item_id"], "usage_number": usage["usage_number"], "eligible": False, "reason": rule}
                )
                continue
            record = {**usage, "masked_text": masked, "mask_rule": rule, "masked_occurrences": count}
            eligible_by_term[term["item_id"]].append(record)
            masking_audit.append(
                {
                    "item_id": term["item_id"],
                    "usage_number": usage["usage_number"],
                    "eligible": True,
                    "reason": None,
                    "mask_rule": rule,
                    "masked_occurrences": count,
                }
            )

    masked_items: list[dict[str, Any]] = []
    eligible_flat = [
        (term, usage)
        for term in terms
        for usage in eligible_by_term.get(term["item_id"], [])
    ]
    for global_index, (term, usage) in enumerate(eligible_flat):
        experiment_id = f"masked__{term['item_id'].lower()}__u{usage['usage_number']}"
        distractors = choose_distractors(term, terms, seed)
        rng = random.Random(seed ^ stable_int(experiment_id))
        rng.shuffle(distractors)
        gold_choice = CHOICE_LABELS[(global_index + seed) % len(CHOICE_LABELS)]
        options: dict[str, str] = {}
        distractor_index = 0
        for label in CHOICE_LABELS:
            if label == gold_choice:
                options[label] = term["term"]
            else:
                options[label] = distractors[distractor_index]["term"]
                distractor_index += 1
        prompt = "\n".join(
            [
                "Community terminology completion",
                f"Context: {usage['masked_text']}",
                "Options:",
                *(f"{label}. {options[label]}" for label in CHOICE_LABELS),
                "Correct option (A, B, C, or D):",
            ]
        )
        masked_items.append(
            {
                "experiment": "masked_term_recovery",
                "experiment_id": experiment_id,
                "item_id": term["item_id"],
                "term": term["term"],
                "domain": term["domain"],
                "community": term["community"],
                "linguistic_construction": term["linguistic_construction"],
                "source_sheet": usage["source_sheet"],
                "source_row": usage["source_row"],
                "usage_number": usage["usage_number"],
                "source_url": usage["source_url"],
                "original_usage": usage["text"],
                "masked_usage": usage["masked_text"],
                "mask_rule": usage["mask_rule"],
                "options": options,
                "distractor_provenance": distractors,
                "gold_choice": gold_choice,
                "prompt": prompt,
            }
        )

    definition_items: list[dict[str, Any]] = []
    conditions = ("A0_term_only", "A1_domain", "A2_community", "A3_one_context", "A4_three_contexts")
    for term in terms:
        valid_contexts = [usage for usage in term["usages"] if usage["text"] and usage["source_url"] and usage["url_valid"]]
        for condition in conditions:
            selected = valid_contexts[:1] if condition == "A3_one_context" else valid_contexts[:3] if condition == "A4_three_contexts" else []
            contexts = [
                {
                    "text": usage["text"],
                    "usage_number": usage["usage_number"],
                    "source_url": usage["source_url"],
                    "source_sheet": usage["source_sheet"],
                    "source_row": usage["source_row"],
                }
                for usage in selected
            ]
            definition_items.append(
                {
                    "experiment": "definition_ablation",
                    "experiment_id": f"definition__{term['item_id'].lower()}__{condition.lower()}",
                    "item_id": term["item_id"],
                    "term": term["term"],
                    "domain": term["domain"],
                    "community": term["community"],
                    "linguistic_construction": term["linguistic_construction"],
                    "source_sheet": term["source_sheet"],
                    "source_row": term["source_row"],
                    "condition": condition,
                    "contexts_used": contexts,
                    "prompt": definition_prompt(term, condition, contexts),
                }
            )

    compatibility_items: list[dict[str, Any]] = []
    first_eligible = {item_id: usages[0] for item_id, usages in eligible_by_term.items() if usages}
    by_id = {term["item_id"]: term for term in terms}
    for term in terms:
        positive = first_eligible.get(term["item_id"])
        if not positive:
            continue
        candidates = choose_distractors(term, terms, seed, count=3)
        negative_term = next((by_id[value["item_id"]] for value in candidates if value["item_id"] in first_eligible), None)
        if negative_term is None:
            continue
        negative = first_eligible[negative_term["item_id"]]
        pairs = [("positive", term, positive, "YES"), ("negative", negative_term, negative, "NO")]
        for pair_type, context_term, context, gold_answer in pairs:
            prompt = "\n".join(
                [
                    "Community terminology compatibility",
                    f"Term: {term['term']}",
                    f"Context: {context['masked_text']}",
                    "This context is a valid use of the term (YES or NO):",
                ]
            )
            compatibility_items.append(
                {
                    "experiment": "context_compatibility",
                    "experiment_id": f"compatibility__{term['item_id'].lower()}__{pair_type}",
                    "item_id": term["item_id"],
                    "term": term["term"],
                    "domain": term["domain"],
                    "community": term["community"],
                    "linguistic_construction": term["linguistic_construction"],
                    "pair_type": pair_type,
                    "context_source_item_id": context_term["item_id"],
                    "context_source_term": context_term["term"],
                    "source_sheet": context["source_sheet"],
                    "source_row": context["source_row"],
                    "usage_number": context["usage_number"],
                    "source_url": context["source_url"],
                    "masked_usage": context["masked_text"],
                    "gold_answer": gold_answer,
                    "prompt": prompt,
                }
            )
    return masked_items, definition_items, compatibility_items, masking_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/terminology_eval.yaml")
    args = parser.parse_args()
    root = project_root()
    config, config_path = load_config(resolve_project_path(root, args.config))
    workbook_path = resolve_project_path(root, config["paths"]["workbook"])
    output_dir = resolve_project_path(root, config["paths"]["processed_dir"])

    workbook = read_xlsx(workbook_path)
    terms, parse_audit = parse_terms(workbook)
    audit = audit_dataset(workbook, terms, parse_audit)
    masked, definitions, compatibility, masking_audit = build_items(terms, int(config["seed"]))
    audit["masking"] = {
        "eligible_contexts": sum(item["eligible"] for item in masking_audit),
        "ineligible_contexts": sum(not item["eligible"] for item in masking_audit),
        "ineligible_by_reason": dict(
            collections.Counter(item["reason"] for item in masking_audit if not item["eligible"])
        ),
        "records": masking_audit,
    }
    audit["built_items"] = {
        "masked_term_recovery": len(masked),
        "definition_ablation": len(definitions),
        "context_compatibility": len(compatibility),
    }

    _write_jsonl(output_dir / "terms.jsonl", terms)
    _write_jsonl(output_dir / "masked_recovery.jsonl", masked)
    _write_jsonl(output_dir / "definition_ablation.jsonl", definitions)
    _write_jsonl(output_dir / "context_compatibility.jsonl", compatibility)
    _write_json(output_dir / "dataset_audit.json", audit)
    _write_json(
        output_dir / "prompt_templates.json",
        {
            "definition_conditions": {condition: next(item["prompt"] for item in definitions if item["condition"] == condition) for condition in config["experiment"]["definition_conditions"]},
            "masked_term_recovery": masked[0]["prompt"] if masked else None,
            "context_compatibility": compatibility[0]["prompt"] if compatibility else None,
            "note": "Examples are instantiated for audit; source URLs never appear in prompts.",
        },
    )
    masking_summary = {key: value for key, value in audit["masking"].items() if key != "records"}
    print(
        json.dumps(
            {
                "config": str(config_path),
                "workbook": str(workbook_path),
                **audit["dataset_counts"],
                **audit["built_items"],
                **masking_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
