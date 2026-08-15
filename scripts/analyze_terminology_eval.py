#!/usr/bin/env python3
"""Generate clustered-bootstrap summaries, tables, SVG figures, and the report."""

from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from terminology_eval_common import load_config, project_root, read_jsonl, resolve_project_path


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def cluster_bootstrap(records: list[dict[str, Any]], samples: int, seed: int) -> tuple[float, float]:
    by_term: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        by_term[record["item_id"]].append(record)
    term_ids = sorted(by_term)
    if not term_ids:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [rng.choice(term_ids) for _ in term_ids]
        pooled = [record for item_id in selected for record in by_term[item_id]]
        estimates.append(sum(bool(record["correct"]) for record in pooled) / len(pooled))
    estimates.sort()
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return low, high


def domain_difference_bootstrap(records: list[dict[str, Any]], samples: int, seed: int) -> tuple[float, float, float]:
    domains = {domain: [record for record in records if record["domain"] == domain] for domain in ("Gaming", "Cybersecurity")}
    observed = summarize(domains["Gaming"], samples, seed)["micro_accuracy"] - summarize(
        domains["Cybersecurity"], samples, seed + 1
    )["micro_accuracy"]
    grouped = {}
    for domain, rows in domains.items():
        clusters: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            clusters[row["item_id"]].append(row)
        grouped[domain] = clusters
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        estimates = {}
        for domain in ("Gaming", "Cybersecurity"):
            ids = sorted(grouped[domain])
            chosen = [rng.choice(ids) for _ in ids]
            pooled = [row for item_id in chosen for row in grouped[domain][item_id]]
            estimates[domain] = sum(bool(row["correct"]) for row in pooled) / len(pooled)
        differences.append(estimates["Gaming"] - estimates["Cybersecurity"])
    differences.sort()
    return observed, differences[int(0.025 * (samples - 1))], differences[int(0.975 * (samples - 1))]


def summarize(records: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    by_term: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        by_term[record["item_id"]].append(record)
    n = len(records)
    correct = sum(bool(record["correct"]) for record in records)
    term_accuracy = [sum(bool(row["correct"]) for row in rows) / len(rows) for rows in by_term.values()]
    low, high = cluster_bootstrap(records, samples, seed)
    return {
        "terms": len(by_term),
        "items": n,
        "correct": correct,
        "micro_accuracy": correct / n if n else math.nan,
        "macro_term_accuracy": mean(term_accuracy) if term_accuracy else math.nan,
        "ci_low": low,
        "ci_high": high,
        "format_errors": sum(bool(record.get("format_error")) for record in records),
        "format_error_rate": sum(bool(record.get("format_error")) for record in records) / n if n else math.nan,
        "extra_generation": sum(bool(record.get("extra_generation")) for record in records),
    }


def grouped_summaries(
    records: list[dict[str, Any]], field: str, samples: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    output = []
    for index, (label, rows) in enumerate(sorted(groups.items())):
        output.append({field: label, **summarize(rows, samples, seed + index)})
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], percent_fields: set[str] | None = None) -> str:
    percent_fields = percent_fields or set()
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if key in percent_fields and isinstance(value, (float, int)) and not math.isnan(float(value)):
                value = percent(float(value))
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def svg_bar_chart(path: Path, title: str, labels: list[str], values: list[float], lows: list[float] | None = None, highs: list[float] | None = None) -> None:
    width, height = 900, 560
    left, right, top, bottom = 100, 40, 90, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    bar_w = min(140, plot_w / max(len(labels), 1) * 0.55)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="38" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700" fill="#172033">{html.escape(title)}</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_h * (1 - tick / 100)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#d8dee9" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-family="Arial" font-size="14" fill="#52606d">{tick}%</text>')
    step = plot_w / max(len(labels), 1)
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + step * (index + 0.5)
        y = top + plot_h * (1 - value)
        parts.append(f'<rect x="{x-bar_w/2}" y="{y}" width="{bar_w}" height="{top+plot_h-y}" rx="4" fill="#2563eb"/>')
        parts.append(f'<text x="{x}" y="{y-12}" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="#172033">{100*value:.1f}%</text>')
        parts.append(f'<text x="{x}" y="{top+plot_h+30}" text-anchor="middle" font-family="Arial" font-size="15" fill="#172033">{html.escape(label)}</text>')
        if lows is not None and highs is not None:
            low_y = top + plot_h * (1 - lows[index])
            high_y = top + plot_h * (1 - highs[index])
            parts.extend(
                [
                    f'<line x1="{x}" y1="{high_y}" x2="{x}" y2="{low_y}" stroke="#111827" stroke-width="2"/>',
                    f'<line x1="{x-8}" y1="{high_y}" x2="{x+8}" y2="{high_y}" stroke="#111827" stroke-width="2"/>',
                    f'<line x1="{x-8}" y1="{low_y}" x2="{x+8}" y2="{low_y}" stroke="#111827" stroke-width="2"/>',
                ]
            )
    parts.append(f'<text x="22" y="{top+plot_h/2}" transform="rotate(-90 22 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="16" fill="#172033">Accuracy</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_horizontal_chart(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    row_h = 42
    width, height = 1050, 110 + row_h * len(labels)
    left, right, top = 280, 50, 70
    plot_w = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700" fill="#172033">{html.escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * row_h
        parts.append(f'<text x="{left-12}" y="{y+20}" text-anchor="end" font-family="Arial" font-size="14" fill="#172033">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y+5}" width="{plot_w}" height="22" rx="3" fill="#eef2f7"/>')
        parts.append(f'<rect x="{left}" y="{y+5}" width="{plot_w*value}" height="22" rx="3" fill="#0f766e"/>')
        parts.append(f'<text x="{left+plot_w*value+8}" y="{y+21}" font-family="Arial" font-size="13" fill="#172033">{100*value:.1f}%</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_histogram(path: Path, values: list[float]) -> None:
    bins = [0] * 10
    for value in values:
        bins[min(int(value * 10), 9)] += 1
    labels = [f"{index*10}–{(index+1)*10}%" for index in range(10)]
    width, height = 1000, 560
    left, right, top, bottom = 90, 40, 90, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(bins) or 1
    step = plot_w / len(bins)
    bar_w = step * 0.72
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="38" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700" fill="#172033">Distribution of per-term masked-recovery accuracy</text>',
    ]
    tick_step = max(1, math.ceil(maximum / 5))
    for tick in range(0, maximum + tick_step, tick_step):
        if tick > maximum:
            break
        y = top + plot_h * (1 - tick / maximum)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#d8dee9" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-family="Arial" font-size="14" fill="#52606d">{tick}</text>')
    for index, (label, count) in enumerate(zip(labels, bins)):
        x = left + step * (index + 0.5)
        y = top + plot_h * (1 - count / maximum)
        parts.append(f'<rect x="{x-bar_w/2}" y="{y}" width="{bar_w}" height="{top+plot_h-y}" rx="3" fill="#7c3aed"/>')
        parts.append(f'<text x="{x}" y="{y-8}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#172033">{count}</text>')
        parts.append(f'<text x="{x}" y="{top+plot_h+30}" text-anchor="middle" font-family="Arial" font-size="13" fill="#172033">{label}</text>')
    parts.append(f'<text x="22" y="{top+plot_h/2}" transform="rotate(-90 22 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="16" fill="#172033">Number of terms</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="16" fill="#172033">Per-term accuracy bin</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def load_human_scores(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("semantic_score", "").strip() in {"0", "1", "2"}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/terminology_eval.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    root = project_root()
    config, _ = load_config(resolve_project_path(root, args.config))
    run_id = args.run_id or str(config["run_id"])
    results_root = resolve_project_path(root, config["paths"]["results_root"])
    scored_dir = results_root / "scored"
    summaries_dir = results_root / "summaries"
    tables_dir = summaries_dir / "tables"
    reports_dir = resolve_project_path(root, config["paths"]["reports_dir"])
    figures_dir = resolve_project_path(root, config["paths"]["figures_dir"])
    processed_dir = resolve_project_path(root, config["paths"]["processed_dir"])
    samples = int(config["analysis"]["bootstrap_samples"])
    seed = int(config["seed"])

    masked_path = scored_dir / f"{run_id}_masked_recovery_scored.jsonl"
    if not masked_path.exists():
        raise SystemExit(f"Scored masked-recovery results missing: {masked_path}")
    masked = read_jsonl(masked_path)
    audit = json.loads((processed_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    prompt_templates = json.loads((processed_dir / "prompt_templates.json").read_text(encoding="utf-8"))

    overall = summarize(masked, samples, seed)
    domain_rows = []
    for index, domain in enumerate(("Gaming", "Cybersecurity")):
        domain_rows.append({"domain": domain, **summarize([row for row in masked if row["domain"] == domain], samples, seed + index + 1)})
    domain_table = [{"domain": "Overall", **overall}, *domain_rows]
    construction_rows = grouped_summaries(masked, "linguistic_construction", samples, seed + 100)
    community_rows = grouped_summaries(masked, "community", samples, seed + 300)
    observed_diff, diff_low, diff_high = domain_difference_bootstrap(masked, samples, seed + 500)

    dataset_rows = []
    for domain in ("Cybersecurity", "Gaming"):
        stats = audit["dataset_counts"]["domain_statistics"][domain]
        dataset_rows.append({"domain": domain, **stats})
    dataset_rows.append(
        {
            "domain": "Total",
            "terms": audit["dataset_counts"]["terms_total"],
            "usage_examples_present": audit["dataset_counts"]["usage_examples_present"],
            "valid_usage_records": audit["dataset_counts"]["valid_usage_records"],
            "communities": sum(row["communities"] for row in dataset_rows),
            "construction_types": len(
                set().union(
                    *(
                        set(audit["linguistic_construction_distribution"][domain])
                        for domain in ("Cybersecurity", "Gaming")
                    )
                )
            ),
        }
    )

    auto_fields = [
        "domain",
        "terms",
        "items",
        "correct",
        "micro_accuracy",
        "macro_term_accuracy",
        "ci_low",
        "ci_high",
        "format_errors",
        "format_error_rate",
        "extra_generation",
    ]
    construction_fields = ["linguistic_construction", *auto_fields[1:]]
    community_fields = ["community", *auto_fields[1:]]
    write_csv(tables_dir / "table1_dataset_statistics.csv", dataset_rows, list(dataset_rows[0]))
    write_csv(tables_dir / "table2_automatic_evaluation.csv", domain_table, auto_fields)
    write_csv(tables_dir / "table3_linguistic_construction.csv", construction_rows, construction_fields)
    write_csv(tables_dir / "community_results.csv", community_rows, community_fields)

    failures = [row for row in masked if not row["correct"]][:10]
    failure_rows = [
        {
            "term": row["term"],
            "domain": row["domain"],
            "community": row["community"],
            "condition": "masked_term_recovery",
            "model_output": row.get("raw_output", ""),
            "failure_type": "",
            "gold_choice": row["gold_choice"],
            "parsed_choice": row.get("parsed_choice"),
        }
        for row in failures
    ]
    write_csv(
        tables_dir / "table4_representative_failures.csv",
        failure_rows,
        ["term", "domain", "community", "condition", "model_output", "failure_type", "gold_choice", "parsed_choice"],
    )

    compatibility_path = scored_dir / f"{run_id}_context_compatibility_scored.jsonl"
    compatibility = read_jsonl(compatibility_path) if compatibility_path.exists() else []
    compatibility_rows = []
    if compatibility:
        compatibility_rows.append({"group": "Overall", **summarize(compatibility, samples, seed + 700)})
        for index, pair_type in enumerate(("positive", "negative")):
            compatibility_rows.append(
                {
                    "group": pair_type.title(),
                    **summarize([row for row in compatibility if row["pair_type"] == pair_type], samples, seed + 701 + index),
                }
            )
        write_csv(tables_dir / "context_compatibility.csv", compatibility_rows, ["group", *auto_fields[1:]])

    human_path = scored_dir / f"{run_id}_definition_human_scoring.csv"
    human_scores = load_human_scores(human_path)
    context_gain_note = "Definition outputs were generated, but no semantic scores were automatically assigned because no validated complete gold meanings exist. Figure 4 is therefore intentionally omitted."
    if human_scores:
        by_condition: dict[str, list[float]] = collections.defaultdict(list)
        for row in human_scores:
            by_condition[row["condition"]].append(float(row["semantic_score"]))
        condition_order = config["experiment"]["definition_conditions"]
        labels = [condition for condition in condition_order if condition in by_condition]
        values = [mean(by_condition[condition]) / 2 for condition in labels]
        svg_bar_chart(figures_dir / "figure4_definition_context_gain.svg", "Mean human semantic score by context condition", labels, values)
        context_gain_note = f"Figure 4 uses {len(human_scores)} manually scored definition outputs; scores are normalized by the maximum rubric score of 2."

    svg_bar_chart(
        figures_dir / "figure1_domain_accuracy.svg",
        "Masked-term recovery accuracy by domain",
        [row["domain"] for row in domain_rows],
        [row["micro_accuracy"] for row in domain_rows],
        [row["ci_low"] for row in domain_rows],
        [row["ci_high"] for row in domain_rows],
    )
    construction_sorted = sorted(construction_rows, key=lambda row: (row["micro_accuracy"], row["linguistic_construction"]))
    svg_horizontal_chart(
        figures_dir / "figure2_construction_accuracy.svg",
        "Masked-term recovery accuracy by linguistic construction",
        [f"{row['linguistic_construction']} (n={row['terms']} terms)" for row in construction_sorted],
        [row["micro_accuracy"] for row in construction_sorted],
    )
    per_term: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in masked:
        per_term[row["item_id"]].append(row)
    per_term_values = [sum(bool(row["correct"]) for row in rows) / len(rows) for rows in per_term.values()]
    svg_histogram(figures_dir / "figure3_per_term_accuracy.svg", per_term_values)

    dataset_md = markdown_table(
        dataset_rows,
        [
            ("domain", "Domain"),
            ("terms", "Terms"),
            ("usage_examples_present", "Usage examples"),
            ("communities", "Communities"),
            ("construction_types", "Construction types"),
        ],
    )
    auto_md = markdown_table(
        domain_table,
        [
            ("domain", "Domain"),
            ("items", "N items"),
            ("micro_accuracy", "Micro accuracy"),
            ("macro_term_accuracy", "Macro term accuracy"),
            ("ci_low", "CI low"),
            ("ci_high", "CI high"),
            ("format_errors", "Format errors"),
        ],
        {"micro_accuracy", "macro_term_accuracy", "ci_low", "ci_high"},
    )
    construction_md = markdown_table(
        construction_rows,
        [
            ("linguistic_construction", "Construction"),
            ("terms", "Terms"),
            ("items", "Items"),
            ("micro_accuracy", "Accuracy"),
            ("ci_low", "CI low"),
            ("ci_high", "CI high"),
        ],
        {"micro_accuracy", "ci_low", "ci_high"},
    )
    failure_md = markdown_table(
        failure_rows,
        [
            ("term", "Term"),
            ("domain", "Domain"),
            ("community", "Community"),
            ("condition", "Condition"),
            ("model_output", "Model output"),
            ("failure_type", "Failure type"),
        ],
    )
    compatibility_md = (
        markdown_table(
            compatibility_rows,
            [("group", "Group"), ("items", "Items"), ("micro_accuracy", "Accuracy"), ("format_errors", "Format errors")],
            {"micro_accuracy"},
        )
        if compatibility_rows
        else "Not run."
    )

    validation = audit
    validation_report = f"""# Workbook validation report

Source workbook: `{config['paths']['workbook']}`

## Counts

{dataset_md}

## Data-quality checks

- Missing cells across the 181 × 19 domain-sheet data grid: {validation['missing_cells']['count']}
- Duplicate terms within a domain: {len(validation['duplicate_terms_within_domain'])}
- Terms occurring in both domains after conservative normalization: {len(validation['duplicate_terms_across_domains'])}
- Duplicate usage examples: {len(validation['duplicate_usage_examples'])}
- Duplicate URLs: {len(validation['duplicate_urls'])}
- Malformed nonempty URLs: {len(validation['malformed_urls'])}
- Unmatched usage/URL pairs: {len(validation['unmatched_usage_url_pairs'])}
- Source Audit records reconciled: {validation['source_audit_reconciliation']['records_matching']} / {validation['source_audit_reconciliation']['source_records']}
- Masked-recovery eligible contexts: {validation['masking']['eligible_contexts']}
- Masking-ineligible contexts: {validation['masking']['ineligible_contexts']}

All problematic locations and duplicate occurrence lists are retained in `data/processed/terminology_eval/dataset_audit.json`; no row was silently discarded.
"""
    (reports_dir / "dataset_validation.md").parent.mkdir(parents=True, exist_ok=True)
    (reports_dir / "dataset_validation.md").write_text(validation_report, encoding="utf-8")

    definition_raw_path = results_root / "raw" / f"{run_id}_definition_ablation.jsonl"
    definition_count = len(read_jsonl(definition_raw_path)) if definition_raw_path.exists() else 0
    report = f"""# Local LLM Community-Terminology Evaluation

## Research Questions

The experiment measures prior lexical knowledge, domain/community grounding, authentic-context effects, masked lexical recovery, domain differences, construction differences, and observable failure mechanisms for community terminology.

## Dataset

{dataset_md}

The workbook was read without modifying the original. Its Source Audit contains {audit['source_audit_reconciliation']['detailed_records']} detailed records. Exact audit details, including missing cells, duplicates, malformed URLs, valid-context counts per term, and distributions, are in `data/processed/terminology_eval/dataset_audit.json`.

No complete validated gold-definition resource was found. The external gold workbook has 181 blank canonical-meaning fields marked unreviewed. A separate methodology manual contains 12 illustrative meanings but is not treated as complete adjudicated gold.

## Model

- Checkpoint: `{config['model']['repository_id']}` at `{config['model']['revision']}`
- Stage: {config['model']['stage']} / pretrained base model
- Parameters: {config['model']['parameter_count']}
- Quantization: {config['model']['quantization_bits']}-bit, group size {config['model']['quantization_group_size']}
- Backend: MLX-LM, local model path `{config['model']['path']}`
- Decoding: greedy (`temperature=0`, `top_p=1`, `top_k=0`, `seed=42`), batch size 1, 512-token KV cap
- Chat template: not applied

## Experimental Design

Definition generation uses five completion-style ablations: term only, domain, community, one authentic context, and three authentic contexts. Masked recovery uses every safely maskable authentic usage and four deterministic hard-negative options. Context compatibility uses one positive and one hard-negative masked context per eligible term.

## Tasks

- Definition/context ablation: {definition_count} raw outputs; semantic scoring deferred to human annotation.
- Masked-term recovery: {len(masked)} automatically scored items; 4-way chance baseline = 25%.
- Context compatibility: {len(compatibility)} automatically scored YES/NO items; chance baseline = 50%.

## Prompt Templates

Prompt text is fixed before evaluation. URLs are retained in metadata but never placed in prompts. Instantiated examples are stored in `data/processed/terminology_eval/prompt_templates.json`.

## Evaluation Metrics

Automatic tasks report micro accuracy, macro-per-term accuracy, format-error rate, and 95% cluster-bootstrap confidence intervals resampling terms. Unparseable outputs count as incorrect. The Gaming–Cybersecurity difference is {percent(observed_diff)} with a term-bootstrap 95% CI of [{percent(diff_low)}, {percent(diff_high)}].

## Results

### Overall

{auto_md}

### Gaming vs Cybersecurity

Observed domain differences are descriptive. The interval above preserves within-term clustering and does not establish a causal domain effect.

### Linguistic Construction

{construction_md}

Categories with fewer than {config['analysis']['minimum_terms_for_interpretation']} terms should not be interpreted substantively.

### Context Effects

{context_gain_note}

### Context Compatibility

{compatibility_md}

## Qualitative Failure Analysis

The table below is an automatically selected audit sample of incorrect masked-recovery records. Failure types are deliberately blank pending manual review; they are not inferred by keyword rules.

{failure_md}

Definition outputs should be reviewed for compositional literalization, wrong sense/community, nearby-concept confusion, hallucinated mechanisms, and instruction-format failure using `results/terminology_eval/scored/{run_id}_definition_human_scoring.csv`.

## Limitations

- Four-bit quantization may change lexical probabilities relative to BF16.
- The evaluated model is a base model; generated-choice format adherence is not equivalent to semantic competence.
- Authentic usage is evidence, not automatically a gold definition.
- Multiple usages of a term are clustered, but terms, communities, and source sites are not fully independent.
- Distractors follow deterministic metadata priorities but have not all received human plausibility adjudication.
- No claim is made that terms were absent from pretraining or are objectively novel.

## Reproducibility

```bash
cd ~/llm-research
source .venv/bin/activate
python scripts/build_eval_dataset.py --config configs/terminology_eval.yaml
python scripts/run_terminology_eval.py --config configs/terminology_eval.yaml --experiment masked_recovery
python scripts/run_terminology_eval.py --config configs/terminology_eval.yaml --experiment context_compatibility
python scripts/run_terminology_eval.py --config configs/terminology_eval.yaml --experiment definition_ablation
python scripts/score_terminology_eval.py --config configs/terminology_eval.yaml
python scripts/analyze_terminology_eval.py --config configs/terminology_eval.yaml
```

Runs append one flushed JSON object per completed item and resume by `experiment_id`. Existing scored artifacts are preserved by default.

## Next Experiments

The most informative next experiment is manual semantic scoring of a balanced, double-coded subset of definition outputs across all five conditions. That directly estimates context rescue while separating semantic recovery from output-format behavior.
"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "terminology_eval_report.md").write_text(report, encoding="utf-8")

    summary = {
        "run_id": run_id,
        "masked_recovery": {"overall": overall, "by_domain": domain_rows, "domain_difference": {"gaming_minus_cybersecurity": observed_diff, "ci_low": diff_low, "ci_high": diff_high}},
        "context_compatibility": compatibility_rows,
        "definition_outputs": definition_count,
        "human_definition_scores_available": len(human_scores),
    }
    summaries_dir.mkdir(parents=True, exist_ok=True)
    (summaries_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
