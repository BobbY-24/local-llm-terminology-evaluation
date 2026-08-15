# Local LLM Community-Terminology Evaluation

## Research Questions

The experiment measures prior lexical knowledge, domain/community grounding, authentic-context effects, masked lexical recovery, domain differences, construction differences, and observable failure mechanisms for community terminology.

## Dataset

| Domain | Terms | Usage examples | Communities | Construction types |
|---|---|---|---|---|
| Cybersecurity | 74 | 592 | 71 | 11 |
| Gaming | 107 | 856 | 107 | 14 |
| Total | 181 | 1448 | 178 | 15 |

The workbook was read without modifying the original. Its Source Audit contains 1448 detailed records. Exact audit details, including missing cells, duplicates, malformed URLs, valid-context counts per term, and distributions, are in `data/processed/terminology_eval/dataset_audit.json`.

Validation found 0 missing term/metadata/usage/source cells, 0 malformed URLs, 0 duplicate terms, 1 duplicated usage text, and 18 duplicated URLs. Every term has eight valid usage/source pairs. Conservative surface masking made 1,335 contexts eligible for masked recovery; 113 were retained in the audit but excluded from that task (108 lacked the canonical surface form and 5 left a detectable residual target after masking). The detailed Source Audit reconciles exactly with the 1,448 sheet-level usage/source records.

No complete validated gold-definition resource was found. The external gold workbook has 181 blank canonical-meaning fields marked unreviewed. A separate methodology manual contains 12 illustrative meanings but is not treated as complete adjudicated gold.

## Model

- Checkpoint: `mlx-community/Qwen2.5-3B-4bit` at `76ae31041917ee0ef78284988aca339694acb7e9`
- Stage: base / pretrained base model
- Parameters: 3.09B
- Quantization: 4-bit, group size 64
- Backend: MLX-LM, local model path `models/Qwen2.5-3B-4bit`
- Decoding: greedy (`temperature=0`, `top_p=1`, `top_k=0`, `seed=42`), batch size 1, 512-token KV cap
- Chat template: not applied

## Experimental Design

Definition generation uses five completion-style ablations: term only, domain, community, one authentic context, and three authentic contexts. Masked recovery uses every safely maskable authentic usage and four deterministic hard-negative options. Context compatibility uses one positive and one hard-negative masked context per eligible term.

## Tasks

- Definition/context ablation: 905 raw outputs; semantic scoring deferred to human annotation.
- Masked-term recovery: 1335 automatically scored items; 4-way chance baseline = 25%.
- Context compatibility: 362 automatically scored YES/NO items; chance baseline = 50%.

## Prompt Templates

Prompt text is fixed before evaluation. URLs are retained in metadata but never placed in prompts. Instantiated examples are stored in `data/processed/terminology_eval/prompt_templates.json`.

## Evaluation Metrics

Automatic tasks report micro accuracy, macro-per-term accuracy, format-error rate, and 95% cluster-bootstrap confidence intervals resampling terms. Unparseable outputs count as incorrect. The Gaming–Cybersecurity difference is -16.9% with a term-bootstrap 95% CI of [-23.5%, -10.5%].

## Results

### Overall

| Domain | N items | Micro accuracy | Macro term accuracy | CI low | CI high | Format errors |
|---|---|---|---|---|---|---|
| Overall | 1335 | 49.4% | 49.5% | 46.0% | 52.8% | 16 |
| Gaming | 774 | 42.2% | 43.0% | 38.2% | 46.6% | 5 |
| Cybersecurity | 561 | 59.2% | 59.1% | 54.4% | 63.9% | 11 |

### Gaming vs Cybersecurity

Observed domain differences are descriptive. The interval above preserves within-term clustering and does not establish a causal domain effect.

### Linguistic Construction

| Construction | Terms | Items | Accuracy | CI low | CI high |
|---|---|---|---|---|---|
| Abbreviation | 4 | 32 | 43.8% | 37.5% | 56.2% |
| Acronym | 6 | 46 | 71.7% | 61.9% | 79.2% |
| Affixation | 12 | 81 | 59.3% | 45.7% | 73.3% |
| Blending | 11 | 73 | 42.5% | 31.7% | 52.9% |
| Borrowing | 3 | 22 | 31.8% | 16.7% | 37.5% |
| Clipping | 4 | 29 | 34.5% | 20.7% | 46.2% |
| Code Word | 1 | 8 | 75.0% | 75.0% | 75.0% |
| Community-specific Jargon | 1 | 8 | 62.5% | 62.5% | 62.5% |
| Compound | 29 | 206 | 48.5% | 39.2% | 58.5% |
| Functional Shift | 3 | 24 | 33.3% | 25.0% | 50.0% |
| Initialism | 19 | 146 | 41.8% | 33.3% | 50.3% |
| Meme Expression | 1 | 8 | 25.0% | 25.0% | 25.0% |
| Metaphor | 74 | 553 | 53.3% | 48.0% | 58.8% |
| Multiword Expression | 8 | 60 | 41.7% | 33.3% | 49.2% |
| Semantic Shift | 5 | 39 | 35.9% | 15.4% | 69.2% |

Categories with fewer than 5 terms should not be interpreted substantively.

### Context Effects

Definition outputs were generated, but no semantic scores were automatically assigned because no validated complete gold meanings exist. Figure 4 is therefore intentionally omitted.

### Context Compatibility

| Group | Items | Accuracy | Format errors |
|---|---|---|---|
| Overall | 362 | 53.6% | 0 |
| Positive | 181 | 62.4% | 0 |
| Negative | 181 | 44.8% | 0 |

## Qualitative Failure Analysis

The following examples were manually reviewed against the workbook's authentic usage evidence. These labels apply only to this small illustrative set; the 905-row human-scoring sheet remains blank rather than receiving unreliable automatic semantic labels.

| Term | Domain | Community | Condition | Model output | Failure type |
|---|---|---|---|---|---|
| credential stuffing | Cybersecurity | Account Takeover | A0 term only | Uses stolen username/password lists to try multiple accounts | `CORRECT` |
| stat stick | Gaming | Warframe | A1 domain | “A stick ... used to measure the height of players” | `COMPOSITIONAL_LITERALIZATION` |
| stat stick | Gaming | Warframe | A4 three contexts | Weapon/mod neighborhood, but says it is used “to mod other weapons” | `PARTIAL_CORE_MEANING` |
| smurfing | Gaming | Matchmaking Abuse | A0 term only | ICMP denial-of-service attack | `WRONG_COMMUNITY` |
| smurfing | Gaming | Matchmaking Abuse | A4 three contexts | New account used to play below the player's skill level | `CORRECT` (context rescue) |
| weaving | Gaming | FFXIV | A0 term only | Interlacing yarn to form fabric | `COMPOSITIONAL_LITERALIZATION` |
| weaving | Gaming | FFXIV | A4 three contexts | Using off-GCD abilities between global-cooldown abilities | `CORRECT` (context rescue) |
| pepper | Cybersecurity | Password Hashing | A0 term only | “a type of spice” | `COMPOSITIONAL_LITERALIZATION` |
| pepper | Cybersecurity | Password Hashing | A2 community | Shared random value added before password hashing | `CORRECT` (community rescue) |
| KWTD | Gaming | Destiny 2 Raid LFG | A4 three contexts | “Kill Wipe Team Deathmatch” | `HALLUCINATED_MECHANISM` |
| blueberries | Gaming | Destiny 2 | A4 three contexts | Rewards from clearing a raid | `WRONG_SENSE` |
| N-day | Cybersecurity | Vulnerability Exploitation | A4 three contexts | Vulnerability discovered and patched in one calendar day | `WRONG_SENSE` |
| kiting | Gaming | Enemy Positioning | masked recovery | “A, B, C, D” | `INSTRUCTION_FORMAT_FAILURE` |

Observed pattern: authentic context can rescue an opaque community sense, but rescue is not guaranteed. The model sometimes moves into the correct semantic neighborhood while hallucinating the mechanism (`stat stick`), and sometimes remains committed to an ordinary or invented expansion despite several contexts (`KWTD`, `N-day`). These are observations from a curated set, not prevalence estimates. The full definition output set should be scored with `results/terminology_eval/scored/terminology_eval_qwen25_3b_base_seed42_v1_definition_human_scoring.csv` before estimating context gains.

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
