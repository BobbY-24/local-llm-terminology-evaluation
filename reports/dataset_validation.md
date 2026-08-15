# Workbook validation report

Source workbook: `data/raw/refined_terminology_usage_example_rich.xlsx`

## Counts

| Domain | Terms | Usage examples | Communities | Construction types |
|---|---|---|---|---|
| Cybersecurity | 74 | 592 | 71 | 11 |
| Gaming | 107 | 856 | 107 | 14 |
| Total | 181 | 1448 | 178 | 15 |

## Data-quality checks

- Missing cells across the 181 × 19 domain-sheet data grid: 0
- Duplicate terms within a domain: 0
- Terms occurring in both domains after conservative normalization: 0
- Duplicate usage examples: 1
- Duplicate URLs: 18
- Malformed nonempty URLs: 0
- Unmatched usage/URL pairs: 0
- Source Audit records reconciled: 1448 / 1448
- Masked-recovery eligible contexts: 1335
- Masking-ineligible contexts: 113

All problematic locations and duplicate occurrence lists are retained in `data/processed/terminology_eval/dataset_audit.json`; no row was silently discarded.
