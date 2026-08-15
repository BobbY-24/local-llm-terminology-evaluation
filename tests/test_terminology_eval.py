from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_eval_dataset import (  # noqa: E402
    audit_dataset,
    build_items,
    choose_distractors,
    parse_terms,
    read_xlsx,
    safe_mask_term,
)
from terminology_eval_common import load_completed_ids, parse_choice  # noqa: E402


class MaskingTests(unittest.TestCase):
    def test_exact_masking(self) -> None:
        masked, rule, count = safe_mask_term("I need a Stat Stick for this build.", "stat stick")
        self.assertEqual(masked, "I need a [TERM] for this build.")
        self.assertEqual(rule, "exact")
        self.assertEqual(count, 1)

    def test_hyphen_space_variant(self) -> None:
        masked, rule, _ = safe_mask_term("This is a pass-the-hash attempt.", "pass the hash")
        self.assertEqual(masked, "This is a [TERM] attempt.")
        self.assertEqual(rule, "hyphen_space_variant")

    def test_conservative_plural(self) -> None:
        masked, rule, _ = safe_mask_term("Two honeypots were deployed.", "honeypot")
        self.assertEqual(masked, "Two [TERM] were deployed.")
        self.assertEqual(rule, "simple_plural_s")

    def test_absent_term_is_ineligible(self) -> None:
        masked, reason, count = safe_mask_term("No lexical occurrence here.", "smurf")
        self.assertIsNone(masked)
        self.assertEqual(reason, "canonical_term_absent")
        self.assertEqual(count, 0)


class InfrastructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workbook_path = ROOT / "data" / "raw" / "refined_terminology_usage_example_rich.xlsx"
        cls.workbook = read_xlsx(cls.workbook_path)
        cls.terms, cls.parse_audit = parse_terms(cls.workbook)

    def test_workbook_parsing(self) -> None:
        counts = Counter(term["domain"] for term in self.terms)
        self.assertEqual(counts, {"Cybersecurity": 74, "Gaming": 107})
        self.assertEqual(len(self.terms), 181)
        self.assertTrue(all(len(term["usages"]) == 8 for term in self.terms))

    def test_distractors_are_deterministic_and_distinct(self) -> None:
        target = self.terms[0]
        first = choose_distractors(target, self.terms, 42)
        second = choose_distractors(target, self.terms, 42)
        self.assertEqual(first, second)
        selected = [value["term"].casefold() for value in first]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertNotIn(target["term"].casefold(), selected)

    def test_dataset_ids_seed_and_option_balance(self) -> None:
        first = build_items(self.terms, 42)
        second = build_items(self.terms, 42)
        masked_first, definitions_first, compatibility_first, _ = first
        masked_second = second[0]
        self.assertEqual(masked_first, masked_second)
        all_ids = [item["experiment_id"] for group in first[:3] for item in group]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        positions = Counter(item["gold_choice"] for item in masked_first)
        self.assertLessEqual(max(positions.values()) - min(positions.values()), 1)
        self.assertEqual(len(definitions_first), 181 * 5)
        self.assertLessEqual(len(compatibility_first), 181 * 2)

    def test_no_context_answer_leakage(self) -> None:
        masked, _, _, _ = build_items(self.terms, 42)
        for item in masked:
            remasked, _, _ = safe_mask_term(item["masked_usage"], item["term"])
            self.assertIsNone(remasked, item["experiment_id"])
            self.assertIn("[TERM]", item["masked_usage"])

    def test_source_audit_reconciles(self) -> None:
        audit = audit_dataset(self.workbook, self.terms, self.parse_audit)
        reconciliation = audit["source_audit_reconciliation"]
        self.assertEqual(reconciliation["detailed_records"], 1448)
        self.assertEqual(reconciliation["only_in_source_audit"], [])
        self.assertEqual(reconciliation["only_in_domain_sheets"], [])

    def test_result_resumption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                json.dumps({"experiment_id": "one"}) + "\n" + json.dumps({"experiment_id": "two"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_completed_ids(path), {"one", "two"})


class ParserTests(unittest.TestCase):
    def test_leading_choice(self) -> None:
        parsed = parse_choice("A\n\nThe following are unrelated questions.")
        self.assertEqual(parsed["parsed_choice"], "A")
        self.assertTrue(parsed["extra_generation"])
        self.assertFalse(parsed["format_error"])

    def test_explicit_choice(self) -> None:
        self.assertEqual(parse_choice("The answer is (C).")['parsed_choice'], "C")

    def test_ambiguous_choice_is_not_forced(self) -> None:
        parsed = parse_choice("It could be A or B")
        self.assertIsNone(parsed["parsed_choice"])
        self.assertTrue(parsed["format_error"])

    def test_term_text_is_not_silently_mapped_to_option(self) -> None:
        parsed = parse_choice("stat stick")
        self.assertIsNone(parsed["parsed_choice"])


if __name__ == "__main__":
    unittest.main()
