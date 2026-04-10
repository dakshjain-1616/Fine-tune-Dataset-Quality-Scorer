"""Tests for dataset quality checks."""

import json
import pytest
from pathlib import Path

from src.checks import (
    load_dataset,
    load_config,
    detect_format,
    check_json_format,
    check_field_consistency,
    check_missing_values,
    check_duplicates,
    check_near_duplicates,
    check_text_length,
    check_label_quality,
    calculate_quality_score,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_load_good_dataset(self):
        data = load_dataset(FIXTURES_DIR / "good_dataset.jsonl")
        assert len(data) == 10
        assert all(isinstance(item, dict) for item in data)

    def test_load_bad_dataset(self):
        data = load_dataset(FIXTURES_DIR / "bad_dataset.jsonl")
        assert len(data) == 10

    def test_load_mixed_dataset(self):
        data = load_dataset(FIXTURES_DIR / "mixed_dataset.jsonl")
        assert len(data) == 10


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_default_when_file_missing(self):
        cfg = load_config("/nonexistent/path.yaml")
        assert "weights" in cfg
        assert "thresholds" in cfg
        assert "score_bands" in cfg

    def test_weights_sum_near_one(self):
        cfg = load_config()
        total = sum(cfg["weights"].values())
        assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------

class TestDetectFormat:
    def test_chatml(self):
        assert detect_format([{"messages": []}]) == "chatml"

    def test_alpaca(self):
        assert detect_format([{"instruction": "x", "output": "y"}]) == "alpaca"

    def test_prompt_completion(self):
        assert detect_format([{"prompt": "x", "completion": "y"}]) == "prompt_completion"

    def test_sharegpt(self):
        assert detect_format([{"conversations": []}]) == "sharegpt"

    def test_generic(self):
        assert detect_format([{"text": "x", "label": "y"}]) == "generic"

    def test_empty(self):
        assert detect_format([]) == "unknown"


# ---------------------------------------------------------------------------
# check_json_format
# ---------------------------------------------------------------------------

class TestJsonFormat:
    def test_valid_json(self):
        passed, message, score, details = check_json_format([{"text": "hi", "label": "x"}])
        assert passed is True
        assert score == 1.0
        assert "detected_format" in details

    def test_empty_dataset(self):
        passed, message, score, details = check_json_format([])
        assert passed is False
        assert score == 0.0


# ---------------------------------------------------------------------------
# check_field_consistency
# ---------------------------------------------------------------------------

class TestFieldConsistency:
    def test_consistent_fields(self):
        data = [{"text": "a", "label": "1"}, {"text": "b", "label": "2"}]
        passed, message, score, details = check_field_consistency(data)
        assert passed is True
        assert score == 1.0
        assert details["total_incomplete"] == 0

    def test_inconsistent_fields(self):
        data = [{"text": "a", "label": "1"}, {"text": "b"}]
        passed, message, score, details = check_field_consistency(data)
        assert score < 1.0
        assert details["total_incomplete"] == 1

    def test_details_contain_row_numbers(self):
        data = [{"text": "a", "label": "1"}, {"text": "b"}]
        _, _, _, details = check_field_consistency(data)
        assert details["incomplete_rows"][0]["row"] == 2
        assert "label" in details["incomplete_rows"][0]["missing_fields"]

    def test_score_penalised_on_soft_fail(self):
        # 8/10 records complete → soft pass, score should be < 0.8
        data = [{"text": "a", "label": "1"}] * 8 + [{"text": "b"}] * 2
        _, _, score, _ = check_field_consistency(data)
        assert score < 0.8

    def test_hard_fail_below_threshold(self):
        # 6/10 records complete → hard fail, score should be heavily penalised
        data = [{"text": "a", "label": "1"}] * 6 + [{"text": "b"}] * 4
        passed, _, score, _ = check_field_consistency(data)
        assert passed is False
        assert score < 0.5


# ---------------------------------------------------------------------------
# check_missing_values
# ---------------------------------------------------------------------------

class TestMissingValues:
    def test_no_missing(self):
        _, _, score, details = check_missing_values([{"text": "hello", "label": "x"}])
        assert score == 1.0
        assert details["empty_field_count"] == 0

    def test_with_empty_string(self):
        _, _, score, details = check_missing_values([{"text": "", "label": "x"}])
        assert score < 1.0
        assert details["empty_field_count"] == 1

    def test_with_none(self):
        _, _, score, details = check_missing_values([{"text": None, "label": "x"}])
        assert score < 1.0

    def test_with_whitespace(self):
        _, _, score, details = check_missing_values([{"text": "   ", "label": "x"}])
        assert score < 1.0

    def test_affected_rows_reported(self):
        data = [{"text": "ok", "label": "x"}, {"text": "", "label": "y"}]
        _, _, _, details = check_missing_values(data)
        assert details["total_affected_rows"] == 1
        assert details["affected_rows"][0]["row"] == 2


# ---------------------------------------------------------------------------
# check_duplicates
# ---------------------------------------------------------------------------

class TestDuplicates:
    def test_no_duplicates(self):
        data = [{"text": "a", "label": "1"}, {"text": "b", "label": "2"}]
        passed, message, score, details = check_duplicates(data)
        assert score == 1.0
        assert details["duplicate_count"] == 0

    def test_with_duplicates(self):
        data = [{"text": "a", "label": "1"}, {"text": "a", "label": "1"}]
        passed, message, score, details = check_duplicates(data)
        assert score < 1.0
        assert details["duplicate_count"] == 1

    def test_duplicate_row_numbers_reported(self):
        data = [{"text": "a"}, {"text": "b"}, {"text": "a"}]
        _, _, _, details = check_duplicates(data)
        assert details["duplicate_rows"][0]["row"] == 3
        assert details["duplicate_rows"][0]["duplicate_of_row"] == 1

    def test_many_duplicates_hard_fail(self):
        # 5 duplicates out of 10 → uniqueness 0.5 → hard fail
        base = {"text": "dup", "label": "x"}
        data = [base] * 10
        passed, _, score, _ = check_duplicates(data)
        assert passed is False
        assert score < 0.5


# ---------------------------------------------------------------------------
# check_near_duplicates
# ---------------------------------------------------------------------------

class TestNearDuplicates:
    def test_diverse_texts_pass(self):
        data = [
            {"text": "The quick brown fox jumps over the lazy dog"},
            {"text": "Machine learning models require large datasets"},
            {"text": "Python is a popular programming language"},
        ]
        passed, _, score, details = check_near_duplicates(data)
        assert passed is True
        assert score == 1.0
        assert details["total_near_duplicate_pairs"] == 0

    def test_near_duplicate_detected(self):
        # Two sentences sharing 15/17 words → Jaccard ≈ 0.88, above the 0.85 threshold
        data = [
            {"text": "what is the best way to train a neural network on a large dataset for classification tasks"},
            {"text": "what is the best way to train a neural network on a large dataset for text classification"},
        ]
        _, _, _, details = check_near_duplicates(data)
        assert details["total_near_duplicate_pairs"] >= 1

    def test_exact_duplicates_excluded(self):
        # Exact dups have similarity=1.0 and must NOT appear in near-dup results
        data = [{"text": "identical text"}, {"text": "identical text"}]
        _, _, _, details = check_near_duplicates(data)
        assert details["total_near_duplicate_pairs"] == 0

    def test_too_few_records(self):
        passed, _, score, _ = check_near_duplicates([{"text": "only one"}])
        assert passed is True
        assert score == 1.0


# ---------------------------------------------------------------------------
# check_text_length
# ---------------------------------------------------------------------------

class TestTextLength:
    def test_good_lengths(self):
        data = [
            {"text": "What is the capital of France?"},
            {"text": "Who wrote Romeo and Juliet?"},
        ]
        passed, _, score, details = check_text_length(data)
        assert passed is True
        assert score == 1.0
        assert details["total_outliers"] == 0

    def test_too_short_flagged(self):
        data = [{"text": "hi"}, {"text": "A normal length question about science?"}]
        _, _, score, details = check_text_length(data)
        assert details["total_outliers"] >= 1
        assert details["too_short_rows"][0]["row"] == 1

    def test_no_text_field(self):
        data = [{"label": "x"}, {"label": "y"}]
        passed, _, score, _ = check_text_length(data)
        assert passed is True  # graceful — no text to check

    def test_row_numbers_in_details(self):
        data = [{"text": "hi"}, {"text": "Good length sentence here okay?"}]
        _, _, _, details = check_text_length(data)
        assert details["too_short_rows"][0]["row"] == 1


# ---------------------------------------------------------------------------
# check_label_quality
# ---------------------------------------------------------------------------

class TestLabelQuality:
    def test_balanced_labels(self):
        data = [
            {"text": "a", "label": "pos"},
            {"text": "b", "label": "neg"},
            {"text": "c", "label": "pos"},
            {"text": "d", "label": "neg"},
        ]
        passed, message, score, details = check_label_quality(data)
        assert score == 1.0
        assert details["num_classes"] == 2

    def test_severe_imbalance(self):
        data = [{"text": str(i), "label": "pos"} for i in range(20)]
        data.append({"text": "x", "label": "neg"})
        passed, message, score, details = check_label_quality(data)
        assert passed is False
        assert score <= 0.3
        assert details["imbalance_ratio"] > 10

    def test_no_label_field_passes(self):
        data = [{"text": "hello"}, {"text": "world"}]
        passed, message, score, _ = check_label_quality(data)
        assert passed is True
        assert score == 1.0

    def test_label_distribution_in_details(self):
        data = [{"label": "a"}, {"label": "b"}, {"label": "a"}]
        _, _, _, details = check_label_quality(data)
        assert details["label_distribution"]["a"] == 2
        assert details["label_distribution"]["b"] == 1


# ---------------------------------------------------------------------------
# calculate_quality_score
# ---------------------------------------------------------------------------

class TestCalculateQualityScore:
    def test_good_dataset_score(self):
        data = load_dataset(FIXTURES_DIR / "good_dataset.jsonl")
        result = calculate_quality_score(data)
        assert result["overall_score"] >= 90
        assert result["num_records"] == 10
        assert result["grade"] == "READY"

    def test_bad_dataset_score_is_low(self):
        data = load_dataset(FIXTURES_DIR / "bad_dataset.jsonl")
        result = calculate_quality_score(data)
        # Bad dataset has missing fields, duplicates, empty values — should NOT be "excellent"
        assert result["overall_score"] < 75
        assert result["grade"] in ("CAUTION", "NEEDS_WORK", "NOT_READY")

    def test_mixed_dataset_score(self):
        data = load_dataset(FIXTURES_DIR / "mixed_dataset.jsonl")
        result = calculate_quality_score(data)
        assert 0 <= result["overall_score"] <= 100

    def test_result_structure(self):
        data = [{"text": "test question here", "label": "x"}]
        result = calculate_quality_score(data)
        assert "overall_score" in result
        assert "grade" in result
        assert "checks" in result
        assert "num_records" in result
        assert "detected_format" in result

    def test_grade_field_present(self):
        data = load_dataset(FIXTURES_DIR / "good_dataset.jsonl")
        result = calculate_quality_score(data)
        assert result["grade"] in ("READY", "CAUTION", "NEEDS_WORK", "NOT_READY")

    def test_weighted_score_uses_config(self):
        # A dataset with all checks passing should score 100
        data = [
            {"text": "What is the speed of light in a vacuum?", "label": "science"},
            {"text": "Who painted the Mona Lisa and when?", "label": "art"},
            {"text": "What is the boiling point of water?", "label": "science"},
        ]
        result = calculate_quality_score(data)
        assert result["overall_score"] == 100.0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestQualityChecksIntegration:
    def test_all_checks_present(self):
        data = load_dataset(FIXTURES_DIR / "good_dataset.jsonl")
        result = calculate_quality_score(data)
        expected_checks = [
            "json_format",
            "field_consistency",
            "missing_values",
            "duplicates",
            "near_duplicates",
            "text_length",
            "label_quality",
        ]
        for check in expected_checks:
            assert check in result["checks"], f"Missing check: {check}"
            assert "passed" in result["checks"][check]
            assert "message" in result["checks"][check]
            assert "score" in result["checks"][check]
            assert "weight" in result["checks"][check]
            assert "details" in result["checks"][check]

    def test_check_weights_positive(self):
        data = [{"text": "test sentence here", "label": "x"}]
        result = calculate_quality_score(data)
        for name, check in result["checks"].items():
            assert check["weight"] > 0, f"Weight for {name} must be positive"

    def test_bad_dataset_has_failures(self):
        data = load_dataset(FIXTURES_DIR / "bad_dataset.jsonl")
        result = calculate_quality_score(data)
        failed = [name for name, c in result["checks"].items() if not c["passed"]]
        assert len(failed) >= 2, "Bad dataset should fail multiple checks"
