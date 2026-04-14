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
    check_output_diversity,
    check_text_length,
    check_token_length,
    check_instruction_quality,
    check_language_consistency,
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

    def test_alpaca_optional_input_not_penalised(self):
        """Empty `input` in Alpaca datasets is structurally valid and must not
        lower the completeness score or trigger a FAIL."""
        data = [
            {"instruction": "What is gravity?", "input": "", "output": "A fundamental force."},
            {"instruction": "Explain recursion.", "input": "", "output": "A function calling itself."},
            {"instruction": "Translate to French.", "input": "Hello world", "output": "Bonjour monde"},
        ]
        passed, message, score, details = check_missing_values(data)
        assert passed is True
        assert score == 1.0
        assert "input" in details["skipped_optional_fields"]

    def test_skipped_optional_fields_key_always_present(self):
        """details must always include skipped_optional_fields, even when empty."""
        data = [{"text": "hello", "label": "x"}]
        _, _, _, details = check_missing_values(data)
        assert "skipped_optional_fields" in details


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

    def test_identical_instruction_different_input_flagged(self):
        """Two records sharing the same instruction but different inputs have
        sim==1.0 on the text field yet differ as full records — they are NOT
        exact duplicates but ARE near-duplicates and must be flagged."""
        data = [
            {
                "instruction": "Summarise this paragraph in one sentence.",
                "input": "The Amazon rainforest spans nine countries.",
                "output": "The Amazon covers nine countries.",
            },
            {
                "instruction": "Summarise this paragraph in one sentence.",
                "input": "The Sahara is the world's largest hot desert.",
                "output": "The Sahara is the largest hot desert.",
            },
        ]
        _, _, _, details = check_near_duplicates(data)
        assert details["total_near_duplicate_pairs"] == 1

    def test_exact_duplicates_not_double_counted(self):
        """Fully identical Alpaca records must not appear in near-duplicate
        results — they are handled exclusively by check_duplicates."""
        data = [
            {"instruction": "What is Python?", "input": "", "output": "A programming language."},
            {"instruction": "What is Python?", "input": "", "output": "A programming language."},
        ]
        _, _, _, details = check_near_duplicates(data)
        assert details["total_near_duplicate_pairs"] == 0


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
# check_output_diversity
# ---------------------------------------------------------------------------

class TestOutputDiversity:
    def test_diverse_outputs_pass(self):
        data = [
            {"instruction": "Q1", "input": "", "output": "The quick brown fox jumps over the lazy dog"},
            {"instruction": "Q2", "input": "", "output": "Machine learning uses statistical methods to find patterns"},
            {"instruction": "Q3", "input": "", "output": "Python is a high-level general-purpose language"},
        ]
        passed, _, score, details = check_output_diversity(data)
        assert passed is True
        assert score == 1.0
        assert details["total_similar_output_pairs"] == 0

    def test_identical_outputs_flagged(self):
        code = "def sort_list(lst): return sorted(lst)"
        data = [
            {"instruction": "Write a sort function in Python", "input": "", "output": code},
            {"instruction": "Create a function that sorts a list", "input": "", "output": code},
        ]
        _, _, _, details = check_output_diversity(data)
        assert details["total_similar_output_pairs"] >= 1

    def test_no_output_field_passes(self):
        data = [{"text": "hello world today"}, {"text": "machine learning concepts"}]
        passed, _, score, _ = check_output_diversity(data)
        assert passed is True
        assert score == 1.0


# ---------------------------------------------------------------------------
# check_token_length
# ---------------------------------------------------------------------------

class TestTokenLength:
    def test_short_records_pass(self):
        data = [
            {"instruction": "What is Python?", "input": "", "output": "A programming language."},
            {"instruction": "What is Java?", "input": "", "output": "A compiled language."},
        ]
        passed, _, score, details = check_token_length(data)
        assert passed is True
        assert score == 1.0
        assert details["total_overflow_rows"] == 0

    def test_long_record_flagged(self):
        long_text = "word " * 2000  # ~2600 estimated tokens
        data = [{"instruction": long_text, "input": "", "output": "short answer"}]
        _, _, _, details = check_token_length(data)
        assert details["total_overflow_rows"] == 1

    def test_avg_tokens_reported_in_details(self):
        data = [{"instruction": "ten words in this instruction text here right", "input": "", "output": "ok"}]
        _, _, _, details = check_token_length(data)
        assert "avg_estimated_tokens" in details
        assert details["avg_estimated_tokens"] > 0


# ---------------------------------------------------------------------------
# check_instruction_quality
# ---------------------------------------------------------------------------

class TestInstructionQuality:
    def test_good_alpaca_instructions_pass(self):
        data = [
            {"instruction": "Write a Python function to sort a list of integers.", "input": "", "output": "..."},
            {"instruction": "Explain the difference between supervised and unsupervised learning.", "input": "", "output": "..."},
        ]
        passed, _, score, details = check_instruction_quality(data)
        assert passed is True
        assert score == 1.0
        assert details["total_flagged"] == 0

    def test_vague_instruction_flagged(self):
        data = [
            {"instruction": "help me", "input": "", "output": "sure"},
            {"instruction": "Explain gradient descent in detail with examples.", "input": "", "output": "..."},
        ]
        _, _, _, details = check_instruction_quality(data)
        assert details["total_flagged"] >= 1

    def test_skipped_for_non_alpaca_format(self):
        # Generic format — check must return 1.0 to avoid false positives
        data = [{"text": "hi", "label": "x"}, {"text": "ok", "label": "y"}]
        passed, _, score, _ = check_instruction_quality(data)
        assert passed is True
        assert score == 1.0


# ---------------------------------------------------------------------------
# check_language_consistency
# ---------------------------------------------------------------------------

class TestLanguageConsistency:
    def test_consistent_ascii_passes(self):
        data = [
            {"text": "What is the capital of France?"},
            {"text": "Explain the concept of machine learning step by step."},
            {"text": "Write a function to sort a list of integers efficiently."},
        ]
        passed, _, score, details = check_language_consistency(data)
        assert passed is True
        assert score == 1.0
        assert details["total_anomalous_rows"] == 0

    def test_script_switch_flagged(self):
        data = [
            {"text": "What is the capital of France?"},
            {"text": "Explain machine learning concepts clearly and simply."},
            {"text": "什么是机器学习？这是一个非常重要的问题，值得深入研究。"},
        ]
        _, _, _, details = check_language_consistency(data)
        assert details["total_anomalous_rows"] >= 1

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
        assert result["overall_score"] >= 80   # tightened scoring lowers clean fixture to ~86
        assert result["num_records"] == 10
        assert result["grade"] in ("READY", "CAUTION")  # may be CAUTION under stricter bands

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
        # A perfectly balanced dataset with no duplicates / empty fields should score very high
        data = [
            {"text": "What is the speed of light in a vacuum?", "label": "science"},
            {"text": "Who painted the Mona Lisa and when?", "label": "art"},
            {"text": "What is the boiling point of water at sea level?", "label": "science"},
            {"text": "Who wrote Romeo and Juliet?", "label": "art"},
        ]
        result = calculate_quality_score(data)
        # With balanced labels (2:2 = 1.0 ratio) every check should pass → 100
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
            "output_diversity",
            "text_length",
            "token_length",
            "instruction_quality",
            "language_consistency",
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
