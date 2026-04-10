"""Dataset quality checks and scoring logic."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Fields considered "primary text" when checking length / near-duplicates
_TEXT_FIELDS = ["text", "instruction", "prompt", "input", "question"]
# Fields considered classification labels
_LABEL_FIELDS = ["label", "category", "class", "target"]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _default_config() -> Dict[str, Any]:
    return {
        "weights": {
            "json_format": 0.10,
            "field_consistency": 0.20,
            "missing_values": 0.20,
            "duplicates": 0.15,
            "near_duplicates": 0.10,
            "text_length": 0.10,
            "label_quality": 0.15,
        },
        "thresholds": {
            "field_consistency_pass": 0.95,
            "field_consistency_soft": 0.80,
            "missing_values_pass": 0.95,
            "missing_values_soft": 0.80,
            "duplicate_soft": 0.90,
            "near_duplicate_similarity": 0.85,
            "near_duplicate_sample": 500,
            "min_text_words": 3,
            "max_text_words": 2000,
        },
        "score_bands": {
            "ready": 90,
            "caution": 75,
            "needs_work": 50,
        },
    }


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from a YAML file (falls back to built-in defaults)."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        # Merge with defaults so missing keys don't cause KeyErrors
        defaults = _default_config()
        for section in ("weights", "thresholds", "score_bands"):
            if section not in cfg:
                cfg[section] = defaults[section]
            else:
                for k, v in defaults[section].items():
                    cfg[section].setdefault(k, v)
        return cfg
    except (FileNotFoundError, yaml.YAMLError):
        return _default_config()


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file and return a list of records."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(data: List[Dict[str, Any]]) -> str:
    """Detect the schema format of the dataset from the first record's keys."""
    if not data:
        return "unknown"
    keys = set(data[0].keys())
    if "messages" in keys:
        return "chatml"
    if "conversations" in keys:
        return "sharegpt"
    if "instruction" in keys and "output" in keys:
        return "alpaca"
    if "prompt" in keys and "completion" in keys:
        return "prompt_completion"
    return "generic"


# ---------------------------------------------------------------------------
# Individual checks  (all return Tuple[passed, message, score, details])
# ---------------------------------------------------------------------------

def check_json_format(data: List[Dict[str, Any]]) -> Tuple[bool, str, float, Dict]:
    """Verify the dataset is a non-empty list of dicts and detect its format."""
    if not data:
        return False, "Empty dataset", 0.0, {}
    if not isinstance(data, list):
        return False, "Data must be a list", 0.0, {}
    if not all(isinstance(item, dict) for item in data):
        return False, "All items must be dictionaries", 0.0, {}
    fmt = detect_format(data)
    return True, f"Valid JSON format (detected: {fmt})", 1.0, {"detected_format": fmt}


def check_field_consistency(
    data: List[Dict[str, Any]], config: Dict = None
) -> Tuple[bool, str, float, Dict]:
    """Check that every record contains the full set of fields present in the dataset."""
    if not data:
        return False, "Empty dataset", 0.0, {}

    cfg = (config or _default_config())["thresholds"]
    pass_t = cfg.get("field_consistency_pass", 0.95)
    soft_t = cfg.get("field_consistency_soft", 0.80)

    all_keys = set()
    for item in data:
        all_keys.update(item.keys())

    incomplete_rows: List[Dict] = []
    for i, item in enumerate(data):
        missing = sorted(all_keys - set(item.keys()))
        if missing:
            incomplete_rows.append({"row": i + 1, "missing_fields": missing})

    complete_count = len(data) - len(incomplete_rows)
    ratio = complete_count / len(data)

    details = {
        "expected_fields": sorted(all_keys),
        "total_incomplete": len(incomplete_rows),
        "incomplete_rows": incomplete_rows[:20],
    }

    if ratio >= pass_t:
        return True, "All records have consistent fields", 1.0, details
    elif ratio >= soft_t:
        score = ratio * 0.85
        return True, f"Most records consistent ({ratio*100:.1f}%)", score, details
    else:
        score = ratio * 0.5
        return False, f"Inconsistent fields ({ratio*100:.1f}% complete)", score, details


def check_missing_values(
    data: List[Dict[str, Any]], config: Dict = None
) -> Tuple[bool, str, float, Dict]:
    """Check for None or empty-string field values within records."""
    if not data:
        return False, "Empty dataset", 0.0, {}

    cfg = (config or _default_config())["thresholds"]
    pass_t = cfg.get("missing_values_pass", 0.95)
    soft_t = cfg.get("missing_values_soft", 0.80)

    total_fields = 0
    missing_count = 0
    affected_rows: List[Dict] = []

    for i, item in enumerate(data):
        row_missing = []
        for key, value in item.items():
            total_fields += 1
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_count += 1
                row_missing.append(key)
        if row_missing:
            affected_rows.append({"row": i + 1, "empty_fields": row_missing})

    completeness = 1 - (missing_count / total_fields) if total_fields > 0 else 0.0

    details = {
        "total_fields_checked": total_fields,
        "empty_field_count": missing_count,
        "total_affected_rows": len(affected_rows),
        "affected_rows": affected_rows[:20],
    }

    if completeness >= pass_t:
        return True, f"High completeness ({completeness*100:.1f}%)", completeness, details
    elif completeness >= soft_t:
        score = completeness * 0.85
        return True, f"Acceptable completeness ({completeness*100:.1f}%)", score, details
    else:
        score = completeness * 0.5
        return False, f"Low completeness ({completeness*100:.1f}%)", score, details


def check_duplicates(
    data: List[Dict[str, Any]], config: Dict = None
) -> Tuple[bool, str, float, Dict]:
    """Detect exact duplicate records (full JSON equality)."""
    if not data:
        return False, "Empty dataset", 0.0, {}

    cfg = (config or _default_config())["thresholds"]
    soft_t = cfg.get("duplicate_soft", 0.90)

    seen: Dict[str, int] = {}
    dup_rows: List[Dict] = []

    for i, item in enumerate(data):
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            dup_rows.append({"row": i + 1, "duplicate_of_row": seen[key] + 1})
        else:
            seen[key] = i

    uniqueness = 1 - len(dup_rows) / len(data)

    details = {
        "duplicate_count": len(dup_rows),
        "uniqueness_ratio": round(uniqueness, 4),
        "duplicate_rows": dup_rows[:20],
    }

    if uniqueness == 1.0:
        return True, "No exact duplicates found", 1.0, details
    elif uniqueness >= soft_t:
        score = uniqueness * 0.80
        return True, f"Few duplicates ({len(dup_rows)} found)", score, details
    else:
        score = uniqueness * 0.40
        return False, f"Many duplicates ({len(dup_rows)} found)", score, details


def check_near_duplicates(
    data: List[Dict[str, Any]], config: Dict = None
) -> Tuple[bool, str, float, Dict]:
    """Detect near-duplicate records using Jaccard word-set similarity.

    Only exact-duplicate-free pairs are flagged here (exact dups are caught by
    check_duplicates).  Comparison is capped at `near_duplicate_sample` records
    for performance.
    """
    if not data or len(data) < 2:
        return True, "Too few records to check near-duplicates", 1.0, {}

    cfg = (config or _default_config())["thresholds"]
    threshold = cfg.get("near_duplicate_similarity", 0.85)
    sample_size = int(cfg.get("near_duplicate_sample", 500))

    check_n = min(len(data), sample_size)
    sampled = check_n < len(data)

    # Build word-sets from primary text fields
    word_sets: List[set] = []
    for item in data[:check_n]:
        text = ""
        for field in _TEXT_FIELDS:
            val = item.get(field)
            if isinstance(val, str) and val.strip():
                text = val.lower()
                break
        word_sets.append(set(text.split()) if text else set())

    near_dup_pairs: List[Dict] = []
    for i in range(check_n):
        if not word_sets[i]:
            continue
        for j in range(i + 1, check_n):
            if not word_sets[j]:
                continue
            union = word_sets[i] | word_sets[j]
            if not union:
                continue
            sim = len(word_sets[i] & word_sets[j]) / len(union)
            # Skip similarity == 1.0 (exact duplicates already handled)
            if threshold <= sim < 1.0:
                near_dup_pairs.append(
                    {"row1": i + 1, "row2": j + 1, "similarity": round(sim, 3)}
                )

    total_pairs = check_n * (check_n - 1) / 2
    near_dup_ratio = len(near_dup_pairs) / total_pairs if total_pairs > 0 else 0.0

    suffix = f" (sampled first {check_n} records)" if sampled else ""
    details = {
        "similarity_threshold": threshold,
        "records_checked": check_n,
        "total_near_duplicate_pairs": len(near_dup_pairs),
        "near_duplicate_pairs": near_dup_pairs[:20],
    }

    if not near_dup_pairs:
        return True, f"No near-duplicates found{suffix}", 1.0, details
    n_pairs = len(near_dup_pairs)
    pair_word = "pair" if n_pairs == 1 else "pairs"
    if near_dup_ratio < 0.05:
        score = max(0.5, 1.0 - near_dup_ratio * 5)
        return True, f"Few near-duplicates ({n_pairs} {pair_word}){suffix}", score, details
    else:
        score = max(0.1, 1.0 - near_dup_ratio * 10)
        return False, f"Many near-duplicates ({n_pairs} {pair_word}){suffix}", score, details


def check_text_length(
    data: List[Dict[str, Any]], config: Dict = None
) -> Tuple[bool, str, float, Dict]:
    """Check that primary text fields fall within acceptable word-count bounds."""
    if not data:
        return False, "Empty dataset", 0.0, {}

    cfg = (config or _default_config())["thresholds"]
    min_w = int(cfg.get("min_text_words", 3))
    max_w = int(cfg.get("max_text_words", 2000))

    lengths: List[int] = []
    too_short: List[Dict] = []
    too_long: List[Dict] = []

    for i, item in enumerate(data):
        text = ""
        for field in _TEXT_FIELDS:
            val = item.get(field)
            if isinstance(val, str) and val.strip():
                text = val
                break
        if not text:
            continue
        wc = len(text.split())
        lengths.append(wc)
        if wc < min_w:
            too_short.append({"row": i + 1, "words": wc})
        elif wc > max_w:
            too_long.append({"row": i + 1, "words": wc})

    if not lengths:
        return True, "No text fields found to check length", 1.0, {}

    outlier_count = len(too_short) + len(too_long)
    outlier_ratio = outlier_count / len(data)
    avg_len = sum(lengths) / len(lengths)
    score = max(0.0, 1.0 - outlier_ratio * 2)

    details = {
        "avg_word_count": round(avg_len, 1),
        "min_word_count": min(lengths),
        "max_word_count": max(lengths),
        "total_outliers": outlier_count,
        "too_short_rows": too_short[:20],
        "too_long_rows": too_long[:20],
    }

    rec_word = "record" if outlier_count == 1 else "records"
    if outlier_ratio == 0:
        return True, f"Good length distribution (avg {avg_len:.1f} words)", 1.0, details
    elif outlier_ratio < 0.10:
        return True, f"Minor length outliers ({outlier_count} {rec_word}, avg {avg_len:.1f} words)", score, details
    else:
        return False, f"Many length outliers ({outlier_count} {rec_word}, avg {avg_len:.1f} words)", score, details


def check_label_quality(
    data: List[Dict[str, Any]], config: Dict = None
) -> Tuple[bool, str, float, Dict]:
    """Check for severe label imbalance in classification datasets."""
    if not data:
        return False, "Empty dataset", 0.0, {}

    labels_found: List[str] = []
    for item in data:
        for field in _LABEL_FIELDS:
            val = item.get(field)
            if val is not None:
                labels_found.append(str(val))
                break

    if not labels_found:
        return True, "No label field found (non-classification dataset)", 1.0, {}

    unique_labels = set(labels_found)
    counts = {lbl: labels_found.count(lbl) for lbl in unique_labels}
    max_c = max(counts.values())
    min_c = min(counts.values())
    imbalance = max_c / min_c if min_c > 0 else float("inf")

    details = {
        "num_classes": len(unique_labels),
        "label_distribution": counts,
        "imbalance_ratio": round(imbalance, 2),
    }

    if imbalance > 10:
        return False, f"Severe label imbalance ({max_c}:{min_c} ratio)", 0.3, details
    elif imbalance > 5:
        return True, f"Moderate label imbalance ({len(unique_labels)} classes)", 0.7, details
    return True, f"Balanced labels ({len(unique_labels)} classes)", 1.0, details


# ---------------------------------------------------------------------------
# Master scoring function
# ---------------------------------------------------------------------------

def calculate_quality_score(
    data: List[Dict[str, Any]], config: Dict = None
) -> Dict[str, Any]:
    """Run all checks and return a weighted overall quality score (0-100).

    Returns a dict with keys: overall_score, grade, checks, num_records,
    detected_format.
    """
    cfg = config or load_config()
    weights = cfg.get("weights", {})
    bands = cfg.get("score_bands", {"ready": 90, "caution": 75, "needs_work": 50})

    check_runners = [
        ("json_format",        lambda: check_json_format(data)),
        ("field_consistency",  lambda: check_field_consistency(data, cfg)),
        ("missing_values",     lambda: check_missing_values(data, cfg)),
        ("duplicates",         lambda: check_duplicates(data, cfg)),
        ("near_duplicates",    lambda: check_near_duplicates(data, cfg)),
        ("text_length",        lambda: check_text_length(data, cfg)),
        ("label_quality",      lambda: check_label_quality(data, cfg)),
    ]

    default_w = 1.0 / len(check_runners)
    results: Dict[str, Any] = {}
    total_weighted = 0.0
    total_weight = 0.0

    for name, runner in check_runners:
        passed, message, score, details = runner()
        w = weights.get(name, default_w)
        results[name] = {
            "passed": passed,
            "message": message,
            "score": score,
            "weight": w,
            "details": details,
        }
        total_weighted += score * w
        total_weight += w

    final_score = (total_weighted / total_weight * 100) if total_weight > 0 else 0.0

    if final_score >= bands["ready"]:
        grade = "READY"
    elif final_score >= bands["caution"]:
        grade = "CAUTION"
    elif final_score >= bands["needs_work"]:
        grade = "NEEDS_WORK"
    else:
        grade = "NOT_READY"

    detected_fmt = results.get("json_format", {}).get("details", {}).get("detected_format", "unknown")

    return {
        "overall_score": final_score,
        "grade": grade,
        "checks": results,
        "num_records": len(data),
        "detected_format": detected_fmt,
    }
