"""Dataset quality checks and scoring logic."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def check_json_format(data: List[Dict[str, Any]]) -> Tuple[bool, str, float]:
    """Check if data is valid JSON format."""
    if not data:
        return False, "Empty dataset", 0.0
    if not isinstance(data, list):
        return False, "Data must be a list", 0.0
    if not all(isinstance(item, dict) for item in data):
        return False, "All items must be dictionaries", 0.0
    return True, "Valid JSON format", 1.0


def check_field_consistency(data: List[Dict[str, Any]]) -> Tuple[bool, str, float]:
    """Check if all records have consistent fields."""
    if not data:
        return False, "Empty dataset", 0.0
    
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    
    complete_count = sum(1 for item in data if set(item.keys()) == all_keys)
    consistency_ratio = complete_count / len(data)
    
    if consistency_ratio == 1.0:
        return True, "All records have consistent fields", consistency_ratio
    elif consistency_ratio >= 0.8:
        return True, "Most records consistent (" + str(round(consistency_ratio*100, 1)) + "%)", consistency_ratio
    else:
        return False, "Inconsistent fields (" + str(round(consistency_ratio*100, 1)) + "% complete)", consistency_ratio


def check_missing_values(data: List[Dict[str, Any]]) -> Tuple[bool, str, float]:
    """Check for missing or null values."""
    if not data:
        return False, "Empty dataset", 0.0
    
    total_fields = 0
    missing_count = 0
    
    for item in data:
        for key, value in item.items():
            total_fields += 1
            if value is None or (isinstance(value, str) and value.strip() == ""):
                missing_count += 1
    
    missing_ratio = missing_count / total_fields if total_fields > 0 else 0
    completeness = 1 - missing_ratio
    
    if completeness >= 0.95:
        return True, "High completeness (" + str(round(completeness*100, 1)) + "%)", completeness
    elif completeness >= 0.8:
        return True, "Acceptable completeness (" + str(round(completeness*100, 1)) + "%)", completeness
    else:
        return False, "Low completeness (" + str(round(completeness*100, 1)) + "%)", completeness


def check_duplicates(data: List[Dict[str, Any]]) -> Tuple[bool, str, float]:
    """Check for duplicate records."""
    if not data:
        return False, "Empty dataset", 0.0
    
    seen = set()
    duplicate_count = 0
    
    for item in data:
        item_str = json.dumps(item, sort_keys=True)
        if item_str in seen:
            duplicate_count += 1
        else:
            seen.add(item_str)
    
    duplicate_ratio = duplicate_count / len(data)
    uniqueness = 1 - duplicate_ratio
    
    if uniqueness == 1.0:
        return True, "No duplicates found", uniqueness
    elif uniqueness >= 0.9:
        return True, "Few duplicates (" + str(duplicate_count) + " found)", uniqueness
    else:
        return False, "Many duplicates (" + str(duplicate_count) + " found)", uniqueness


def check_label_quality(data: List[Dict[str, Any]]) -> Tuple[bool, str, float]:
    """Check label quality for classification tasks."""
    if not data:
        return False, "Empty dataset", 0.0
    
    label_fields = ['label', 'category', 'class', 'target', 'answer']
    labels_found = []
    
    for item in data:
        for field in label_fields:
            if field in item:
                labels_found.append(item[field])
                break
    
    if not labels_found:
        return True, "No label field found (may not be classification)", 1.0
    
    unique_labels = set(labels_found)
    label_counts = {label: labels_found.count(label) for label in unique_labels}
    
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    
    if min_count == 0 or max_count / min_count > 10:
        return False, "Severe label imbalance", 0.5
    
    return True, "Balanced labels (" + str(len(unique_labels)) + " classes)", 1.0


def calculate_quality_score(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall quality score for a dataset."""
    checks = [
        ("json_format", check_json_format(data)),
        ("field_consistency", check_field_consistency(data)),
        ("missing_values", check_missing_values(data)),
        ("duplicates", check_duplicates(data)),
        ("label_quality", check_label_quality(data)),
    ]
    
    scores = {}
    total_score = 0
    
    for name, (passed, message, score) in checks:
        scores[name] = {
            "passed": passed,
            "message": message,
            "score": score
        }
        total_score += score
    
    final_score = (total_score / len(checks)) * 100
    
    return {
        "overall_score": final_score,
        "checks": scores,
        "num_records": len(data)
    }
