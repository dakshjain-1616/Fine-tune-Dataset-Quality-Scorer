"""Dataset quality checks and scoring logic."""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Warn once if datasets/huggingface_hub not available (optional dep)
try:
    from datasets import load_dataset as _hf_load_dataset  # type: ignore
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

_LARGE_DATASET_WARN = 50_000  # warn when loading more than this many records

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Fields considered "primary text" when checking length / near-duplicates
_TEXT_FIELDS = ["text", "instruction", "prompt", "input", "question"]
# Fields considered classification labels
_LABEL_FIELDS = ["label", "category", "class", "target"]

# Fields that are intentionally optional (empty = valid) per detected format.
# Empty values in these fields are excluded from check_missing_values so that
# structurally-optional fields don't produce false-positive completeness failures.
_FORMAT_OPTIONAL_FIELDS: Dict[str, set] = {
    "alpaca": {"input"},   # Alpaca `input` provides extra context — absent means none needed
}

# ---------------------------------------------------------------------------
# Domain detection constants
# ---------------------------------------------------------------------------

# Keywords used to score each subject-matter domain.
# Lists are intentionally broad so a single text match contributes signal.
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "coding": [
        "python", "javascript", "java", "typescript", "c++", "rust", "golang", "bash",
        "function", "algorithm", "implement", "write a function", "code", "program",
        "def ", "class ", "sql", "select", "query", "debug", "fix the", "refactor",
        "loop", "recursion", "array", "string", "sort", "search", "api", "database",
        "regex", "script", "variable", "return ", "import ",
    ],
    "translation": [
        "translate", "translation", "french", "spanish", "german", "chinese",
        "japanese", "portuguese", "arabic", "hindi", "korean", "italian", "language",
    ],
    "summarization": [
        "summarize", "summarise", "summary", "tldr", "key points", "main idea",
        "following paragraph", "following passage", "following article", "condense",
    ],
    "qa": [
        "what is", "who is", "when did", "where is", "why does", "how does",
        "what are", "explain", "describe", "define", "what does", "capital of",
        "who wrote", "how many", "what year", "tell me about",
    ],
}

# Domain-specific suggestions shown in the analyse command.
_DOMAIN_SUGGESTIONS: Dict[str, List[str]] = {
    "coding": [
        "Add multiple programming languages — JavaScript, SQL, and Bash improve model versatility",
        "Include debugging/fixing tasks (~30% of data): 'find the bug in this function'",
        "Add edge-case examples: empty inputs, None/null values, boundary conditions",
        "Vary task complexity: short snippets (5–10 lines), medium functions, multi-function programs",
        "Add code-explanation tasks: 'what does this code do?' — teaches analytical reasoning",
        "Include unit-test writing tasks to teach correctness awareness",
        "Add refactoring tasks: 'improve this code for readability / performance'",
    ],
    "translation": [
        "Add bidirectional pairs (A→B and B→A) for balanced language coverage",
        "Include formal, informal, and technical register examples",
        "Add longer complex sentences with nested clauses — not just simple phrases",
        "Include idiomatic expressions that don't translate word-for-word",
        "Expand to multiple target languages if multilingual capability is needed",
    ],
    "summarization": [
        "Vary input length: short paragraphs (~100 words), medium passages (~500 words), long articles (1000+)",
        "Keep output-to-input compression ratio consistent (~20–30%) across records",
        "Include diverse source genres: news, academic, technical documentation, narrative",
        "Mix extractive style (key sentences verbatim) and abstractive style (rephrased)",
        "Add single-sentence TL;DR tasks alongside multi-sentence summaries",
    ],
    "qa": [
        "Balance question types: factual (who/what/when), conceptual (why/how), procedural",
        "Cover diverse knowledge domains: science, history, math, everyday common sense",
        "Mix short one-sentence answers with longer explanatory responses",
        "Add multi-hop questions that require chaining two or more facts",
        "Include graceful 'I don't know' examples for clearly out-of-scope questions",
    ],
    "classification": [
        "Ensure balanced classes — severe imbalance degrades minority-class performance",
        "Cover the full expected label set, including boundary and ambiguous cases",
        "Add genuinely ambiguous examples that sit near decision boundaries",
        "Augment under-represented classes with paraphrased or reworded variants",
    ],
    "conversation": [
        "Include multi-turn exchanges, not just single Q&A pairs",
        "Vary conversation depth: 2-turn, 4-turn, and longer threads",
        "Add clarification and follow-up question patterns",
        "Include graceful refusal examples for out-of-scope requests",
        "Ensure consistent assistant persona and tone across all conversations",
    ],
    "general": [
        "Group records by task type for a cleaner, more consistent training signal",
        "Standardise instruction phrasing — mixing imperatives and questions can weaken training",
        "Review output length variance — high variance often signals inconsistent quality",
        "Consider adding a task_type metadata field to enable stratified sampling",
    ],
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _default_config() -> Dict[str, Any]:
    return {
        "weights": {
            "json_format": 0.08,
            "field_consistency": 0.16,
            "missing_values": 0.16,
            "duplicates": 0.12,
            "near_duplicates": 0.08,
            "output_diversity": 0.07,
            "text_length": 0.07,
            "token_length": 0.07,
            "instruction_quality": 0.05,
            "language_consistency": 0.04,
            "label_quality": 0.10,
        },
        "thresholds": {
            "field_consistency_pass": 0.97,
            "field_consistency_soft": 0.90,
            "missing_values_pass": 0.98,
            "missing_values_soft": 0.92,
            "duplicate_soft": 0.97,
            "near_duplicate_similarity": 0.72,
            "near_duplicate_sample": 500,
            "min_text_words": 5,
            "max_text_words": 2000,
            "max_token_estimate": 2048,
            "min_instruction_words": 5,
        },
        "score_bands": {
            "ready": 92,
            "caution": 80,
            "needs_work": 60,
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

def load_dataset(filepath: str, limit: int = 0) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file and return a list of records.

    Args:
        filepath: Path to JSONL file.
        limit: If > 0, stop after this many records (useful for previewing large files).
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if limit > 0 and len(data) >= limit:
                break
    if len(data) >= _LARGE_DATASET_WARN and limit == 0:
        warnings.warn(
            f"Loaded {len(data):,} records. Near-duplicate checks are capped at "
            f"{500} records for performance. Use --limit to reduce memory usage.",
            stacklevel=2,
        )
    return data


def load_hf_dataset(
    dataset_name: str,
    split: str = "train",
    config_name: Optional[str] = None,
    limit: int = 1000,
    prompt_field: Optional[str] = None,
    completion_field: Optional[str] = None,
    filter_field: Optional[str] = None,
    filter_value: Optional[str] = None,
    streaming: bool = True,
) -> List[Dict[str, Any]]:
    """Download a HuggingFace dataset and return records as plain dicts.

    If ``prompt_field`` and ``completion_field`` are given the returned records
    will have ``{"prompt": ..., "completion": ...}`` keys, making them
    directly scoreable by the quality checker.  Otherwise the raw HF fields
    are preserved.

    Args:
        dataset_name: HuggingFace dataset identifier, e.g. ``"open-index/hacker-news"``.
        split: Dataset split (``"train"``, ``"test"``, …).
        config_name: Named configuration / subset, if any.
        limit: Maximum number of records to load (default 1000).
        prompt_field: HF column to map to ``"prompt"``.
        completion_field: HF column to map to ``"completion"``.
        filter_field: Column to filter on (e.g. ``"type"``).
        filter_value: Keep only rows where ``filter_field == filter_value``.
        streaming: Use HF streaming mode to avoid downloading the full dataset.
    """
    if not _HF_AVAILABLE:
        raise RuntimeError(
            "The 'datasets' package is required for HuggingFace support.\n"
            "Install it with:  pip install datasets"
        )

    load_kwargs: Dict[str, Any] = {"streaming": streaming}
    if config_name:
        load_kwargs["name"] = config_name

    hf_ds = _hf_load_dataset(dataset_name, split=split, **load_kwargs)

    records: List[Dict[str, Any]] = []
    for row in hf_ds:
        if filter_field and filter_value is not None:
            row_val = str(row.get(filter_field, ""))
            if row_val != str(filter_value):
                continue

        if prompt_field or completion_field:
            record: Dict[str, Any] = {}
            if prompt_field:
                val = row.get(prompt_field)
                record["prompt"] = str(val) if val is not None else ""
            if completion_field:
                val = row.get(completion_field)
                record["completion"] = str(val) if val is not None else ""
            # Carry a few useful metadata fields; ensure JSON-serialisable types
            for meta in ("id", "score", "type", "by", "time"):
                if meta in row and meta not in record:
                    val = row[meta]
                    record[meta] = str(val) if not isinstance(val, (str, int, float, bool, type(None))) else val
        else:
            # Flatten: convert every value to a JSON-serialisable type
            record = {}
            for k, v in row.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    record[k] = v
                else:
                    record[k] = str(v)

        # Skip records where both main text fields are empty
        if not any(record.get(f, "").strip() for f in ("prompt", "text", "title", "instruction") if isinstance(record.get(f), str)):
            if not any(isinstance(record.get(f), str) and record.get(f, "").strip() for f in record):
                continue

        records.append(record)
        if len(records) >= limit:
            break

    return records


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(data: List[Dict[str, Any]]) -> str:
    """Detect the schema format from the first few records (majority vote)."""
    if not data:
        return "unknown"
    counts: Dict[str, int] = {}
    for record in data[:min(5, len(data))]:
        keys = set(record.keys())
        if "messages" in keys:
            fmt = "chatml"
        elif "conversations" in keys:
            fmt = "sharegpt"
        elif "instruction" in keys and "output" in keys:
            fmt = "alpaca"
        elif "prompt" in keys and "completion" in keys:
            fmt = "prompt_completion"
        else:
            fmt = "generic"
        counts[fmt] = counts.get(fmt, 0) + 1
    return max(counts, key=counts.get)


def _extract_user_texts(data: List[Dict[str, Any]], limit: int = 50) -> List[str]:
    """Extract the primary user-facing text from each record, format-aware."""
    fmt = detect_format(data)
    texts: List[str] = []
    for item in data[:limit]:
        text = ""
        if fmt == "alpaca":
            text = item.get("instruction", "")
        elif fmt == "chatml":
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    text = msg.get("content", "")
                    break
        elif fmt == "prompt_completion":
            text = item.get("prompt", "")
        elif fmt == "sharegpt":
            for conv in item.get("conversations", []):
                if conv.get("from") == "human":
                    text = conv.get("value", "")
                    break
        else:
            for field in _TEXT_FIELDS:
                val = item.get(field, "")
                if isinstance(val, str) and val.strip():
                    text = val
                    break
        if text:
            texts.append(text.lower())
    return texts


def _extract_text_for_record(item: Dict[str, Any], fmt: str) -> str:
    """Return the primary user-facing text for a single record, format-aware."""
    text = ""
    if fmt == "alpaca":
        text = item.get("instruction", "")
    elif fmt == "chatml":
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                text = msg.get("content", "")
                break
    elif fmt == "prompt_completion":
        text = item.get("prompt", "")
    elif fmt == "sharegpt":
        for conv in item.get("conversations", []):
            if conv.get("from") == "human":
                text = conv.get("value", "")
                break
    else:
        for field in _TEXT_FIELDS:
            val = item.get(field)
            if isinstance(val, str) and val.strip():
                text = val
                break
    return text


def detect_domain(data: List[Dict[str, Any]]) -> Tuple[str, float]:
    """Detect the subject-matter domain of a dataset from content analysis.

    Returns ``(domain, confidence)`` where *domain* is one of:
    ``coding`` | ``translation`` | ``summarization`` | ``qa`` |
    ``classification`` | ``conversation`` | ``general``
    and *confidence* is a float in [0, 1].
    """
    if not data:
        return "unknown", 0.0

    fmt = detect_format(data)

    # Classification datasets are identified by the presence of a label field.
    if any(field in data[0] for field in _LABEL_FIELDS):
        return "classification", 0.9

    texts = _extract_user_texts(data)
    if not texts:
        return "unknown", 0.0

    joined = " ".join(texts)

    # Score each domain by keyword hit-rate.
    scores: Dict[str, float] = {
        domain: sum(1 for kw in kws if kw in joined) / len(kws)
        for domain, kws in _DOMAIN_KEYWORDS.items()
    }

    best = max(scores, key=scores.get)
    best_score = scores[best]

    if best_score < 0.04:
        # Not enough content signal — fall back on structural format.
        return ("conversation", 0.7) if fmt in ("chatml", "sharegpt") else ("general", 0.3)

    return best, round(min(best_score * 5, 1.0), 2)


def get_domain_coverage(data: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
    """Return domain-specific coverage statistics for the dataset.

    The returned dict shape varies by domain and is consumed by the
    ``analyse`` CLI command to surface what the dataset is lacking.
    """
    texts = _extract_user_texts(data, limit=100)
    if not texts:
        return {}

    joined = " ".join(texts)

    if domain == "coding":
        language_patterns: Dict[str, List[str]] = {
            "Python":     ["python", "def ", ".py", "import ", "pip "],
            "JavaScript": ["javascript", "const ", "let ", "=>", ".js", "node"],
            "SQL":        ["sql", "select ", "insert into", "update ", "from "],
            "Java":       ["java", "public class", "system.out", ".java"],
            "Bash/Shell": ["bash", "shell", "#!/bin/", "echo ", "grep "],
            "TypeScript": ["typescript", ": string", ": number", "interface "],
            "Rust":       ["rust", "fn ", "let mut", "println!"],
        }
        languages = [
            lang for lang, kws in language_patterns.items()
            if any(kw in joined for kw in kws)
        ]
        task_types = {
            "write / implement": sum(1 for t in texts if any(
                kw in t for kw in ["write", "implement", "create", "build", "generate"]
            )),
            "debug / fix": sum(1 for t in texts if any(
                kw in t for kw in ["debug", "fix", "bug", "error", "wrong", "mistake"]
            )),
            "explain / analyse": sum(1 for t in texts if any(
                kw in t for kw in ["explain", "what does", "how does", "analyse", "analyze"]
            )),
            "refactor / optimise": sum(1 for t in texts if any(
                kw in t for kw in ["refactor", "optimise", "optimize", "improve", "clean", "simplify"]
            )),
        }
        return {
            "languages_detected": languages,
            "language_diversity": len(languages),
            "task_type_counts": task_types,
            "has_edge_cases": any(kw in joined for kw in [
                "edge case", "empty list", "none", "null", "zero", "overflow", "boundary", "empty string",
            ]),
            "has_error_handling": any(kw in joined for kw in [
                "exception", "try", "except", "catch", "raise", "error handling", "valueerror",
            ]),
        }

    if domain == "qa":
        starters = {"what": 0, "who": 0, "how": 0, "why": 0, "when": 0, "where": 0}
        for t in texts:
            for s in starters:
                if t.startswith(s):
                    starters[s] += 1
                    break
        return {
            "question_type_distribution": {k: v for k, v in starters.items() if v > 0},
            "question_type_diversity": sum(1 for v in starters.values() if v > 0),
        }

    if domain == "translation":
        lang_signals: Dict[str, List[str]] = {
            "French":     ["french", "français", "bonjour", "le ", "la "],
            "Spanish":    ["spanish", "español", "hola", "buenos"],
            "German":     ["german", "deutsch", "guten"],
            "Chinese":    ["chinese", "mandarin", "pinyin"],
            "Japanese":   ["japanese"],
            "Arabic":     ["arabic"],
            "Portuguese": ["portuguese"],
        }
        langs = [lang for lang, kws in lang_signals.items() if any(kw in joined for kw in kws)]
        return {
            "target_languages_detected": langs,
            "language_pair_count": len(langs),
        }

    return {}


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
        score = ratio * 0.65          # tightened from 0.85
        return True, f"Most records consistent ({ratio*100:.1f}%)", score, details
    else:
        score = ratio * 0.35          # tightened from 0.50
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

    # Exclude fields that are structurally optional for the detected format so
    # that e.g. an empty Alpaca `input` does not produce a false-positive failure.
    fmt = detect_format(data)
    optional_fields = _FORMAT_OPTIONAL_FIELDS.get(fmt, set())

    total_fields = 0
    missing_count = 0
    affected_rows: List[Dict] = []

    for i, item in enumerate(data):
        row_missing = []
        for key, value in item.items():
            if key in optional_fields:
                continue
            total_fields += 1
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_count += 1
                row_missing.append(key)
            elif key == "messages" and isinstance(value, list):
                # ChatML: flag turns with empty content
                for turn in value:
                    if isinstance(turn, dict):
                        content = turn.get("content")
                        if content is None or (isinstance(content, str) and not content.strip()):
                            missing_count += 1
                            row_missing.append(f"messages[].content (role={turn.get('role', '?')})")
            elif key == "conversations" and isinstance(value, list):
                # ShareGPT: flag turns with empty value
                for turn in value:
                    if isinstance(turn, dict):
                        val = turn.get("value")
                        if val is None or (isinstance(val, str) and not val.strip()):
                            missing_count += 1
                            row_missing.append(f"conversations[].value (from={turn.get('from', '?')})")
        if row_missing:
            affected_rows.append({"row": i + 1, "empty_fields": row_missing})

    completeness = 1 - (missing_count / total_fields) if total_fields > 0 else 0.0

    optional_note = (
        f" ('{', '.join(sorted(optional_fields))}' excluded as optional)"
        if optional_fields else ""
    )

    details = {
        "total_fields_checked": total_fields,
        "empty_field_count": missing_count,
        "total_affected_rows": len(affected_rows),
        "affected_rows": affected_rows[:20],
        "skipped_optional_fields": sorted(optional_fields),
    }

    if completeness >= pass_t:
        return True, f"High completeness ({completeness*100:.1f}%){optional_note}", completeness, details
    elif completeness >= soft_t:
        score = completeness * 0.65          # tightened from 0.85
        return True, f"Acceptable completeness ({completeness*100:.1f}%){optional_note}", score, details
    else:
        score = completeness * 0.35          # tightened from 0.50
        return False, f"Low completeness ({completeness*100:.1f}%){optional_note}", score, details


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
        score = uniqueness * 0.50          # tightened from 0.80 — even a few dups hurt training
        return True, f"Few duplicates ({len(dup_rows)} found)", score, details
    else:
        score = uniqueness * 0.25          # tightened from 0.40
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

    # Build word-sets using format-aware text extraction
    fmt = detect_format(data)
    word_sets: List[set] = []
    for item in data[:check_n]:
        text = _extract_text_for_record(item, fmt)
        word_sets.append(set(text.lower().split()) if text else set())

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
            if sim < threshold:
                continue
            # At sim == 1.0 the primary text fields are identical. Only flag
            # this pair if the full records differ — byte-identical records are
            # exact duplicates already handled by check_duplicates and must not
            # be double-counted here.
            if sim == 1.0 and json.dumps(data[i], sort_keys=True) == json.dumps(data[j], sort_keys=True):
                continue
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
    if near_dup_ratio < 0.03:                        # tightened threshold from 0.05
        score = max(0.2, 1.0 - near_dup_ratio * 10)  # floor lowered from 0.5; multiplier raised from 5
        return True, f"Few near-duplicates ({n_pairs} {pair_word}){suffix}", score, details
    else:
        score = max(0.05, 1.0 - near_dup_ratio * 15) # floor lowered from 0.1; multiplier raised from 10
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
    score = max(0.0, 1.0 - outlier_ratio * 4)    # tightened from *2 — 10% outliers now scores 0.6

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
    elif outlier_ratio < 0.05:                    # tightened from 0.10
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

    if imbalance > 5:                                                         # tightened from >10
        return False, f"Severe label imbalance ({max_c}:{min_c} ratio)", 0.15, details  # score 0.3→0.15
    elif imbalance > 3:                                                       # tightened from >5
        return True, f"Moderate label imbalance ({len(unique_labels)} classes)", 0.50, details  # score 0.7→0.50
    elif imbalance > 1.5:
        return True, f"Slight label imbalance ({len(unique_labels)} classes)", 0.85, details
    return True, f"Balanced labels ({len(unique_labels)} classes)", 1.0, details


def _extract_output_text(item: Dict[str, Any]) -> str:
    """Return the output/completion/assistant text from a record, format-aware."""
    for field in ("output", "completion", "answer", "response"):  # added "response" for Dolly etc.
        val = item.get(field, "")
        if isinstance(val, str) and val.strip():
            return val.lower()
    # ChatML: last assistant message
    for msg in reversed(item.get("messages", [])):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.lower()
    # ShareGPT: last gpt/assistant turn
    for conv in reversed(item.get("conversations", [])):
        if conv.get("from") in ("gpt", "assistant"):
            val = conv.get("value", "")
            if isinstance(val, str) and val.strip():
                return val.lower()
    return ""


def _estimate_record_tokens(item: Dict[str, Any]) -> int:
    """Estimate token count for a record using word_count × 1.3 heuristic."""
    _ALL_FIELDS = ("instruction", "input", "output", "text", "prompt", "completion", "question", "answer")
    words = sum(len(str(item.get(f, "")).split()) for f in _ALL_FIELDS if item.get(f))
    for msg in item.get("messages", []):
        words += len(str(msg.get("content", "")).split())
    for conv in item.get("conversations", []):
        words += len(str(conv.get("value", "") or conv.get("text", "")).split())
    return int(words * 1.3)


def check_output_diversity(
    data: List[Dict[str, Any]], config: Dict = None
) -> Tuple[bool, str, float, Dict]:
    """Detect near-duplicate *outputs* using Jaccard similarity.

    Two records with different instructions but nearly identical outputs
    (e.g. the same generated code block repeated) waste training signal.
    Complements check_near_duplicates, which only inspects input fields.
    """
    if not data or len(data) < 2:
        return True, "Too few records to check output diversity", 1.0, {}

    cfg = (config or _default_config())["thresholds"]
    threshold = cfg.get("near_duplicate_similarity", 0.85)
    sample_size = int(cfg.get("near_duplicate_sample", 500))
    check_n = min(len(data), sample_size)

    output_sets: List[set] = []
    for item in data[:check_n]:
        text = _extract_output_text(item)
        output_sets.append(set(text.split()) if text else set())

    # If no output field is present at all, the check is not applicable
    if all(not s for s in output_sets):
        return True, "No output field found (check not applicable)", 1.0, {"total_similar_output_pairs": 0}

    similar_pairs: List[Dict] = []
    for i in range(check_n):
        if not output_sets[i]:
            continue
        for j in range(i + 1, check_n):
            if not output_sets[j]:
                continue
            union = output_sets[i] | output_sets[j]
            if not union:
                continue
            sim = len(output_sets[i] & output_sets[j]) / len(union)
            if sim < threshold:
                continue
            if sim == 1.0 and json.dumps(data[i], sort_keys=True) == json.dumps(data[j], sort_keys=True):
                continue  # exact duplicate — already handled by check_duplicates
            similar_pairs.append({"row1": i + 1, "row2": j + 1, "similarity": round(sim, 3)})

    total_pairs = check_n * (check_n - 1) / 2
    dup_ratio = len(similar_pairs) / total_pairs if total_pairs > 0 else 0.0

    details = {
        "similarity_threshold": threshold,
        "records_checked": check_n,
        "total_similar_output_pairs": len(similar_pairs),
        "similar_output_pairs": similar_pairs[:20],
    }

    if not similar_pairs:
        return True, "Output fields are diverse (no near-duplicate outputs detected)", 1.0, details
    n = len(similar_pairs)
    pair_word = "pair" if n == 1 else "pairs"
    if dup_ratio < 0.03:                                                              # tightened from 0.05
        return True, f"Few similar outputs ({n} {pair_word})", max(0.2, 1.0 - dup_ratio * 10), details   # floor 0.5→0.2
    return False, f"Many similar outputs ({n} {pair_word})", max(0.05, 1.0 - dup_ratio * 15), details    # floor 0.1→0.05


def check_token_length(
    data: List[Dict[str, Any]], config: Dict = None
) -> Tuple[bool, str, float, Dict]:
    """Flag records whose estimated token count exceeds the context-window limit.

    Uses a word_count × 1.3 heuristic (average English word ≈ 1.3 tokens).
    All text fields in a record are summed so the full prompt+response is checked.
    """
    if not data:
        return False, "Empty dataset", 0.0, {}

    cfg = (config or _default_config())["thresholds"]
    max_tokens = int(cfg.get("max_token_estimate", 2048))

    estimates: List[int] = []
    overflow_rows: List[Dict] = []
    for i, item in enumerate(data):
        est = _estimate_record_tokens(item)
        estimates.append(est)
        if est > max_tokens:
            overflow_rows.append({"row": i + 1, "estimated_tokens": est})

    avg_tokens = sum(estimates) / len(estimates) if estimates else 0.0
    overflow_ratio = len(overflow_rows) / len(data)
    score = max(0.0, 1.0 - overflow_ratio * 4)    # tightened from *2

    details = {
        "max_token_estimate": max_tokens,
        "avg_estimated_tokens": round(avg_tokens),
        "total_overflow_rows": len(overflow_rows),
        "overflow_rows": overflow_rows[:20],
    }

    if not overflow_rows:
        return True, f"No token overflows (avg {avg_tokens:.0f} est. tokens, limit {max_tokens})", 1.0, details
    n = len(overflow_rows)
    rec_word = "record" if n == 1 else "records"
    if overflow_ratio < 0.05:
        return True, f"Few token overflows ({n} {rec_word} exceed ~{max_tokens} tokens)", score, details
    return False, f"Token overflows detected ({n} {rec_word} exceed ~{max_tokens} tokens)", score, details


# Patterns that signal vague, under-specified instructions
_VAGUE_PATTERNS: List[str] = [
    "help me", "do something", "write something", "tell me something",
    "please help", "make something", "i need help", "can you do",
    "do this for me", "just do it",
]

# Markers that suggest an instruction bundles multiple unrelated tasks
_MULTI_TASK_MARKERS: List[str] = [
    " and also ", " additionally, ", " furthermore, ", " also please ",
    " as well as this", " on top of that", " at the same time ",
]


def check_instruction_quality(
    data: List[Dict[str, Any]], config: Dict = None
) -> Tuple[bool, str, float, Dict]:
    """Heuristically flag low-quality instructions.

    Checks for:
    - Vague instructions that match known low-signal patterns ("help me", etc.)
    - Multi-task instructions that bundle unrelated requests in one record
    - Instructions shorter than ``min_instruction_words``

    Only applied to datasets with an explicit ``instruction`` field (Alpaca format).
    Generic or ChatML datasets return a pass so quality scores are not polluted
    by false positives on short questions or conversation turns.
    """
    if not data:
        return False, "Empty dataset", 0.0, {}

    fmt = detect_format(data)
    if fmt != "alpaca":
        return True, "Instruction quality check applies to Alpaca format only (skipped)", 1.0, {}

    cfg = (config or _default_config())["thresholds"]
    min_words = int(cfg.get("min_instruction_words", 4))

    vague_rows: List[Dict] = []
    multi_task_rows: List[Dict] = []
    short_rows: List[Dict] = []

    for i, item in enumerate(data):
        instruction = item.get("instruction", "")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        text = instruction.lower()
        word_count = len(text.split())

        if word_count < min_words:
            short_rows.append({"row": i + 1, "words": word_count, "text": instruction[:80]})
        # Removed word_count < 8 guard — vague patterns are bad regardless of length
        if any(pattern in text for pattern in _VAGUE_PATTERNS):
            vague_rows.append({"row": i + 1, "text": instruction[:80]})
        if any(marker in text for marker in _MULTI_TASK_MARKERS):
            multi_task_rows.append({"row": i + 1, "text": instruction[:80]})

    flagged_set = (
        {r["row"] for r in vague_rows}
        | {r["row"] for r in multi_task_rows}
        | {r["row"] for r in short_rows}
    )
    total_flagged = len(flagged_set)
    flag_ratio = total_flagged / len(data)
    score = max(0.0, 1.0 - flag_ratio * 3)    # tightened from *2

    details = {
        "total_flagged": total_flagged,
        "vague_rows": vague_rows[:10],
        "multi_task_rows": multi_task_rows[:10],
        "short_instruction_rows": short_rows[:10],
        "min_instruction_words": min_words,
    }

    if total_flagged == 0:
        return True, "Instruction quality looks good (no vague, multi-task, or too-short instructions found)", 1.0, details
    n_word = "record" if total_flagged == 1 else "records"
    if flag_ratio < 0.10:
        return True, f"Minor instruction quality issues ({total_flagged} {n_word} flagged)", score, details
    return False, f"Instruction quality issues detected ({total_flagged} {n_word} flagged)", score, details


def check_language_consistency(
    data: List[Dict[str, Any]], config: Dict = None
) -> Tuple[bool, str, float, Dict]:
    """Detect unexpected language switches using a non-ASCII character ratio heuristic.

    No external libraries required. Computes the non-ASCII ratio for each record's
    primary text, determines the dataset's dominant script (ASCII-heavy = Latin/code,
    non-ASCII-heavy = CJK/Arabic/etc.), then flags records that deviate significantly.
    """
    if not data:
        return False, "Empty dataset", 0.0, {}

    def non_ascii_ratio(text: str) -> float:
        return sum(1 for c in text if ord(c) > 127) / len(text) if text else 0.0

    ratios: List[Tuple[int, float]] = []
    for i, item in enumerate(data):
        text = ""
        for field in _TEXT_FIELDS:
            val = item.get(field, "")
            if isinstance(val, str) and val.strip():
                text = val
                break
        if not text and "messages" in item:
            for msg in item.get("messages", []):
                content = msg.get("content", "")
                if content:
                    text = content
                    break
        if not text and "conversations" in item:
            for conv in item.get("conversations", []):
                val = conv.get("value", "")
                if val:
                    text = val
                    break
        if text:
            ratios.append((i + 1, non_ascii_ratio(text)))

    if not ratios:
        return True, "No text content to check language consistency", 1.0, {}

    avg_ratio = sum(r for _, r in ratios) / len(ratios)
    dominant = "non-ASCII" if avg_ratio > 0.40 else "ASCII"

    # Flag records that deviate from the dominant script
    if dominant == "ASCII":
        anomalous = [(row, r) for row, r in ratios if r > 0.30]
    else:
        anomalous = [(row, r) for row, r in ratios if r < 0.05]

    anomaly_rows = [{"row": row, "non_ascii_ratio": round(r, 3)} for row, r in anomalous]
    anomaly_ratio = len(anomaly_rows) / len(data)
    score = max(0.0, 1.0 - anomaly_ratio * 3)

    details = {
        "avg_non_ascii_ratio": round(avg_ratio, 3),
        "dominant_script": dominant,
        "total_anomalous_rows": len(anomaly_rows),
        "anomalous_rows": anomaly_rows[:20],
    }

    if not anomaly_rows:
        return True, f"Language is consistent throughout (dominant: {dominant})", 1.0, details
    n = len(anomaly_rows)
    n_word = "record" if n == 1 else "records"
    if anomaly_ratio < 0.05:
        return True, f"Minor language inconsistencies ({n} {n_word} may switch script)", score, details
    return False, f"Language inconsistencies detected ({n} {n_word} appear to switch script)", score, details


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
        ("json_format",          lambda: check_json_format(data)),
        ("field_consistency",    lambda: check_field_consistency(data, cfg)),
        ("missing_values",       lambda: check_missing_values(data, cfg)),
        ("duplicates",           lambda: check_duplicates(data, cfg)),
        ("near_duplicates",      lambda: check_near_duplicates(data, cfg)),
        ("output_diversity",     lambda: check_output_diversity(data, cfg)),
        ("text_length",          lambda: check_text_length(data, cfg)),
        ("token_length",         lambda: check_token_length(data, cfg)),
        ("instruction_quality",  lambda: check_instruction_quality(data, cfg)),
        ("language_consistency", lambda: check_language_consistency(data, cfg)),
        ("label_quality",        lambda: check_label_quality(data, cfg)),
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
