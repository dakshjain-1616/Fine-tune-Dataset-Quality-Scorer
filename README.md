# Fine-tune Dataset Quality Scorer

> Built autonomously by [NEO](https://heyneo.com) — your fully autonomous AI coding agent. &nbsp; [![NEO for VS Code](https://img.shields.io/badge/VS%20Code-NEO%20Extension-5C2D91?logo=visual-studio-code&logoColor=white)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)

![Architecture](./infographic.svg)

A CLI tool that analyses JSONL fine-tuning datasets and tells you — before you burn GPU hours — whether your data is actually worth training on.

---

## Why This Exists

Fine-tuning a model costs money and time. Bad training data silently produces bad models. Most teams discover data quality problems only after training, when:

- The model hallucinates more than the base model
- It learned from duplicate or near-duplicate examples (wasted compute)
- Short or malformed records caused silent truncation
- Field inconsistencies corrupted the training signal

This tool runs those checks in seconds and gives a **score out of 100** with specific, actionable fixes.

---

## What It Checks

| Check | What it detects |
|-------|----------------|
| **Format validation** | All records are valid JSON dicts; detects schema (Alpaca / ChatML / etc.) |
| **Field consistency** | Every record has the same set of fields; reports which rows are missing which fields |
| **Missing values** | `null` or empty-string field values; reports affected rows |
| **Exact duplicates** | Identical records; reports which rows are duplicates of which |
| **Near-duplicates** | High Jaccard word-set similarity (≥ 0.85 by default); reports similar pairs |
| **Text length** | Primary text field word-count outliers (too short or too long) |
| **Label balance** | Class imbalance for classification datasets (imbalance ratio > 10 = fail) |

---

## Score Interpretation

```
90 – 100   READY       High quality — proceed with fine-tuning
75 –  89   CAUTION     Minor issues — review flagged checks first
50 –  74   NEEDS WORK  Significant problems — do not train yet
 0 –  49   NOT READY   Critical issues found — fix before training
```

---

## Supported JSONL Formats

Format is auto-detected from the first record — no flags needed:

| Format | Key fields |
|--------|-----------|
| Alpaca | `instruction`, `input`, `output` |
| ChatML | `messages` (array with `role`/`content`) |
| Prompt/Completion | `prompt`, `completion` |
| ShareGPT | `conversations` (array with `from`/`value`) |
| Generic | any other field combination |

---

## Quick Start

### Install

```bash
pip install typer rich pyyaml
```

### Score a dataset

```bash
python3 src/main.py score your_dataset.jsonl
```

Sample output:

```
Overall Score: 100.0/100 — Grade: READY
Records analysed : 10
Detected format  : generic

 Check               Status  Score  Weight  Details
 Json Format          PASS   100.0    10%  Valid JSON format (detected: generic)
 Field Consistency    PASS   100.0    20%  All records have consistent fields
 Missing Values       PASS   100.0    20%  High completeness (100.0%)
 Duplicates           PASS   100.0    15%  No exact duplicates found
 Near Duplicates      PASS   100.0    10%  No near-duplicates found
 Text Length          PASS   100.0    10%  Good length distribution (avg 4.6 words)
 Label Quality        PASS   100.0    15%  Balanced labels (2 classes)
```

---

## All Commands

### `score` — full quality report

```bash
python3 src/main.py score your_dataset.jsonl
python3 src/main.py score your_dataset.jsonl --format json
python3 src/main.py score your_dataset.jsonl --format json  --output report.json
python3 src/main.py score your_dataset.jsonl --format html  --output report.html
```

Options:

| Flag | Description |
|------|-------------|
| `--format` / `-f` | `terminal` (default), `json`, or `html` |
| `--output` / `-o` | Save JSON or HTML report to a file |
| `--min-score` | Exit code 1 if score is below this threshold (CI/CD gate) |
| `--config` / `-c` | Path to a custom `config.yaml` |

---

### `quick` — bare score for scripts

Returns only the number — nothing else on stdout:

```bash
python3 src/main.py quick your_dataset.jsonl
# → 87.4
```

---

### `fix` — actionable suggestions with row numbers

```bash
python3 src/main.py fix your_dataset.jsonl
```

Sample output for a dataset with issues:

```
  1. Field Consistency — 3 records are missing fields (label, text). Affected rows: 1, 2, 9
  2. Exact Duplicates  — 2 duplicate records found. Remove: row 5 (dup of row 4), row 6 (dup of row 4)
  3. Text Length       — 5 outlier records (avg 1.4 words). too short: rows 1, 4, 5, 6, 10
```

---

### `compare` — side-by-side comparison

```bash
python3 src/main.py compare train.jsonl test.jsonl
```

Prints a full report for each dataset, then shows which scores higher and by how much.

---

### `stats` — field and label statistics

```bash
python3 src/main.py stats your_dataset.jsonl
```

Shows field presence counts, non-empty counts, and label distribution.

---

## CI/CD Integration

### Option A — built-in `--min-score` gate (simplest)

```bash
python3 src/main.py score data/train.jsonl --min-score 75
# Exits 0 if score ≥ 75, exits 1 if below — ready to use in any CI pipeline
```

### Option B — parse the JSON report

```bash
# Step 1: generate the report
python3 src/main.py score data/train.jsonl --format json --output /tmp/report.json

# Step 2: fail the build if below threshold
python3 -c "
import json, sys
r = json.load(open('/tmp/report.json'))
print(f\"Score: {r['overall_score']} ({r['grade']})\")
sys.exit(0 if r['overall_score'] >= 75 else 1)
"
```

> **Note:** When `--format json` is used without `--output`, JSON goes to stdout and all progress messages go to stderr — so the output can be safely piped to `jq` or any JSON parser.

---

## Configuration

Edit `config.yaml` to customise check weights and thresholds:

```yaml
# Score weights (must sum to 1.0)
weights:
  json_format: 0.10
  field_consistency: 0.20
  missing_values: 0.20
  duplicates: 0.15
  near_duplicates: 0.10
  text_length: 0.10
  label_quality: 0.15

# Thresholds
thresholds:
  min_text_words: 3           # below this → too-short outlier
  max_text_words: 2000        # above this → too-long outlier
  near_duplicate_similarity: 0.85   # Jaccard threshold (0–1)
  near_duplicate_sample: 500        # max records compared (performance cap)
  field_consistency_pass: 0.95
  duplicate_soft: 0.90

# Grade cutoffs
score_bands:
  ready: 90
  caution: 75
  needs_work: 50
```

Pass a custom config with `--config path/to/config.yaml`.

---

## Project Structure

```
finetune-dataset-quality-scorer/
├── src/
│   ├── __init__.py
│   ├── checks.py      # All quality checks + weighted scoring logic
│   ├── main.py        # Typer CLI — 5 commands
│   └── reporter.py    # Terminal (Rich), JSON, and HTML report generators
├── tests/
│   ├── conftest.py
│   ├── test_checks.py         # 48 tests
│   └── fixtures/
│       ├── good_dataset.jsonl  # 10 clean records → score 100
│       ├── bad_dataset.jsonl   # duplicates, missing fields, empty values → score ~61
│       └── mixed_dataset.jsonl # some issues → score ~87
├── config.yaml
└── requirements.txt
```

---

## Running Tests

```bash
python3 -m pytest tests/ -v
```

48 tests covering all checks, scoring logic, weighted thresholds, format detection, and CLI behaviour. No GPU or external service required.

---

## License

MIT
