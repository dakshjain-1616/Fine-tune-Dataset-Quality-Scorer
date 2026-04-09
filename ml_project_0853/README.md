# Fine-tune Dataset Quality Scorer

> Built autonomously by [NEO](https://heyneo.com) — your fully autonomous AI coding agent. &nbsp; [![NEO for VS Code](https://img.shields.io/badge/VS%20Code-NEO%20Extension-5C2D91?logo=visual-studio-code&logoColor=white)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)

![Architecture](./infographic.svg)

A CLI tool that analyzes JSONL fine-tuning datasets and tells you — before you burn GPU hours — whether your data is actually worth training on.

---

## Why This Exists

Fine-tuning a model costs money and time. Bad training data silently produces bad models. Most teams discover data quality problems only after training, when:

- The model hallucinates more than the base model
- It's sycophantic or repetitive (bad response quality)
- It learned from duplicate examples (wasted compute)
- Token length distribution caused silent truncation

This tool runs those checks in seconds and gives a **score out of 100** with specific, actionable fixes.

---

## What It Does

```
your_dataset.jsonl  ──►  quality checks  ──►  score + report
```

Checks cover:
- **Format validation** — are all records valid JSON with the right fields?
- **Deduplication** — exact and near-duplicate detection
- **Length distribution** — finds outliers that will get silently truncated
- **Instruction complexity** — flags trivially easy or impossibly complex examples
- **Response quality** — detects sycophantic prefixes, truncated responses, LLM fingerprints
- **Semantic diversity** — are all your examples too similar to each other?
- **Language consistency** — are instruction and response in the same language?

---

## Score Interpretation

```
90 – 100   READY TO TRAIN      High quality, proceed
75 – 89    TRAIN WITH CAUTION  Minor issues, review top fixes first
50 – 74    NEEDS WORK          Significant problems, do not train yet
 0 – 49    DO NOT TRAIN        Critical issues found
```

---

## Who Should Use This

| Role | How it helps |
|------|-------------|
| ML Engineers | Gate fine-tuning pipelines — don't train on bad data |
| Data Annotators | Validate exports before handing off to training |
| Researchers | Understand dataset composition before experiments |
| CI/CD Pipelines | Fail builds automatically if dataset quality drops |

---

## Supported JSONL Formats

The tool auto-detects format — no flags needed:

| Format | Example fields |
|--------|---------------|
| Alpaca | `instruction`, `input`, `output` |
| ChatML | `messages` array with `role`/`content` |
| Prompt/Completion | `prompt`, `completion` |
| ShareGPT | `conversations` array with `from`/`value` |

---

## Quick Start

### Install

```bash
pip install typer rich
```

### Score a dataset

```bash
python src/main.py score your_dataset.jsonl
```

### Save report to file

```bash
python src/main.py score your_dataset.jsonl --format json --output report.json
python src/main.py score your_dataset.jsonl --format html --output report.html
```

---

## Output Formats

**Terminal** — color-coded score, component breakdown table, top recommended fixes. Instant feedback.

**JSON** — machine-readable report with per-check scores, flagged row numbers, and aggregate metrics. Use in CI/CD to fail builds below a threshold.

**HTML** — interactive visual report with gauge chart for overall score, histograms, and a filterable table of flagged samples.

---

## CI/CD Integration

Add a quality gate to your training pipeline:

```bash
python src/main.py score data/train.jsonl --format json --output /tmp/report.json
python -c "import json; r=json.load(open('/tmp/report.json')); exit(0 if r['score'] >= 75 else 1)"
```

If the dataset drops below 75, the pipeline fails before wasting GPU time.

---

## Project Structure

```
ml_project_0853/
├── src/
│   ├── __init__.py
│   ├── checks.py      # All quality checks + scoring logic
│   ├── main.py        # Typer CLI (score command)
│   └── reporter.py    # Terminal, JSON, and HTML report generators
├── tests/
│   ├── test_checks.py
│   └── fixtures/      # good, bad, and mixed dataset samples
├── config.yaml
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v
```

18 tests covering check logic, scoring thresholds, and format validation. No GPU or external service needed.

---

## License

MIT
