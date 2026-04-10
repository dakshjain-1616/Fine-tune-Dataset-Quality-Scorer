"""Main CLI entry point for Dataset Quality Scorer."""

import json
import sys
from pathlib import Path
from typing import Optional

# Ensure the project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.checks import (
    load_dataset, load_config, calculate_quality_score,
    detect_domain, get_domain_coverage,
    _DOMAIN_SUGGESTIONS,
)
from src.reporter import generate_terminal_report, generate_json_report, generate_html_report

console = Console()
# Status/progress messages go to stderr so stdout stays clean for piped JSON
_err = Console(stderr=True)

app = typer.Typer(
    help="Fine-tune Dataset Quality Scorer — analyse a JSONL dataset before training.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

@app.command("score")
def score_dataset(
    filepath: str = typer.Argument(..., help="Path to JSONL dataset file"),
    output_format: str = typer.Option(
        "terminal", "--format", "-f",
        help="Output format: terminal | json | html",
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save report to this file (json/html formats)"
    ),
    min_score: Optional[float] = typer.Option(
        None, "--min-score",
        help="Exit with code 1 if overall score is below this threshold (CI/CD gate)",
    ),
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to a custom config YAML file"
    ),
):
    """Score a dataset and generate a detailed quality report."""
    try:
        config = load_config(config_file)
        _err.print(f"Loading dataset: [cyan]{filepath}[/cyan]")
        data = load_dataset(filepath)
        score_result = calculate_quality_score(data, config)

        if output_format == "terminal":
            generate_terminal_report(score_result, filepath)

        elif output_format == "json":
            report = generate_json_report(score_result, filepath)
            if output_file:
                Path(output_file).write_text(report, encoding="utf-8")
                _err.print(f"JSON report saved to: [cyan]{output_file}[/cyan]")
            else:
                # Use plain print so piped consumers get clean JSON on stdout
                print(report)

        elif output_format == "html":
            report = generate_html_report(score_result, filepath)
            if output_file:
                Path(output_file).write_text(report, encoding="utf-8")
                _err.print(f"HTML report saved to: [cyan]{output_file}[/cyan]")
            else:
                _err.print("[yellow]HTML report generated — use --output <file> to save it.[/yellow]")

        else:
            _err.print(f"[red]Unknown format '{output_format}'. Use: terminal | json | html[/red]")
            raise typer.Exit(code=1)

        # CI/CD gate
        if min_score is not None and score_result["overall_score"] < min_score:
            _err.print(
                f"[red]Score {score_result['overall_score']:.1f} is below the required "
                f"minimum of {min_score}. Failing.[/red]"
            )
            raise typer.Exit(code=1)

        return score_result

    except FileNotFoundError:
        _err.print(f"[red]Error: File not found: {filepath}[/red]")
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        _err.print(f"[red]Error: Invalid JSON in dataset: {e}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# quick
# ---------------------------------------------------------------------------

@app.command("quick")
def quick_score(
    filepath: str = typer.Argument(..., help="Path to JSONL dataset file"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Print only the overall score (suitable for CI/CD scripts)."""
    try:
        config = load_config(config_file)
        data = load_dataset(filepath)
        result = calculate_quality_score(data, config)
        # Plain print — no Rich so shell scripts capture a bare float
        print(f"{result['overall_score']:.1f}")
        return result["overall_score"]
    except Exception as e:
        _err.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

@app.command("compare")
def compare_datasets(
    filepath1: str = typer.Argument(..., help="Path to the first JSONL dataset"),
    filepath2: str = typer.Argument(..., help="Path to the second JSONL dataset"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Compare quality scores between two datasets side-by-side."""
    try:
        config = load_config(config_file)

        console.print(f"\n[bold]Dataset 1:[/bold] {filepath1}")
        data1 = load_dataset(filepath1)
        score1 = calculate_quality_score(data1, config)
        generate_terminal_report(score1, filepath1)

        console.print(f"\n[bold]Dataset 2:[/bold] {filepath2}")
        data2 = load_dataset(filepath2)
        score2 = calculate_quality_score(data2, config)
        generate_terminal_report(score2, filepath2)

        console.print("\n[bold]Comparison[/bold]")
        diff = score1["overall_score"] - score2["overall_score"]
        if diff > 0:
            console.print(f"[green]Dataset 1 scores higher by {diff:.1f} points[/green]")
        elif diff < 0:
            console.print(f"[green]Dataset 2 scores higher by {abs(diff):.1f} points[/green]")
        else:
            console.print("[yellow]Both datasets have equal quality scores[/yellow]")

        # Per-check breakdown table
        tbl = Table(title="Check-by-check breakdown", show_lines=True)
        tbl.add_column("Check", style="cyan", no_wrap=True)
        tbl.add_column(Path(filepath1).name, justify="right")
        tbl.add_column(Path(filepath2).name, justify="right")
        tbl.add_column("Δ (2 vs 1)", justify="right")
        for check_name in score1["checks"]:
            s1 = score1["checks"][check_name]["score"] * 100
            s2 = score2["checks"][check_name]["score"] * 100
            delta = s2 - s1
            if delta > 0:
                delta_str = f"[green]+{delta:.1f}[/green]"
            elif delta < 0:
                delta_str = f"[red]{delta:.1f}[/red]"
            else:
                delta_str = "[dim]=[/dim]"
            tbl.add_row(
                check_name.replace("_", " ").title(),
                f"{s1:.1f}",
                f"{s2:.1f}",
                delta_str,
            )
        console.print(tbl)

        return score1, score2

    except Exception as e:
        _err.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# fix
# ---------------------------------------------------------------------------

@app.command("fix")
def fix_suggestions(
    filepath: str = typer.Argument(..., help="Path to JSONL dataset file"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Print actionable suggestions for fixing dataset quality issues."""
    try:
        config = load_config(config_file)
        data = load_dataset(filepath)
        result = calculate_quality_score(data, config)
        checks = result["checks"]

        console.print(Panel(
            f"Fix Suggestions for: [cyan]{filepath}[/cyan]  "
            f"(score: {result['overall_score']:.1f}/100, grade: {result['grade']})",
            title="Quality Fix Recommendations",
        ))

        suggestions = []

        # Field consistency
        fc = checks["field_consistency"]
        if not fc["passed"]:
            rows = [str(r["row"]) for r in fc["details"].get("incomplete_rows", [])]
            missing = set()
            for r in fc["details"].get("incomplete_rows", []):
                missing.update(r.get("missing_fields", []))
            total_incomplete = fc["details"].get("total_incomplete", len(rows))
            remainder = total_incomplete - min(10, len(rows))
            row_str = ", ".join(rows[:10]) + (f" … and {remainder} more" if remainder > 0 else "")
            field_str = ", ".join(sorted(missing))
            suggestions.append(
                f"[bold yellow]Field Consistency[/bold yellow] — "
                f"{fc['details'].get('total_incomplete', '?')} records are missing fields "
                f"({field_str}). Affected rows: {row_str}"
            )

        # Missing values
        mv = checks["missing_values"]
        if not mv["passed"]:
            rows = [str(r["row"]) for r in mv["details"].get("affected_rows", [])]
            total_affected = mv["details"].get("total_affected_rows", len(rows))
            remainder = total_affected - min(10, len(rows))
            row_str = ", ".join(rows[:10]) + (f" … and {remainder} more" if remainder > 0 else "")
            suggestions.append(
                f"[bold yellow]Missing Values[/bold yellow] — "
                f"{mv['details'].get('empty_field_count', '?')} empty/null field values "
                f"across {mv['details'].get('total_affected_rows', '?')} records. "
                f"Affected rows: {row_str}"
            )

        # Exact duplicates
        dup = checks["duplicates"]
        if not dup["passed"]:
            dup_rows = dup["details"].get("duplicate_rows", [])
            dup_info = [
                f"row {r['row']} (dup of row {r['duplicate_of_row']})"
                for r in dup_rows[:10]
            ]
            total_dups = dup["details"].get("duplicate_count", len(dup_rows))
            remainder = total_dups - len(dup_info)
            dup_str = ", ".join(dup_info) + (f" … and {remainder} more" if remainder > 0 else "")
            suggestions.append(
                f"[bold yellow]Exact Duplicates[/bold yellow] — "
                f"{dup['details'].get('duplicate_count', '?')} duplicate records found. "
                f"Remove: {dup_str}"
            )

        # Near duplicates
        nd = checks["near_duplicates"]
        if not nd["passed"]:
            nd_pairs = nd["details"].get("near_duplicate_pairs", [])
            pair_info = [
                f"rows {p['row1']}&{p['row2']} (sim {p['similarity']})"
                for p in nd_pairs[:5]
            ]
            total_nd = nd["details"].get("total_near_duplicate_pairs", len(nd_pairs))
            remainder = total_nd - len(pair_info)
            pair_str = ", ".join(pair_info) + (f" … and {remainder} more pairs" if remainder > 0 else "")
            suggestions.append(
                f"[bold yellow]Near-Duplicates[/bold yellow] — "
                f"{nd['details'].get('total_near_duplicate_pairs', '?')} near-duplicate pairs "
                f"(Jaccard ≥ {nd['details'].get('similarity_threshold', 0.85)}). "
                f"Review: {pair_str}"
            )

        # Text length
        tl = checks["text_length"]
        if not tl["passed"]:
            short_rows = tl["details"].get("too_short_rows", [])
            long_rows = tl["details"].get("too_long_rows", [])
            short = [str(r["row"]) for r in short_rows[:5]]
            long_ = [str(r["row"]) for r in long_rows[:5]]
            n_out = tl["details"].get("total_outliers", "?")
            rec_w = "record" if n_out == 1 else "records"
            parts = []
            if short:
                short_more = len(short_rows) - len(short)
                s_str = f"too short: rows {', '.join(short)}"
                if short_more > 0:
                    s_str += f" … and {short_more} more"
                parts.append(s_str)
            if long_:
                long_more = len(long_rows) - len(long_)
                l_str = f"too long: rows {', '.join(long_)}"
                if long_more > 0:
                    l_str += f" … and {long_more} more"
                parts.append(l_str)
            suggestions.append(
                f"[bold yellow]Text Length[/bold yellow] — "
                f"{n_out} outlier {rec_w} "
                f"(avg {tl['details'].get('avg_word_count', '?')} words). "
                + "; ".join(parts)
            )

        # Label quality
        lq = checks["label_quality"]
        if not lq["passed"]:
            dist = lq["details"].get("label_distribution", {})
            dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
            suggestions.append(
                f"[bold yellow]Label Imbalance[/bold yellow] — "
                f"imbalance ratio {lq['details'].get('imbalance_ratio', '?')}:1. "
                f"Distribution: {dist_str}. Consider oversampling minority classes."
            )

        if suggestions:
            for i, s in enumerate(suggestions, 1):
                console.print(f"\n  {i}. {s}")
        else:
            console.print(
                "\n  [green]No critical issues found — dataset looks ready for fine-tuning.[/green]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# analyse helpers
# ---------------------------------------------------------------------------

def _rich_color(score: float) -> str:
    if score >= 90:
        return "green"
    elif score >= 75:
        return "yellow"
    elif score >= 50:
        return "dark_orange"
    return "red"


def _build_findings(name: str, check: dict, config: dict) -> list[str]:
    """Return row-level finding strings for one check result."""
    details = check.get("details", {})
    findings = []

    if name == "duplicates":
        for row in details.get("duplicate_rows", []):
            findings.append(
                f"Row {row['row']} is an exact copy of row {row['duplicate_of_row']} — remove it"
            )
        remainder = details.get("duplicate_count", 0) - len(details.get("duplicate_rows", []))
        if remainder > 0:
            findings.append(f"… and {remainder} more (run 'fix' for the full list)")

    elif name == "near_duplicates":
        for pair in details.get("near_duplicate_pairs", []):
            findings.append(
                f"Rows {pair['row1']} & {pair['row2']}: similarity {pair['similarity']}"
                " — remove or rephrase the weaker example"
            )
        remainder = (
            details.get("total_near_duplicate_pairs", 0)
            - len(details.get("near_duplicate_pairs", []))
        )
        if remainder > 0:
            findings.append(f"… and {remainder} more pairs")

    elif name == "output_diversity":
        for pair in details.get("similar_output_pairs", []):
            findings.append(
                f"Rows {pair['row1']} & {pair['row2']}: output similarity {pair['similarity']}"
                " — rewrite or remove the redundant response"
            )
        remainder = (
            details.get("total_similar_output_pairs", 0)
            - len(details.get("similar_output_pairs", []))
        )
        if remainder > 0:
            findings.append(f"… and {remainder} more pairs")

    elif name == "token_length":
        max_t = details.get("max_token_estimate", 2048)
        for row in details.get("overflow_rows", []):
            findings.append(
                f"Row {row['row']}: ~{row['estimated_tokens']} estimated tokens"
                f" — exceeds {max_t}-token limit; trim instruction or output"
            )
        remainder = details.get("total_overflow_rows", 0) - len(details.get("overflow_rows", []))
        if remainder > 0:
            findings.append(f"… and {remainder} more rows")

    elif name == "instruction_quality":
        for row in details.get("vague_rows", []):
            findings.append(f"Row {row['row']}: vague instruction — \"{row['text']}\"")
        for row in details.get("multi_task_rows", []):
            findings.append(f"Row {row['row']}: multi-task instruction — \"{row['text']}\"")
        for row in details.get("short_instruction_rows", []):
            findings.append(
                f"Row {row['row']}: only {row['words']} word(s) — \"{row['text']}\""
                f" (minimum {details.get('min_instruction_words', 4)} words)"
            )

    elif name == "language_consistency":
        dominant = details.get("dominant_script", "ASCII")
        for row in details.get("anomalous_rows", []):
            findings.append(
                f"Row {row['row']}: non-ASCII ratio {row['non_ascii_ratio']}"
                f" — deviates from dominant {dominant} script"
            )
        remainder = details.get("total_anomalous_rows", 0) - len(details.get("anomalous_rows", []))
        if remainder > 0:
            findings.append(f"… and {remainder} more rows")

    elif name == "missing_values":
        for row in details.get("affected_rows", []):
            fields = ", ".join(f"'{f}'" for f in row["empty_fields"])
            verb = "is" if len(row["empty_fields"]) == 1 else "are"
            findings.append(f"Row {row['row']}: {fields} {verb} empty — fill in the expected value")
        remainder = details.get("total_affected_rows", 0) - len(details.get("affected_rows", []))
        if remainder > 0:
            findings.append(f"… and {remainder} more rows")

    elif name == "field_consistency":
        for row in details.get("incomplete_rows", []):
            missing = ", ".join(f"'{f}'" for f in row["missing_fields"])
            findings.append(f"Row {row['row']}: missing field(s) {missing} — add the field or remove the record")
        remainder = details.get("total_incomplete", 0) - len(details.get("incomplete_rows", []))
        if remainder > 0:
            findings.append(f"… and {remainder} more rows")

    elif name == "text_length":
        cfg = config.get("thresholds", {})
        min_w = cfg.get("min_text_words", 3)
        max_w = cfg.get("max_text_words", 2000)
        for row in details.get("too_short_rows", []):
            findings.append(
                f"Row {row['row']}: {row['words']} word(s) — below the {min_w}-word minimum; expand or remove"
            )
        for row in details.get("too_long_rows", []):
            findings.append(
                f"Row {row['row']}: {row['words']} words — above the {max_w}-word maximum; trim or split"
            )

    elif name == "label_quality":
        dist = details.get("label_distribution", {})
        if dist:
            sorted_dist = sorted(dist.items(), key=lambda x: x[1])
            findings.append(f"Imbalance ratio: {details.get('imbalance_ratio', '?')}:1")
            findings.append("Distribution: " + ", ".join(f"{k}: {v}" for k, v in sorted_dist))
            minority_cls, minority_cnt = sorted_dist[0]
            majority_cls, majority_cnt = sorted_dist[-1]
            needed = majority_cnt - minority_cnt
            findings.append(
                f"Class '{minority_cls}' has only {minority_cnt} record(s);"
                f" add ~{needed} more to match '{majority_cls}'"
            )

    return findings


def _action_line(name: str, check: dict, config: dict) -> str:
    """One-line action description for the prioritised action plan."""
    details = check.get("details", {})

    if name == "duplicates":
        n = details.get("duplicate_count", "?")
        shown = details.get("duplicate_rows", [])[:5]
        rows = [str(r["row"]) for r in shown]
        remainder = (details.get("duplicate_count") or 0) - len(rows)
        row_str = ", ".join(rows) + (f" … +{remainder} more" if remainder > 0 else "")
        return f"Remove {n} exact duplicate row(s): {row_str}"

    elif name == "near_duplicates":
        n = details.get("total_near_duplicate_pairs", "?")
        return f"Review {n} near-duplicate pair(s) — remove or rephrase the redundant record in each"

    elif name == "missing_values":
        n = details.get("empty_field_count", "?")
        r = details.get("total_affected_rows", "?")
        return f"Fill {n} empty field value(s) across {r} record(s)"

    elif name == "field_consistency":
        n = details.get("total_incomplete", "?")
        return f"Add missing fields to {n} incomplete record(s)"

    elif name == "text_length":
        cfg = config.get("thresholds", {})
        min_w = cfg.get("min_text_words", 3)
        max_w = cfg.get("max_text_words", 2000)
        short_n = len(details.get("too_short_rows", []))
        long_n = len(details.get("too_long_rows", []))
        parts = []
        if short_n:
            parts.append(f"expand {short_n} short record(s) to ≥{min_w} words")
        if long_n:
            parts.append(f"trim {long_n} long record(s) to ≤{max_w} words")
        return "Fix text length: " + "; ".join(parts)

    elif name == "output_diversity":
        n = details.get("total_similar_output_pairs", "?")
        return f"Review {n} near-duplicate output pair(s) — rewrite or remove redundant responses"

    elif name == "token_length":
        max_t = details.get("max_token_estimate", 2048)
        n = details.get("total_overflow_rows", "?")
        return f"Shorten {n} record(s) estimated to exceed {max_t} tokens (trim instruction or output)"

    elif name == "instruction_quality":
        n = details.get("total_flagged", "?")
        return f"Rewrite {n} flagged instruction(s): remove vague phrasing, split multi-task records"

    elif name == "language_consistency":
        n = details.get("total_anomalous_rows", "?")
        dominant = details.get("dominant_script", "unknown")
        return f"Review {n} record(s) that deviate from the dominant {dominant} script"

    elif name == "label_quality":
        ratio = details.get("imbalance_ratio", "?")
        return f"Rebalance class distribution (imbalance ratio: {ratio}:1) via oversampling or additional data"

    elif name == "json_format":
        return "Fix malformed JSON records before rerunning"

    return f"Address {name.replace('_', ' ')} issues (see row-level findings above)"


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------

@app.command("analyse")
def analyse_dataset(
    filepath: str = typer.Argument(..., help="Path to JSONL dataset file"),
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d",
        help="Override domain detection (coding|qa|translation|summarization|classification|conversation|general)",
    ),
    config_file: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Deep-dive analysis: every issue sorted by score impact, row-level findings,
    auto-detected domain with content gap suggestions, and a prioritised action plan."""
    try:
        config = load_config(config_file)
        _err.print(f"Loading dataset: [cyan]{filepath}[/cyan]")
        data = load_dataset(filepath)
        result = calculate_quality_score(data, config)

        overall = result["overall_score"]
        grade = result["grade"]
        checks = result["checks"]
        color = _rich_color(overall)

        console.print(Panel(
            f"[bold blue]Deep Analysis: {filepath}[/bold blue]",
            title="Dataset Analyser",
        ))
        console.print(
            f"\n[bold]Overall Score: [{color}]{overall:.1f}/100[/{color}]"
            f" — Grade: [{color}]{grade}[/{color}][/bold]"
        )
        console.print(
            f"Records analysed: {result['num_records']}  |  "
            f"Format: {result['detected_format']}\n"
        )

        # ── Domain intelligence ────────────────────────────────────────────
        if domain:
            detected_domain, confidence = domain.lower(), 1.0
            console.print(f"[bold]Domain:[/bold] [cyan]{detected_domain}[/cyan]  [dim](manually specified)[/dim]")
        else:
            detected_domain, confidence = detect_domain(data)
            conf_pct = int(confidence * 100)
            console.print(
                f"[bold]Detected domain:[/bold] [cyan]{detected_domain}[/cyan]  "
                f"[dim](confidence {conf_pct}%)[/dim]"
            )

        coverage = get_domain_coverage(data, detected_domain)

        if coverage:
            console.print("\n[bold]Domain coverage[/bold]")
            if detected_domain == "coding":
                langs = coverage.get("languages_detected", [])
                lang_str = ", ".join(langs) if langs else "[red]none detected[/red]"
                console.print(f"  Languages detected   : {lang_str}  [dim]({len(langs)} of 7 common)[/dim]")

                tc = coverage.get("task_type_counts", {})
                tc_str = "  |  ".join(f"{k}: {v}" for k, v in tc.items())
                console.print(f"  Task types           : {tc_str}")

                ec = coverage.get("has_edge_cases", False)
                eh = coverage.get("has_error_handling", False)
                console.print(f"  Edge-case examples   : {'[green]✓ found[/green]' if ec else '[red]✗ none detected[/red]'}")
                console.print(f"  Error-handling       : {'[green]✓ found[/green]' if eh else '[red]✗ none detected[/red]'}")

            elif detected_domain == "qa":
                dist = coverage.get("question_type_distribution", {})
                dist_str = "  |  ".join(f"{k}: {v}" for k, v in dist.items()) if dist else "[red]none[/red]"
                diversity = coverage.get("question_type_diversity", 0)
                console.print(f"  Question types       : {dist_str}  [dim]({diversity} of 6 types)[/dim]")

            elif detected_domain == "translation":
                langs = coverage.get("target_languages_detected", [])
                lang_str = ", ".join(langs) if langs else "[red]none detected[/red]"
                console.print(f"  Languages detected   : {lang_str}  [dim]({len(langs)} pair(s))[/dim]")

        # Domain-specific content gaps
        suggestions = _DOMAIN_SUGGESTIONS.get(detected_domain, _DOMAIN_SUGGESTIONS["general"])
        console.print(f"\n[bold]What this dataset is lacking[/bold]  [dim]({detected_domain} domain)[/dim]")
        for suggestion in suggestions:
            console.print(f"  [magenta]▸[/magenta] {suggestion}")

        console.print(f"\n{'─' * 72}")

        # ── Data-quality issues ────────────────────────────────────────────
        issues = []
        for name, check in checks.items():
            score_gap = 1.0 - check["score"]
            impact = score_gap * check["weight"] * 100
            if score_gap > 1e-9:
                issues.append((name, check, impact))
        issues.sort(key=lambda x: x[2], reverse=True)

        if not issues:
            console.print(Panel(
                "[green]No data-quality issues found — all checks are at 100%.[/green]",
                title="Data Quality",
            ))
        else:
            # Summary table
            tbl = Table(title="Data-quality issues (sorted by score impact)", show_lines=True)
            tbl.add_column("#", justify="right", style="dim", no_wrap=True)
            tbl.add_column("Check", style="cyan", no_wrap=True)
            tbl.add_column("Status", justify="center")
            tbl.add_column("Score", justify="right")
            tbl.add_column("Score lost", justify="right")
            tbl.add_column("Finding")
            for rank, (name, check, impact) in enumerate(issues, 1):
                status = "[red]FAIL[/red]" if not check["passed"] else "[yellow]WARN[/yellow]"
                tbl.add_row(
                    str(rank),
                    name.replace("_", " ").title(),
                    status,
                    f"{check['score']*100:.1f}",
                    f"[red]-{impact:.1f} pts[/red]",
                    check["message"],
                )
            console.print(tbl)

            # Row-level findings
            console.print("\n[bold]Row-level findings[/bold]")
            for rank, (name, check, impact) in enumerate(issues, 1):
                findings = _build_findings(name, check, config)
                if not findings:
                    continue
                console.print(
                    f"\n  [bold cyan]#{rank} — {name.replace('_', ' ').title()}[/bold cyan]"
                    f"  [dim](costs {impact:.1f} pts)[/dim]"
                )
                for finding in findings:
                    console.print(f"    [yellow]▶[/yellow] {finding}")

            # Action plan
            console.print(f"\n{'─' * 72}")
            console.print(
                "[bold]Action plan[/bold]  "
                "[dim](apply in order — highest score gain first)[/dim]\n"
            )
            for rank, (name, check, impact) in enumerate(issues, 1):
                action = _action_line(name, check, config)
                console.print(f"  [bold]{rank}.[/bold] [green]+{impact:.1f} pts[/green]  {action}")

            potential = sum(impact for _, _, impact in issues)
            new_score = min(overall + potential, 100.0)
            console.print(
                f"\n  Fixing all issues above would raise the score: "
                f"[{color}]{overall:.1f}[/{color}] → [green]{new_score:.1f}[/green] / 100\n"
            )

    except FileNotFoundError:
        _err.print(f"[red]Error: File not found: {filepath}[/red]")
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        _err.print(f"[red]Error: Invalid JSON in dataset: {e}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# autofix
# ---------------------------------------------------------------------------

@app.command("autofix")
def autofix_dataset(
    filepath: str = typer.Argument(..., help="Path to JSONL dataset file"),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Output path for the cleaned dataset (default: <name>.fixed.jsonl)",
    ),
    config_file: Optional[str] = typer.Option(None, "--config", "-c"),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show what would change without writing any files",
    ),
):
    """Auto-remove exact duplicates and write a cleaned dataset file.

    Shows a before/after score. Use --dry-run to preview without writing.
    """
    try:
        config = load_config(config_file)
        _err.print(f"Loading dataset: [cyan]{filepath}[/cyan]")
        data = load_dataset(filepath)
        result = calculate_quality_score(data, config)

        overall_before = result["overall_score"]
        dup_details = result["checks"]["duplicates"]["details"]
        dup_row_set = {r["row"] for r in dup_details.get("duplicate_rows", [])}

        if not dup_row_set:
            console.print("[green]No exact duplicates found — dataset is already clean.[/green]")
            return

        console.print(Panel(
            f"[bold blue]Auto-fix: {filepath}[/bold blue]",
            title="Dataset Autofix",
        ))

        # Partition into kept / removed (rows are 1-indexed in details)
        cleaned, removed_rows = [], []
        for i, record in enumerate(data, 1):
            if i in dup_row_set:
                removed_rows.append(i)
            else:
                cleaned.append(record)

        console.print(f"\n[bold]Removals[/bold]")
        for row_num in removed_rows:
            orig_row = data[row_num - 1]
            # Show the first field value as a short preview
            preview_val = next((str(v)[:60] for v in orig_row.values() if v), "")
            console.print(f"  [red]–[/red] Row {row_num}  [dim]{preview_val}[/dim]")

        console.print(f"\n  Removed : [red]{len(removed_rows)}[/red] duplicate record(s)")
        console.print(f"  Records : {len(data)} → {len(cleaned)}")

        # Rescore the cleaned dataset
        cleaned_result = calculate_quality_score(cleaned, config)
        overall_after = cleaned_result["overall_score"]
        cb = _rich_color(overall_before)
        ca = _rich_color(overall_after)
        console.print(
            f"  Score   : [{cb}]{overall_before:.1f}[/{cb}]"
            f" → [{ca}]{overall_after:.1f}[/{ca}] / 100\n"
        )

        if dry_run:
            console.print("[yellow]Dry run — no file written. Remove --dry-run to apply.[/yellow]")
            return

        out_path = output_file or (Path(filepath).stem + ".fixed.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for record in cleaned:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        console.print(f"[green]Cleaned dataset written to:[/green] [cyan]{out_path}[/cyan]")

    except FileNotFoundError:
        _err.print(f"[red]Error: File not found: {filepath}[/red]")
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        _err.print(f"[red]Error: Invalid JSON: {e}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@app.command("stats")
def dataset_stats(
    filepath: str = typer.Argument(..., help="Path to JSONL dataset file"),
):
    """Show detailed field and label statistics for a dataset."""
    try:
        data = load_dataset(filepath)

        console.print(f"[bold]Dataset Statistics:[/bold] {filepath}")
        console.print(f"Total records: {len(data)}")

        all_keys: set = set()
        for item in data:
            all_keys.update(item.keys())

        console.print("\n[bold]Fields:[/bold]")
        for key in sorted(all_keys):
            count = sum(1 for item in data if key in item and item[key] is not None
                        and (not isinstance(item[key], str) or item[key].strip()))
            total = sum(1 for item in data if key in item)
            console.print(
                f"  {key}: present in {total}/{len(data)} records, "
                f"non-empty in {count}/{len(data)} records"
            )

        label_fields = ["label", "category", "class", "target"]
        for field in label_fields:
            if field in all_keys:
                labels = [str(item[field]) for item in data if field in item and item[field] is not None]
                if not labels:
                    continue
                unique = sorted(set(labels))
                console.print(f"\n[bold]Label distribution ({field}):[/bold]")
                for lbl in unique:
                    cnt = labels.count(lbl)
                    console.print(f"  {lbl}: {cnt} ({cnt/len(labels)*100:.1f}%)")
                break

    except Exception as e:
        _err.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# crosscheck
# ---------------------------------------------------------------------------

@app.command("crosscheck")
def crosscheck_datasets(
    filepath1: str = typer.Argument(..., help="First JSONL file (e.g. train set)"),
    filepath2: str = typer.Argument(..., help="Second JSONL file (e.g. test set)"),
):
    """Detect records from Dataset 2 that also appear in Dataset 1 (train/test leakage).

    Performs exact-match comparison using full-record JSON fingerprints.
    """
    try:
        _err.print(f"Loading: [cyan]{filepath1}[/cyan]")
        data1 = load_dataset(filepath1)
        _err.print(f"Loading: [cyan]{filepath2}[/cyan]")
        data2 = load_dataset(filepath2)

        fingerprints1 = {json.dumps(r, sort_keys=True) for r in data1}

        overlap: list[dict] = []
        for i, record in enumerate(data2, 1):
            if json.dumps(record, sort_keys=True) in fingerprints1:
                overlap.append({"row": i, "record": record})

        console.print(Panel(
            f"[bold blue]Cross-dataset Check[/bold blue]\n"
            f"[dim]{Path(filepath1).name}[/dim]  ×  [dim]{Path(filepath2).name}[/dim]",
            title="Dataset Crosscheck",
        ))
        console.print(f"\n  {Path(filepath1).name} records : {len(data1)}")
        console.print(f"  {Path(filepath2).name} records : {len(data2)}")

        if not overlap:
            console.print(f"\n  [green]✓ No overlap found — datasets are clean.[/green]\n")
            return

        leakage_pct = len(overlap) / len(data2) * 100
        console.print(
            f"\n  [red]✗ {len(overlap)} record(s) from {Path(filepath2).name}"
            f" also appear in {Path(filepath1).name}[/red]"
        )
        console.print(f"  Leakage rate : {leakage_pct:.1f}% of {Path(filepath2).name}\n")

        tbl = Table(
            title=f"Leaked records ({Path(filepath2).name} rows found in {Path(filepath1).name})",
            show_lines=True,
        )
        tbl.add_column(f"Row in {Path(filepath2).name}", justify="right", style="red", no_wrap=True)
        tbl.add_column("Content preview")

        for item in overlap[:20]:
            preview = next((str(v)[:90] for v in item["record"].values() if v), "")
            tbl.add_row(str(item["row"]), preview)
        if len(overlap) > 20:
            tbl.add_row("…", f"and {len(overlap) - 20} more")
        console.print(tbl)

    except FileNotFoundError as e:
        _err.print(f"[red]Error: File not found: {e.filename}[/red]")
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        _err.print(f"[red]Error: Invalid JSON: {e}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
