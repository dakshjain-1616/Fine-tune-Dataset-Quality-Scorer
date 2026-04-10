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

from src.checks import load_dataset, load_config, calculate_quality_score
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
            row_str = ", ".join(rows[:10]) + ("…" if len(rows) > 10 else "")
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
            row_str = ", ".join(rows[:10]) + ("…" if len(rows) > 10 else "")
            suggestions.append(
                f"[bold yellow]Missing Values[/bold yellow] — "
                f"{mv['details'].get('empty_field_count', '?')} empty/null field values "
                f"across {mv['details'].get('total_affected_rows', '?')} records. "
                f"Affected rows: {row_str}"
            )

        # Exact duplicates
        dup = checks["duplicates"]
        if not dup["passed"]:
            dup_info = [
                f"row {r['row']} (dup of row {r['duplicate_of_row']})"
                for r in dup["details"].get("duplicate_rows", [])[:10]
            ]
            suggestions.append(
                f"[bold yellow]Exact Duplicates[/bold yellow] — "
                f"{dup['details'].get('duplicate_count', '?')} duplicate records found. "
                f"Remove: {', '.join(dup_info)}"
            )

        # Near duplicates
        nd = checks["near_duplicates"]
        if not nd["passed"]:
            pair_info = [
                f"rows {p['row1']}&{p['row2']} (sim {p['similarity']})"
                for p in nd["details"].get("near_duplicate_pairs", [])[:5]
            ]
            suggestions.append(
                f"[bold yellow]Near-Duplicates[/bold yellow] — "
                f"{nd['details'].get('total_near_duplicate_pairs', '?')} near-duplicate pairs "
                f"(Jaccard ≥ {nd['details'].get('similarity_threshold', 0.85)}). "
                f"Review: {', '.join(pair_info)}"
            )

        # Text length
        tl = checks["text_length"]
        if not tl["passed"]:
            short = [str(r["row"]) for r in tl["details"].get("too_short_rows", [])[:5]]
            long_ = [str(r["row"]) for r in tl["details"].get("too_long_rows", [])[:5]]
            n_out = tl["details"].get("total_outliers", "?")
            rec_w = "record" if n_out == 1 else "records"
            parts = []
            if short:
                parts.append(f"too short: rows {', '.join(short)}")
            if long_:
                parts.append(f"too long: rows {', '.join(long_)}")
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


if __name__ == "__main__":
    app()
