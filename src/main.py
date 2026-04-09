"""Main CLI entry point for Dataset Quality Scorer."""

import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel

from src.checks import load_dataset, calculate_quality_score
from src.reporter import generate_terminal_report, generate_json_report, generate_html_report

console = Console()

app = typer.Typer(help="Fine-tune Dataset Quality Scorer CLI")


@app.command("score")
def score_dataset(
    filepath: str = typer.Argument(..., help="Path to JSONL dataset file"),
    output_format: str = typer.Option("terminal", "--format", "-f", help="Output format: terminal, json, html"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save report to file")
):
    """Score a dataset and generate a quality report."""
    try:
        console.print(f"Loading dataset: {filepath}")
        data = load_dataset(filepath)
        score_result = calculate_quality_score(data)
        
        if output_format == "terminal":
            report = generate_terminal_report(score_result, filepath)
            console.print(report)
        elif output_format == "json":
            report = generate_json_report(score_result, filepath)
            console.print(report)
        elif output_format == "html":
            report = generate_html_report(score_result, filepath)
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                console.print(f"HTML report saved to: {output_file}")
            else:
                console.print("HTML report generated (use --output to save)")
        
        return score_result
        
    except FileNotFoundError:
        console.print(f"[red]Error: File not found: {filepath}[/red]")
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON format: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("quick")
def quick_score(filepath: str = typer.Argument(..., help="Path to JSONL dataset file")):
    """Quick score check - returns only the overall score."""
    try:
        data = load_dataset(filepath)
        score_result = calculate_quality_score(data)
        console.print(f"[bold green]{score_result['overall_score']:.1f}[/bold green]")
        return score_result['overall_score']
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("compare")
def compare_datasets(
    filepath1: str = typer.Argument(..., help="Path to first JSONL dataset"),
    filepath2: str = typer.Argument(..., help="Path to second JSONL dataset")
):
    """Compare quality scores between two datasets."""
    try:
        console.print(f"\n[bold]Dataset 1: {filepath1}[/bold]")
        data1 = load_dataset(filepath1)
        score1 = calculate_quality_score(data1)
        generate_terminal_report(score1, filepath1)
        
        console.print(f"\n[bold]Dataset 2: {filepath2}[/bold]")
        data2 = load_dataset(filepath2)
        score2 = calculate_quality_score(data2)
        generate_terminal_report(score2, filepath2)
        
        console.print("\n[bold]Comparison:[/bold]")
        diff = score1['overall_score'] - score2['overall_score']
        if diff > 0:
            console.print(f"[green]Dataset 1 is better by {diff:.1f} points[/green]")
        elif diff < 0:
            console.print(f"[green]Dataset 2 is better by {abs(diff):.1f} points[/green]")
        else:
            console.print("[yellow]Both datasets have equal quality[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("fix")
def fix_suggestions(filepath: str = typer.Argument(..., help="Path to JSONL dataset file")):
    """Get suggestions for fixing dataset quality issues."""
    try:
        data = load_dataset(filepath)
        score_result = calculate_quality_score(data)
        
        console.print(Panel(f"Quality Fix Suggestions for: {filepath}", title="Fix Recommendations"))
        
        suggestions = []
        checks = score_result["checks"]
        
        if not checks["field_consistency"]["passed"]:
            suggestions.append("Standardize fields across all records")
        if not checks["missing_values"]["passed"]:
            suggestions.append("Fill in missing values or remove records with empty fields")
        if not checks["duplicates"]["passed"]:
            suggestions.append("Remove duplicate records from the dataset")
        if not checks["label_quality"]["passed"]:
            suggestions.append("Balance label distribution")
        if score_result["overall_score"] < 50:
            suggestions.append("Consider collecting more high-quality examples")
        
        if suggestions:
            for suggestion in suggestions:
                console.print(suggestion)
        else:
            console.print("[green]No critical issues found. Dataset is ready for fine-tuning.[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("stats")
def dataset_stats(filepath: str = typer.Argument(..., help="Path to JSONL dataset file")):
    """Show detailed statistics about a dataset."""
    try:
        data = load_dataset(filepath)
        
        console.print(f"[bold]Dataset Statistics: {filepath}[/bold]")
        console.print(f"Total records: {len(data)}")
        
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        console.print(f"\n[bold]Fields found:[/bold]")
        for key in sorted(all_keys):
            count = sum(1 for item in data if key in item)
            console.print(f"  - {key}: {count}/{len(data)} records ({count/len(data)*100:.1f}%)")
        
        label_fields = ['label', 'category', 'class', 'target', 'answer']
        for field in label_fields:
            if field in all_keys:
                labels = [item[field] for item in data if field in item]
                unique_labels = set(labels)
                console.print(f"\n[bold]Label distribution ({field}):[/bold]")
                for label in sorted(unique_labels):
                    count = labels.count(label)
                    console.print(f"  - {label}: {count} ({count/len(labels)*100:.1f}%)")
                break
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
