"""Report generation for dataset quality scores - Terminal, JSON, and HTML formats."""

import json
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def generate_terminal_report(score_result: Dict[str, Any], filepath: str = None) -> str:
    """Generate a terminal report using Rich."""
    console = Console()
    
    console.print(Panel(
        Text("Dataset Quality Report: " + (filepath or "Unknown"), style="bold blue"),
        title="Dataset Quality Scorer"
    ))
    
    overall_score = score_result["overall_score"]
    score_color = "green" if overall_score >= 70 else "yellow" if overall_score >= 50 else "red"
    
    console.print("")
    console.print("[bold]Overall Score: [" + score_color + "]" + f"{overall_score:.1f}" + "/100[/" + score_color + "][/bold]")
    console.print("Records analyzed: " + str(score_result["num_records"]))
    
    table = Table(title="Quality Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Score", style="green")
    table.add_column("Details", style="yellow")
    
    for check_name, check_data in score_result["checks"].items():
        status = "PASS" if check_data["passed"] else "FAIL"
        score_text = f"{check_data['score']*100:.1f}"
        table.add_row(
            check_name.replace("_", " ").title(),
            status,
            score_text,
            check_data["message"]
        )
    
    console.print(table)
    
    if overall_score >= 80:
        recommendation = "Dataset quality is excellent. Ready for fine-tuning."
    elif overall_score >= 60:
        recommendation = "Dataset quality is acceptable but could be improved."
    else:
        recommendation = "Dataset quality is low. Consider fixing issues before fine-tuning."
    
    console.print(Panel(recommendation, title="Recommendation"))
    
    return "Score: " + f"{overall_score:.1f}" + "/100"


def generate_json_report(score_result: Dict[str, Any], filepath: str = None) -> str:
    """Generate a JSON report."""
    report = {
        "dataset_path": filepath,
        "overall_score": score_result["overall_score"],
        "num_records": score_result["num_records"],
        "checks": score_result["checks"]
    }
    return json.dumps(report, indent=2)


def generate_html_report(score_result: Dict[str, Any], filepath: str = None) -> str:
    """Generate an HTML report with Chart.js visualization."""
    overall_score = score_result["overall_score"]
    
    check_names = []
    check_scores = []
    
    for check_name, check_data in score_result["checks"].items():
        check_names.append(check_name.replace("_", " ").title())
        check_scores.append(check_data["score"] * 100)
    
    check_names_json = json.dumps(check_names)
    check_scores_json = json.dumps(check_scores)
    
    if overall_score >= 70:
        score_color = "#27ae60"
    elif overall_score >= 50:
        score_color = "#f39c12"
    else:
        score_color = "#e74c3c"
    
    if overall_score >= 80:
        recommendation = "Dataset quality is excellent. Ready for fine-tuning."
    elif overall_score >= 60:
        recommendation = "Dataset quality is acceptable but could be improved."
    else:
        recommendation = "Dataset quality is low. Consider fixing issues before fine-tuning."
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Quality Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .score-display { text-align: center; font-size: 48px; font-weight: bold; color: """ + score_color + """; margin: 20px 0; }
        .chart-container { margin: 30px 0; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #3498db; color: white; }
        .pass { color: green; } .fail { color: red; }
        .recommendation { background: #e8f4f8; padding: 15px; border-radius: 5px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dataset Quality Report</h1>
        <p>Dataset: """ + (filepath or "Unknown") + """</p>
        <div class="score-display">""" + str(round(overall_score, 1)) + """ / 100</div>
        <div class="chart-container"><canvas id="qualityChart"></canvas></div>
        <table><thead><tr><th>Check</th><th>Status</th><th>Score</th><th>Details</th></tr></thead><tbody>
"""
    
    for check_name, check_data in score_result["checks"].items():
        status_class = "pass" if check_data["passed"] else "fail"
        status_icon = "[PASS]" if check_data["passed"] else "[FAIL]"
        html_content += "<tr><td>" + check_name.replace("_", " ").title() + "</td><td class=\"" + status_class + "\">" + status_icon + "</td><td>" + str(round(check_data["score"]*100, 1)) + "</td><td>" + check_data["message"] + "</td></tr>\n"
    
    html_content += """</tbody></table>
        <div class="recommendation"><strong>Recommendation:</strong> """ + recommendation + """</div>
    </div>
    <script>
        const ctx = document.getElementById('qualityChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: { labels: """ + check_names_json + """}, datasets: [{ label: 'Check Score', data: """ + check_scores_json + """}, backgroundColor: ['rgba(52,152,219,0.7)','rgba(46,204,113,0.7)','rgba(155,89,182,0.7)','rgba(241,196,15,0.7)','rgba(230,126,34,0.7)'] }] },
            options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
        });
    </script>
</body>
</html>"""
    
    return html_content


def get_score_color(score: float) -> str:
    """Get color hex code based on score."""
    if score >= 70:
        return "#27ae60"
    elif score >= 50:
        return "#f39c12"
    else:
        return "#e74c3c"
