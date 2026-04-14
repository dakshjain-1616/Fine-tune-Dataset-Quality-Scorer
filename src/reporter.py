"""Report generation — terminal (Rich), JSON, and HTML formats."""

import json
from typing import Any, Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Score-band thresholds (mirror config.yaml defaults)
_READY = 92
_CAUTION = 80
_NEEDS_WORK = 60


def _grade_label(score: float, grade: str = None) -> str:
    if grade:
        return grade
    if score >= _READY:
        return "READY"
    elif score >= _CAUTION:
        return "CAUTION"
    elif score >= _NEEDS_WORK:
        return "NEEDS_WORK"
    return "NOT_READY"


def _recommendation(score: float, grade: str = None) -> str:
    g = _grade_label(score, grade)
    if g == "READY":
        return "Dataset quality is excellent. Ready for fine-tuning."
    elif g == "CAUTION":
        return "Dataset has minor quality issues. Review flagged checks before fine-tuning."
    elif g == "NEEDS_WORK":
        return "Dataset has significant quality issues. Fix problems before fine-tuning."
    return "Dataset has critical quality issues. Do NOT fine-tune on this data yet."


def _score_color_rich(score: float) -> str:
    if score >= _READY:
        return "green"
    elif score >= _CAUTION:
        return "yellow"
    elif score >= _NEEDS_WORK:
        return "dark_orange"
    return "red"


def _score_color_hex(score: float) -> str:
    if score >= _READY:
        return "#27ae60"
    elif score >= _CAUTION:
        return "#f39c12"
    elif score >= _NEEDS_WORK:
        return "#e67e22"
    return "#e74c3c"


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def generate_terminal_report(score_result: Dict[str, Any], filepath: str = None) -> str:
    """Print a Rich-formatted quality report to the terminal and return a summary string."""
    console = Console()
    overall = score_result["overall_score"]
    grade = score_result.get("grade", _grade_label(overall))
    color = _score_color_rich(overall)
    fmt = score_result.get("detected_format", "unknown")

    console.print(Panel(
        Text(f"Dataset Quality Report: {filepath or 'Unknown'}", style="bold blue"),
        title="Dataset Quality Scorer",
    ))

    console.print("")
    console.print(
        f"[bold]Overall Score: [{color}]{overall:.1f}/100[/{color}] "
        f"— Grade: [{color}]{grade}[/{color}][/bold]"
    )
    console.print(f"Records analysed : {score_result['num_records']}")
    console.print(f"Detected format  : {fmt}")
    console.print("")

    table = Table(title="Quality Checks", show_lines=True)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Details")

    for check_name, check_data in score_result["checks"].items():
        status_str = "[green]PASS[/green]" if check_data["passed"] else "[red]FAIL[/red]"
        score_pct = check_data["score"] * 100
        weight_pct = check_data.get("weight", 0) * 100
        table.add_row(
            check_name.replace("_", " ").title(),
            status_str,
            f"{score_pct:.1f}",
            f"{weight_pct:.0f}%",
            check_data["message"],
        )

    console.print(table)

    rec = _recommendation(overall, grade)
    rec_style = color
    console.print(Panel(f"[{rec_style}]{rec}[/{rec_style}]", title="Recommendation"))

    return f"Score: {overall:.1f}/100 ({grade})"


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def generate_json_report(score_result: Dict[str, Any], filepath: str = None) -> str:
    """Serialise the full score result as indented JSON and return the string."""
    report = {
        "dataset_path": filepath,
        "overall_score": round(score_result["overall_score"], 2),
        "grade": score_result.get("grade", _grade_label(score_result["overall_score"])),
        "num_records": score_result["num_records"],
        "detected_format": score_result.get("detected_format", "unknown"),
        "checks": {
            name: {
                "passed": v["passed"],
                "message": v["message"],
                "score": round(v["score"], 4),
                "weight": v.get("weight", 0),
                "details": v.get("details", {}),
            }
            for name, v in score_result["checks"].items()
        },
    }
    return json.dumps(report, indent=2)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(score_result: Dict[str, Any], filepath: str = None) -> str:
    """Generate a self-contained HTML report with a Chart.js bar chart."""
    overall = score_result["overall_score"]
    grade = score_result.get("grade", _grade_label(overall))
    score_color = _score_color_hex(overall)
    rec = _recommendation(overall, grade)

    check_names = []
    check_scores = []
    for name, data in score_result["checks"].items():
        check_names.append(name.replace("_", " ").title())
        check_scores.append(round(data["score"] * 100, 1))

    # Build table rows
    table_rows = ""
    for name, data in score_result["checks"].items():
        css = "pass" if data["passed"] else "fail"
        icon = "✔ PASS" if data["passed"] else "✘ FAIL"
        table_rows += (
            f"<tr>"
            f"<td>{name.replace('_', ' ').title()}</td>"
            f"<td class='{css}'>{icon}</td>"
            f"<td>{round(data['score']*100, 1)}</td>"
            f"<td>{round(data.get('weight', 0)*100)}%</td>"
            f"<td>{data['message']}</td>"
            f"</tr>\n"
        )

    labels_json = json.dumps(check_names)
    scores_json = json.dumps(check_scores)
    bg_colors = json.dumps([
        "rgba(52,152,219,0.75)",
        "rgba(46,204,113,0.75)",
        "rgba(155,89,182,0.75)",
        "rgba(241,196,15,0.75)",
        "rgba(230,126,34,0.75)",
        "rgba(231,76,60,0.75)",
        "rgba(26,188,156,0.75)",
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dataset Quality Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 900px; margin: 40px auto; padding: 20px; background: #f5f5f5;
    }}
    .container {{ background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,.1); }}
    h1 {{ color: #2c3e50; text-align: center; margin-bottom: 4px; }}
    .meta {{ text-align: center; color: #666; margin-bottom: 20px; }}
    .score-display {{
      text-align: center; font-size: 56px; font-weight: bold;
      color: {score_color}; margin: 16px 0 4px;
    }}
    .grade {{
      text-align: center; font-size: 20px; font-weight: 600;
      color: {score_color}; margin-bottom: 24px;
    }}
    .chart-container {{ max-height: 300px; margin: 20px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }}
    th {{ background: #3498db; color: #fff; }}
    tr:hover {{ background: #f8f8f8; }}
    .pass {{ color: #27ae60; font-weight: 600; }}
    .fail {{ color: #e74c3c; font-weight: 600; }}
    .recommendation {{
      background: #eaf4fb; border-left: 4px solid {score_color};
      padding: 14px 18px; border-radius: 4px; margin-top: 24px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Dataset Quality Report</h1>
    <p class="meta">Dataset: {filepath or 'Unknown'} &nbsp;|&nbsp; Records: {score_result['num_records']} &nbsp;|&nbsp; Format: {score_result.get('detected_format', 'unknown')}</p>
    <div class="score-display">{round(overall, 1)} / 100</div>
    <div class="grade">{grade}</div>

    <div class="chart-container">
      <canvas id="qualityChart"></canvas>
    </div>

    <table>
      <thead>
        <tr><th>Check</th><th>Status</th><th>Score</th><th>Weight</th><th>Details</th></tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>

    <div class="recommendation"><strong>Recommendation:</strong> {rec}</div>
  </div>

  <script>
    const ctx = document.getElementById('qualityChart').getContext('2d');
    new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: {labels_json},
        datasets: [{{
          label: 'Check Score (0-100)',
          data: {scores_json},
          backgroundColor: {bg_colors}
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          y: {{ beginAtZero: true, max: 100, ticks: {{ callback: v => v + '%' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Legacy helper (kept for compatibility)
# ---------------------------------------------------------------------------

def get_score_color(score: float) -> str:
    """Return a hex colour code based on score."""
    return _score_color_hex(score)
