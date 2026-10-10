"""Command-line interface for llmtrack analytics reporting."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from .storage import Storage

# Load SQL queries from queries.sql
QUERIES_FILE = Path(__file__).parent / "queries.sql"


def load_queries() -> dict[str, str]:
    """Parse queries.sql into a dict of query_name -> sql_string."""
    if not QUERIES_FILE.exists():
        return {}
    content = QUERIES_FILE.read_text(encoding="utf-8")
    queries: dict[str, str] = {}
    current_name: Optional[str] = None
    buffer: list[str] = []

    for line in content.splitlines():
        name_match = re.match(r"^--\s*NAME:\s*(\w+)", line)
        if name_match:
            if current_name and buffer:
                queries[current_name] = "\n".join(buffer).strip()
                buffer = []
            current_name = name_match.group(1)
        elif current_name:
            buffer.append(line)

    if current_name and buffer:
        queries[current_name] = "\n".join(buffer).strip()

    return queries


def parse_since(since_str: Optional[str]) -> Optional[str]:
    """Convert human duration string like '7d', '24h', '30m' into ISO timestamp cutoff."""
    if not since_str:
        return None

    s = since_str.strip().lower()
    now = datetime.now(timezone.utc)

    # Check for unit suffixes
    match = re.match(r"^(\d+)\s*(s|m|h|d|w)$", s)
    if match:
        val = int(match.group(1))
        unit = match.group(2)
        if unit == "s":
            cutoff = now - timedelta(seconds=val)
        elif unit == "m":
            cutoff = now - timedelta(minutes=val)
        elif unit == "h":
            cutoff = now - timedelta(hours=val)
        elif unit == "d":
            cutoff = now - timedelta(days=val)
        elif unit == "w":
            cutoff = now - timedelta(weeks=val)
        else:
            return None
        return cutoff.isoformat()

    # Maybe already an ISO date/timestamp string
    try:
        dt = datetime.fromisoformat(s)
        return dt.isoformat()
    except ValueError:
        return None


def print_ascii_table(title: str, rows: list[dict[str, Any]]) -> None:
    """Print a clean formatted plain-text table."""
    print(f"\n=== {title} ===")
    if not rows:
        print("  (no data)")
        return

    columns = list(rows[0].keys())
    # Calculate widths
    col_widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            val_str = str(row.get(col, "") if row.get(col) is not None else "")
            col_widths[col] = max(col_widths[col], len(val_str))

    header_line = " | ".join(col.ljust(col_widths[col]) for col in columns)
    sep_line = "-+-".join("-" * col_widths[col] for col in columns)
    print(header_line)
    print(sep_line)

    for row in rows:
        vals = []
        for col in columns:
            v = row.get(col)
            val_str = str(v) if v is not None else ""
            if isinstance(v, (int, float)):
                vals.append(val_str.rjust(col_widths[col]))
            else:
                vals.append(val_str.ljust(col_widths[col]))
        print(" | ".join(vals))


def render_rich_table(title: str, rows: list[dict[str, Any]]) -> None:
    """Render a table using Rich if available."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=title, title_style="bold cyan", header_style="bold magenta")

        if not rows:
            table.add_column("Notice")
            table.add_row("No data found")
            console.print(table)
            return

        columns = list(rows[0].keys())
        for col in columns:
            is_right = any(k in col for k in ("cost", "tokens", "ms", "calls"))
            table.add_column(col, justify="right" if is_right else "left")

        for row in rows:
            formatted_vals = []
            for col in columns:
                v = row.get(col)
                if v is None:
                    formatted_vals.append("[dim]-[/dim]")
                elif isinstance(v, float):
                    if "cost" in col:
                        formatted_vals.append(f"[green]${v:.4f}[/green]")
                    elif "ms" in col:
                        formatted_vals.append(f"{v:.1f} ms")
                    elif "pct" in col or "rate" in col:
                        style = "red" if v > 10.0 else "yellow" if v > 0.0 else "green"
                        formatted_vals.append(f"[{style}]{v:.1f}%[/{style}]")
                    else:
                        formatted_vals.append(str(v))
                else:
                    formatted_vals.append(str(v))
            table.add_row(*formatted_vals)

        console.print(table)
    except ImportError:
        print_ascii_table(title, rows)


def generate_html_report(results: dict[str, list[dict[str, Any]]]) -> str:
    """Generate a self-contained dark-mode HTML dashboard."""
    json_data = json.dumps(results)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>llmtrack // Observability Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-heading: #f0f6fc;
            --accent: #58a6ff;
            --green: #3fb950;
            --red: #f85149;
            --yellow: #d29922;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        .header h1 {{
            margin: 0;
            color: var(--text-heading);
            font-size: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .badge {{
            background: #238636;
            color: #fff;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}
        .card h2 {{
            margin-top: 0;
            font-size: 16px;
            color: var(--text-heading);
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            text-align: left;
            padding: 8px 10px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: var(--text-heading);
            background: rgba(255, 255, 255, 0.02);
        }}
        .num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .cost {{ color: var(--green); font-weight: 600; }}
        .error {{ color: var(--red); }}
    </style>
</head>
<body>
    <div class="header">
        <h1><span>⚡</span> llmtrack Dashboard</h1>
        <div class="badge">SQLite WAL Mode Active</div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Cost & Volume by Day</h2>
            <div id="cost-table-container"></div>
        </div>
        <div class="card">
            <h2>p95 Latency by Model</h2>
            <div id="p95-table-container"></div>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Top 10 Most Expensive Tags</h2>
            <div id="tags-table-container"></div>
        </div>
        <div class="card">
            <h2>Error & Retry Rates</h2>
            <div id="errors-table-container"></div>
        </div>
    </div>

    <script>
        const data = {json_data};

        function renderTable(containerId, rows) {{
            const el = document.getElementById(containerId);
            if (!rows || rows.length === 0) {{
                el.innerHTML = "<p style='color:#8b949e'>No records found</p>";
                return;
            }}
            const cols = Object.keys(rows[0]);
            let ths = cols.map(c => `<th>${{c}}</th>`).join("");
            let html = "<table><thead><tr>" + ths + "</tr></thead><tbody>";
            for (const r of rows) {{
                html += "<tr>" + cols.map(c => {{
                    const val = r[c] ?? "-";
                    const isNum = typeof val === 'number';
                    const isCost = c.includes('cost');
                    const cls = isCost ? 'num cost' : (isNum ? 'num' : '');
                    const txt = isCost && isNum ? '$' + val.toFixed(4) : val;
                    return `<td class="${{cls}}">${{txt}}</td>`;
                }}).join("") + "</tr>";
            }}
            html += "</tbody></table>";
            el.innerHTML = html;
        }}

        renderTable('cost-table-container', data['Cost & Volume Per Day']);
        renderTable('p95-table-container', data['p95 Latency Per Model (ms)']);
        renderTable('tags-table-container', data['Top 10 Most Expensive Tags']);
        renderTable('errors-table-container', data['Error & Retry Rates Per Model']);
    </script>
</body>
</html>"""


def run_report(
    db_path: str = "llm_calls.db",
    since: Optional[str] = None,
    output_format: str = "table",
    out_file: Optional[str] = None,
) -> int:
    """Run analytics queries and render formatted output."""
    db_file = Path(db_path)
    if not db_file.exists():
        print(f"llmtrack: Database file '{db_path}' does not exist.", file=sys.stderr)
        return 1

    storage = Storage(db_path=db_path)
    queries = load_queries()
    cutoff_ts = parse_since(since)

    all_results: dict[str, list[dict[str, Any]]] = {}

    order = [
        ("cost_per_day", "Cost & Volume Per Day"),
        ("p95_latency", "p95 Latency Per Model (ms)"),
        ("error_and_retry_rates", "Error & Retry Rates Per Model"),
        ("top_tags", "Top 10 Most Expensive Tags"),
        ("slowest_calls", "Slowest 20 Calls"),
        ("rate_limit_incidents", "Rate Limit Incidents (Hourly)"),
    ]

    try:
        for key, title in order:
            if key in queries:
                sql = queries[key]
                try:
                    # Two params for (? IS NULL OR ts_utc >= ?)
                    rows = storage.query(sql, (cutoff_ts, cutoff_ts))
                    all_results[title] = rows
                except Exception as e:
                    all_results[title] = [{"error": str(e)}]
    finally:
        storage.close()

    if output_format == "html":
        html_content = generate_html_report(all_results)
        if out_file:
            Path(out_file).write_text(html_content, encoding="utf-8")
            print(f"[+] Wrote HTML report to '{out_file}'")
        else:
            print(html_content)
        return 0

    if output_format == "json":
        json_str = json.dumps(all_results, indent=2)
        if out_file:
            Path(out_file).write_text(json_str, encoding="utf-8")
            print(f"[+] Wrote JSON report to '{out_file}'")
        else:
            print(json_str)
        return 0

    if output_format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        for title, rows in all_results.items():
            writer.writerow([f"=== {title} ==="])
            if rows and "error" not in rows[0]:
                writer.writerow(list(rows[0].keys()))
                for r in rows:
                    writer.writerow(list(r.values()))
            else:
                writer.writerow(["No data"])
            writer.writerow([])
        csv_str = output.getvalue()
        if out_file:
            Path(out_file).write_text(csv_str, encoding="utf-8")
            print(f"[+] Wrote CSV report to '{out_file}'")
        else:
            print(csv_str)
        return 0

    # Table format (Rich if available, fallback to plain text)
    use_rich = "rich" in sys.modules or _can_import_rich()
    for title, rows in all_results.items():
        if use_rich:
            render_rich_table(title, rows)
            print()
        else:
            print_ascii_table(title, rows)

    return 0


def _can_import_rich() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="llmtrack",
        description="llmtrack: CLI reporting tool for LLM token usage, costs, and latency.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    report_parser = subparsers.add_parser("report", help="Generate analytics report from SQLite")
    report_parser.add_argument(
        "--db",
        default="llm_calls.db",
        help="Path to SQLite database file (default: llm_calls.db)",
    )
    report_parser.add_argument(
        "--since",
        default=None,
        help="Filter metrics since a duration (e.g. 7d, 24h, 30d) or ISO timestamp",
    )
    report_parser.add_argument(
        "--format",
        choices=["table", "csv", "json", "html"],
        default="table",
        help="Output format: table (default), csv, json, or html",
    )
    report_parser.add_argument(
        "--out",
        default=None,
        help="Optional path to output file (for json, csv, or html)",
    )

    args = parser.parse_args(argv)

    if args.command == "report" or args.command is None:
        db_path = getattr(args, "db", "llm_calls.db")
        since = getattr(args, "since", None)
        fmt = getattr(args, "format", "table")
        out_file = getattr(args, "out", None)
        return run_report(db_path=db_path, since=since, output_format=fmt, out_file=out_file)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
