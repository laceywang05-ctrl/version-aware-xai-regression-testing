"""Local-only HTTP server for generated dashboard reports."""

from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


REPORT_ROUTE = "/data/report.json"
BENCHMARK_ROUTE = "/data/benchmarks.json"


def validate_report_file(path: str | Path) -> Path:
    """Validate that a file has the unified dashboard report shape."""
    report_path = Path(path).resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Dashboard report not found: {report_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Dashboard report is not valid JSON: {report_path}") from error
    if not isinstance(report.get("methods"), list) or not isinstance(
        report.get("metrics"), list
    ):
        raise ValueError("Dashboard report must contain methods and metrics lists.")
    return report_path


def validate_benchmark_files(paths: list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    """Validate benchmark summaries and return their resolved paths."""
    validated: list[Path] = []
    for path in paths:
        benchmark_path = Path(path).resolve()
        try:
            report = json.loads(benchmark_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Dashboard benchmark report not found: {benchmark_path}"
            ) from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Dashboard benchmark report is not valid JSON: {benchmark_path}"
            ) from error
        if not isinstance(report.get("scenario_results"), list) or not isinstance(
            report.get("paired_control_comparisons"), list
        ):
            raise ValueError(
                "Dashboard benchmark report must contain scenario_results and "
                "paired_control_comparisons lists."
            )
        validated.append(benchmark_path)
    return validated


def _handler_class(report_path: Path, benchmark_paths: list[Path]):
    class DashboardRequestHandler(SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - inherited HTTP method name
            route = urlparse(self.path).path
            if route == REPORT_ROUTE:
                payload = report_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == BENCHMARK_ROUTE:
                reports = [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in benchmark_paths
                ]
                payload = json.dumps(
                    {"benchmarks": reports}, ensure_ascii=False
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

    return DashboardRequestHandler


def serve_dashboard(
    *,
    report_path: str | Path,
    dashboard_dir: str | Path,
    benchmark_paths: list[str | Path] | tuple[str | Path, ...] = (),
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve the dashboard and expose the selected report at a fixed route."""
    report = validate_report_file(report_path)
    benchmarks = validate_benchmark_files(benchmark_paths)
    dashboard = Path(dashboard_dir).resolve()
    if not (dashboard / "index.html").is_file():
        raise FileNotFoundError(f"Dashboard directory is invalid: {dashboard}")
    handler = partial(_handler_class(report, benchmarks), directory=str(dashboard))
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as error:
        raise RuntimeError(
            f"Could not start dashboard at http://{host}:{port}: {error}"
        ) from error
    print(f"Dashboard: http://{host}:{port}", flush=True)
    print(f"Serving report: {report}", flush=True)
    if benchmarks:
        print(f"Serving benchmark reports: {len(benchmarks)}", flush=True)
    print("Press Control+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.", flush=True)
    finally:
        server.server_close()
