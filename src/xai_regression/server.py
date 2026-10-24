"""Local-only HTTP server for generated dashboard reports."""

from __future__ import annotations

import json
from email.parser import BytesParser
from email.policy import default
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from xai_regression.upload_comparison import (
    compare_upload_payloads,
    summarize_upload_models,
    validate_upload_models,
)
from xai_regression.regression_comparison import (
    compare_regression_upload_payloads,
    summarize_regression_upload_models,
)


REPORT_ROUTE = "/data/report.json"
BENCHMARK_ROUTE = "/data/benchmarks.json"
PORTFOLIO_ROUTE = "/data/portfolio.json"
ROBUSTNESS_ROUTE = "/data/robustness.json"
CLASSIFICATION_COMPARE_ROUTE = "/api/classification/compare"
CLASSIFICATION_OVERVIEW_ROUTE = "/api/classification/overview"
CLASSIFICATION_VALIDATE_ROUTE = "/api/classification/validate"
REGRESSION_COMPARE_ROUTE = "/api/regression/compare"
REGRESSION_OVERVIEW_ROUTE = "/api/regression/overview"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def parse_multipart_form(content_type: str, payload: bytes) -> dict[str, tuple[str, bytes]]:
    """Parse a bounded multipart request into named in-memory file fields."""
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("Comparison requests must use multipart/form-data.")
    message = BytesParser(policy=default).parsebytes(
        (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8")
        + payload
    )
    if not message.is_multipart():
        raise ValueError("Could not parse multipart upload.")
    fields: dict[str, tuple[str, bytes]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename() or name
        fields[str(name)] = (str(filename), part.get_payload(decode=True) or b"")
    return fields


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


def validate_portfolio_file(path: str | Path | None) -> Path | None:
    """Validate the optional cross-dataset summary."""
    if path is None:
        return None
    portfolio_path = Path(path).resolve()
    try:
        report = json.loads(portfolio_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Dashboard cross-dataset report not found: {portfolio_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Dashboard cross-dataset report is not valid JSON: {portfolio_path}"
        ) from error
    if not isinstance(report.get("entries"), list) or not isinstance(
        report.get("coverage"), dict
    ):
        raise ValueError(
            "Dashboard cross-dataset report must contain entries and coverage."
        )
    return portfolio_path


def validate_robustness_file(path: str | Path | None) -> Path | None:
    """Validate the optional split-and-probe robustness summary."""
    if path is None:
        return None
    robustness_path = Path(path).resolve()
    try:
        report = json.loads(robustness_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Dashboard robustness report not found: {robustness_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Dashboard robustness report is not valid JSON: {robustness_path}"
        ) from error
    if not isinstance(report.get("dataset_summaries"), list) or not isinstance(
        report.get("split_results"), list
    ):
        raise ValueError(
            "Dashboard robustness report must contain dataset_summaries and "
            "split_results lists."
        )
    return robustness_path


def _handler_class(
    report_path: Path,
    benchmark_paths: list[Path],
    portfolio_path: Path | None,
    robustness_path: Path | None,
):
    class DashboardRequestHandler(SimpleHTTPRequestHandler):
        def _send_json(self, status: int, value: object) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):  # noqa: N802 - inherited HTTP method name
            route = urlparse(self.path).path
            if route not in {
                CLASSIFICATION_COMPARE_ROUTE,
                CLASSIFICATION_OVERVIEW_ROUTE,
                CLASSIFICATION_VALIDATE_ROUTE,
                REGRESSION_COMPARE_ROUTE,
                REGRESSION_OVERVIEW_ROUTE,
            }:
                self._send_json(404, {"error": "Unknown API route."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1:
                    raise ValueError("The upload request is empty.")
                if length > MAX_UPLOAD_BYTES:
                    raise ValueError("The upload exceeds the 100 MB local limit.")
                fields = parse_multipart_form(
                    self.headers.get("Content-Type", ""),
                    self.rfile.read(length),
                )
                required = {"dataset", "manifest"}
                if route in {CLASSIFICATION_COMPARE_ROUTE, REGRESSION_COMPARE_ROUTE}:
                    required.update({"model_v1", "model_v2"})
                missing = sorted(required - fields.keys())
                if missing:
                    raise ValueError(
                        "Missing upload fields: " + ", ".join(missing)
                    )
                if route in {CLASSIFICATION_OVERVIEW_ROUTE, CLASSIFICATION_VALIDATE_ROUTE, REGRESSION_OVERVIEW_ROUTE}:
                    model_fields = [
                        fields[name]
                        for name in sorted(fields)
                        if name.startswith("model_") and name[6:].isdigit()
                    ]
                    operation = validate_upload_models if route == CLASSIFICATION_VALIDATE_ROUTE else (
                        summarize_regression_upload_models
                        if route == REGRESSION_OVERVIEW_ROUTE
                        else summarize_upload_models
                    )
                    result = operation(
                        models=model_fields,
                        dataset=fields["dataset"],
                        manifest=fields["manifest"][1],
                    )
                elif route == CLASSIFICATION_COMPARE_ROUTE:
                    image_field = fields.get("images")
                    if image_field is not None and not image_field[1]:
                        image_field = None
                    result = compare_upload_payloads(
                        model_v1=fields["model_v1"],
                        model_v2=fields["model_v2"],
                        dataset=fields["dataset"],
                        manifest=fields["manifest"][1],
                        images=image_field,
                    )
                else:
                    result = compare_regression_upload_payloads(
                        model_v1=fields["model_v1"],
                        model_v2=fields["model_v2"],
                        dataset=fields["dataset"],
                        manifest=fields["manifest"][1],
                    )
                self._send_json(200, result)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
            except Exception:
                self._send_json(
                    500,
                    {"error": "The local comparison failed unexpectedly."},
                )

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
            if route == PORTFOLIO_ROUTE and portfolio_path is not None:
                payload = portfolio_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == ROBUSTNESS_ROUTE and robustness_path is not None:
                payload = robustness_path.read_bytes()
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
    portfolio_path: str | Path | None = None,
    robustness_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve the dashboard and expose the selected report at a fixed route."""
    report = validate_report_file(report_path)
    benchmarks = validate_benchmark_files(benchmark_paths)
    portfolio = validate_portfolio_file(portfolio_path)
    robustness = validate_robustness_file(robustness_path)
    dashboard = Path(dashboard_dir).resolve()
    if not (dashboard / "index.html").is_file():
        raise FileNotFoundError(f"Dashboard directory is invalid: {dashboard}")
    handler = partial(
        _handler_class(report, benchmarks, portfolio, robustness),
        directory=str(dashboard),
    )
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
    if portfolio is not None:
        print(f"Serving cross-dataset report: {portfolio}", flush=True)
    if robustness is not None:
        print(f"Serving split-robustness report: {robustness}", flush=True)
    print("Press Control+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.", flush=True)
    finally:
        server.server_close()
