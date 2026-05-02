#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a proxy CaseOps TTT batch")
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_metrics(path: Path) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    if not path.exists():
        return metrics
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "prequant_val_bpb": r"pre-quant(?:ization)?[^\n]*val_bpb[:=]\s*([0-9.]+)",
        "quantized_val_bpb": r"quantized[^\n]*val_bpb[:=]\s*([0-9.]+)",
        "ttt_val_bpb": r"quantized_ttt val_loss:[0-9.]+ val_bpb:([0-9.]+)",
        "peak_mem_mib": r"peak memory allocated:\s*([0-9]+) MiB",
        "bytes_total": r"Total submission size quantized\+brotli:\s*([0-9]+)",
        "train_time_ms": r"stopping_early:[^\n]*train_time:\s*([0-9.]+)ms",
        "step": r"stopping_early:[^\n]*step:\s*([0-9]+)/",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            metrics[key] = float(value) if "." in value else int(value)
    if "prequant_val_bpb" in metrics and "quantized_val_bpb" in metrics:
        metrics["quant_gap_bpb"] = float(metrics["quantized_val_bpb"]) - float(metrics["prequant_val_bpb"])
    return metrics


def classify_status(run_id: str, status_map: dict[str, str]) -> str:
    if run_id in status_map:
        return status_map[run_id]
    run_dir = RUNS_ROOT / run_id
    if (run_dir / "train.log").exists() and not (run_dir / "README.md").exists():
        return "running"
    if run_dir.exists():
        return "archived"
    return "pending"


def format_float(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.8f}"


def main() -> int:
    args = parse_args()
    metadata_rows = parse_tsv(Path(args.metadata_file))
    status_rows = parse_tsv(Path(args.status_file))
    status_map = {row["run_id"]: row["status"] for row in status_rows if row.get("run_id")}

    runs: list[dict[str, object]] = []
    for row in metadata_rows:
        run_id = row["run_id"]
        run_dir = RUNS_ROOT / run_id
        metrics = parse_metrics(run_dir / "train.log")
        runs.append(
            {
                "run_id": run_id,
                "variant": row["variant"],
                "status": classify_status(run_id, status_map),
                "metrics": metrics,
            }
        )

    ok_runs = [r for r in runs if r["status"] == "ok" and "quantized_val_bpb" in r["metrics"]]
    ok_runs.sort(key=lambda r: float(r["metrics"].get("ttt_val_bpb", r["metrics"]["quantized_val_bpb"])))

    by_variant = {str(r["variant"]): r for r in runs}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines = [
        "# Local CaseOps Legal TTT Proxy Batch",
        "",
        "This note is updated automatically while the proxy CaseOps legal-TTT batch is running.",
        "",
        f"Last updated: `{now}`",
        "",
        "## Batch Status",
        "",
        f"- planned runs: `{len(runs)}`",
        f"- completed ok: `{sum(1 for r in runs if r['status'] == 'ok')}`",
        f"- running: `{sum(1 for r in runs if r['status'] == 'running')}`",
        f"- pending: `{sum(1 for r in runs if r['status'] == 'pending')}`",
        f"- failed/nonzero: `{sum(1 for r in runs if str(r['status']).startswith('rc='))}`",
        "",
        "## Planned Runs",
        "",
    ]
    for run in runs:
        lines.append(f"- `{run['run_id']}`: status `{run['status']}`, variant `{run['variant']}`")
    lines.append("")

    lines += [
        "## Ranking",
        "",
    ]
    if ok_runs:
        lines.append("| rank | run | variant | prequant_bpb | quantized_bpb | ttt_bpb | steps |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for idx, run in enumerate(ok_runs, start=1):
            metrics = run["metrics"]
            lines.append(
                "| "
                f"{idx} | `{run['run_id']}` | `{run['variant']}` | "
                f"{format_float(metrics.get('prequant_val_bpb'))} | "
                f"{format_float(metrics.get('quantized_val_bpb'))} | "
                f"{format_float(metrics.get('ttt_val_bpb'))} | "
                f"{format_float(metrics.get('step'))} |"
            )
    else:
        lines.append("No completed runs yet.")
    lines.append("")

    control = by_variant.get("control")
    ttt = by_variant.get("legal_ttt")
    if control and ttt:
        c_metrics = control["metrics"]
        t_metrics = ttt["metrics"]
        lines += [
            "## Proxy Delta",
            "",
        ]
        if "quantized_val_bpb" in c_metrics and "ttt_val_bpb" in t_metrics:
            delta = float(t_metrics["ttt_val_bpb"]) - float(c_metrics["quantized_val_bpb"])
            lines.append(f"- legal TTT delta vs same-subset control: `{delta:+.8f} bpb`")
        else:
            lines.append("- delta not available yet")
        lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
