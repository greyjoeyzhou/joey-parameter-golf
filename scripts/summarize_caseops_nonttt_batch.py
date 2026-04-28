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
    parser = argparse.ArgumentParser(description="Summarize a CaseOps non-TTT technique batch")
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


def reference_metrics() -> tuple[float, float] | None:
    reference_run = RUNS_ROOT / "2026-04-25_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid" / "train.log"
    metrics = parse_metrics(reference_run)
    if "prequant_val_bpb" in metrics and "quantized_val_bpb" in metrics:
        return float(metrics["prequant_val_bpb"]), float(metrics["quantized_val_bpb"])
    return None


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
                "stack": row["stack"],
                "status": classify_status(run_id, status_map),
                "metrics": metrics,
            }
        )

    ok_runs = [r for r in runs if r["status"] == "ok" and "quantized_val_bpb" in r["metrics"]]
    ok_runs.sort(key=lambda r: float(r["metrics"]["quantized_val_bpb"]))
    reference = reference_metrics()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Local CaseOps Non-TTT Technique Batch",
        "",
        "This note is updated automatically while the CaseOps non-TTT technique batch is running.",
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
    ]

    if reference is not None:
        lines += [
            "## Reference",
            "",
            "Current best 30-minute non-QAT CaseOps reference:",
            f"- `2026-04-25_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid` prequant `val_bpb`: `{reference[0]:.8f}`",
            f"- `2026-04-25_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid` quantized `val_bpb`: `{reference[1]:.8f}`",
            "",
        ]

    lines += [
        "## Planned Runs",
        "",
    ]
    for run in runs:
        lines.append(f"- `{run['run_id']}`: status `{run['status']}`, stack `{run['stack']}`")
    lines.append("")

    lines += [
        "## Ranking",
        "",
    ]
    if ok_runs:
        lines.append("| rank | run | stack | prequant_bpb | quantized_bpb | gap | steps |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for idx, run in enumerate(ok_runs, start=1):
            metrics = run["metrics"]
            lines.append(
                "| "
                f"{idx} | `{run['run_id']}` | `{run['stack']}` | "
                f"{format_float(metrics.get('prequant_val_bpb'))} | "
                f"{format_float(metrics.get('quantized_val_bpb'))} | "
                f"{format_float(metrics.get('quant_gap_bpb'))} | "
                f"{format_float(metrics.get('step'))} |"
            )
    else:
        lines.append("No completed runs yet.")
    lines.append("")

    if ok_runs:
        best = ok_runs[0]
        best_metrics = best["metrics"]
        lines += [
            "## Current Takeaway",
            "",
            f"- current batch leader: `{best['run_id']}`",
            f"- stack: `{best['stack']}`",
            f"- quantized `val_bpb`: `{float(best_metrics['quantized_val_bpb']):.8f}`",
            f"- prequant `val_bpb`: `{float(best_metrics['prequant_val_bpb']):.8f}`",
            f"- quantization gap: `{float(best_metrics['quant_gap_bpb']):.8f}`",
        ]
        if reference is not None:
            delta = float(best_metrics["quantized_val_bpb"]) - reference[1]
            lines.append(f"- delta vs current 30m CaseOps reference: `{delta:+.8f} bpb`")
        lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
