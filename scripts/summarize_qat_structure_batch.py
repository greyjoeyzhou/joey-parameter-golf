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
    parser = argparse.ArgumentParser(description="Summarize a local QAT structure batch")
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("run_ids", nargs="+")
    return parser.parse_args()


def parse_status_file(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            run_id = row.get("run_id")
            if run_id:
                rows[run_id] = row
    return rows


def parse_command_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    export_re = re.compile(r"^export ([A-Z0-9_]+)=(.*)$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = export_re.match(line)
        if match:
            env[match.group(1)] = match.group(2)
    return env


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
        if not matches:
            continue
        value = matches[-1]
        metrics[key] = float(value) if "." in value else int(value)
    if "prequant_val_bpb" in metrics and "quantized_val_bpb" in metrics:
        metrics["quant_gap_bpb"] = float(metrics["quantized_val_bpb"]) - float(metrics["prequant_val_bpb"])
    return metrics


def layer_label(env: dict[str, str]) -> str:
    start = env.get("QAT_LAYER_START", "0")
    end = env.get("QAT_LAYER_END")
    if end is None:
        return f"{start}+"
    return f"{start}-{end}"


def infer_target(run_id: str) -> str:
    if "_Attn" in run_id:
        return "attn"
    if "_MLP" in run_id:
        return "mlp"
    return "all"


def infer_layers(run_id: str) -> str:
    if "_L7plus" in run_id:
        return "7-10"
    if "_L8plus" in run_id:
        return "8-10"
    if "_EarlyOnly" in run_id:
        return "0-6"
    return "0-10"


def classify_run(run_id: str, status_rows: dict[str, dict[str, str]]) -> str:
    row = status_rows.get(run_id)
    if row is not None:
        return row.get("status", "unknown")
    run_dir = RUNS_ROOT / run_id
    if (run_dir / "train.log").exists() and not (run_dir / "README.md").exists():
        return "running"
    if run_dir.exists():
        return "archived"
    return "pending"


def summarize_reference(run_id: str) -> tuple[float, float] | None:
    run_dir = RUNS_ROOT / run_id
    metrics = parse_metrics(run_dir / "train.log")
    if "prequant_val_bpb" in metrics and "quantized_val_bpb" in metrics:
        return float(metrics["prequant_val_bpb"]), float(metrics["quantized_val_bpb"])
    return None


def format_float(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.8f}"


def main() -> int:
    args = parse_args()
    status_file = Path(args.status_file)
    output = Path(args.output)
    status_rows = parse_status_file(status_file)

    runs: list[dict[str, object]] = []
    for run_id in args.run_ids:
        run_dir = RUNS_ROOT / run_id
        env = parse_command_env(run_dir / "command.sh")
        metrics = parse_metrics(run_dir / "train.log")
        runs.append(
            {
                "run_id": run_id,
                "status": classify_run(run_id, status_rows),
                "target": env.get("QAT_TARGET", infer_target(run_id)),
                "layers": layer_label(env) if env else infer_layers(run_id),
                "metrics": metrics,
            }
        )

    ok_runs = [r for r in runs if r["status"] == "ok" and "quantized_val_bpb" in r["metrics"]]
    ok_runs.sort(key=lambda r: float(r["metrics"]["quantized_val_bpb"]))

    reference = summarize_reference("2026-04-25_CasefoldV2_ParResid_QAT10_QKGain40")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines = [
        "# Local QAT Structure Results",
        "",
        "This note is updated automatically while the QAT structure batch is running.",
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
            "Current best pre-structure QAT reference from the earlier batch:",
            f"- `2026-04-25_CasefoldV2_ParResid_QAT10_QKGain40` prequant `val_bpb`: `{reference[0]:.8f}`",
            f"- `2026-04-25_CasefoldV2_ParResid_QAT10_QKGain40` quantized `val_bpb`: `{reference[1]:.8f}`",
            "",
        ]

    lines += [
        "## Planned Runs",
        "",
    ]
    for run in runs:
        lines.append(
            f"- `{run['run_id']}`: status `{run['status']}`, target `{run['target']}`, layers `{run['layers']}`"
        )
    lines.append("")

    lines += [
        "## Ranking",
        "",
    ]
    if ok_runs:
        lines.append("| rank | run | target | layers | prequant_bpb | quantized_bpb | gap |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
        for idx, run in enumerate(ok_runs, start=1):
            metrics = run["metrics"]
            lines.append(
                "| "
                f"{idx} | `{run['run_id']}` | `{run['target']}` | `{run['layers']}` | "
                f"{format_float(metrics.get('prequant_val_bpb'))} | "
                f"{format_float(metrics.get('quantized_val_bpb'))} | "
                f"{format_float(metrics.get('quant_gap_bpb'))} |"
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
            f"- quantized `val_bpb`: `{float(best_metrics['quantized_val_bpb']):.8f}`",
            f"- prequant `val_bpb`: `{float(best_metrics['prequant_val_bpb']):.8f}`",
            f"- quantization gap: `{float(best_metrics['quant_gap_bpb']):.8f}`",
        ]
        if reference is not None:
            delta = float(best_metrics["quantized_val_bpb"]) - reference[1]
            lines.append(f"- delta vs `QAT10_QKGain40`: `{delta:+.8f} bpb`")
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
