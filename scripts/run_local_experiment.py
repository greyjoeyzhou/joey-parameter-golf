#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMETER_GOLF_ROOT = REPO_ROOT / "parameter-golf"
RUNS_ROOT = REPO_ROOT / "runs"
WSL_NVIDIA_DIR = Path("/usr/lib/wsl/lib")
LOCAL_SHIMS_DIR = REPO_ROOT / "local_shims"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive and run a local parameter-golf experiment")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-script", required=True)
    parser.add_argument("--snapshot", action="append", default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--torchrun", default=str(PARAMETER_GOLF_ROOT / ".venv312x" / "bin" / "torchrun"))
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--shim-flash-attn", action="store_true")
    parser.add_argument("--copy-logfile", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    return parser.parse_args()


def parse_env(items: list[str]) -> dict[str, str]:
    env = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def ensure_path_tools(env: dict[str, str], shim_flash_attn: bool) -> None:
    path_entries = []
    if WSL_NVIDIA_DIR.joinpath("nvidia-smi").exists() and shutil.which("nvidia-smi", path=env.get("PATH")) is None:
        path_entries.append(str(WSL_NVIDIA_DIR))
    if path_entries:
        env["PATH"] = ":".join(path_entries + [env["PATH"]])

    if shim_flash_attn:
        current = env.get("PYTHONPATH", "")
        parts = [str(LOCAL_SHIMS_DIR)]
        if current:
            parts.append(current)
        env["PYTHONPATH"] = ":".join(parts)


def snapshot_files(run_dir: Path, paths: list[Path]) -> list[Path]:
    copied = []
    for path in paths:
        dest = run_dir / path.name
        shutil.copy2(path, dest)
        copied.append(dest)
    return copied


def write_command(run_dir: Path, env_updates: dict[str, str], command: list[str], workdir: Path) -> None:
    lines = []
    for key in sorted(env_updates):
        lines.append(f"export {key}={env_updates[key]}")
    lines.append(f"cd {workdir}")
    lines.append(" ".join(command))
    (run_dir / "command.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_metrics(log_text: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    patterns = {
        "prequant_val_bpb": r"pre-quant(?:ization)?[^\n]*val_bpb[:=]\s*([0-9.]+)",
        "quantized_val_bpb": r"quantized[^\n]*val_bpb[:=]\s*([0-9.]+)",
        "final_val_bpb": r"val_bpb[:=]\s*([0-9.]+)",
        "peak_mem_mib": r"peak memory allocated:\s*([0-9]+) MiB",
        "bytes_total": r"Total submission bytes:\s*([0-9]+)",
        "train_time_ms": r"train_time:([0-9.]+)ms",
        "step_avg_ms": r"step_avg:([0-9.]+)ms",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, log_text)
        if matches:
            value = matches[-1]
            metrics[key] = float(value) if "." in value else int(value)
    return metrics


def write_stub_submission(run_dir: Path, run_id: str, metrics: dict[str, object]) -> None:
    val_bpb = metrics.get("quantized_val_bpb") or metrics.get("prequant_val_bpb") or metrics.get("final_val_bpb") or 0.0
    bytes_total = int(metrics.get("bytes_total", 0))
    submission = {
        "author": "na",
        "github_id": "na",
        "name": run_id,
        "blurb": "Local 1x5090 experiment run",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "val_loss": 0.0,
        "val_bpb": float(val_bpb),
        "bytes_total": bytes_total,
        "bytes_code": 0,
    }
    (run_dir / "submission.json").write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")


def write_readme(run_dir: Path, run_id: str, source_script: Path, snapshots: list[Path], env_updates: dict[str, str], metrics: dict[str, object], returncode: int, elapsed_s: float) -> None:
    lines = [
        f"# {run_id}",
        "",
        "## Source",
        "",
        f"- source script: `{source_script}`",
        f"- archived files: {', '.join(f'`{p.name}`' for p in snapshots)}",
        "",
        "## Environment",
        "",
    ]
    for key in sorted(env_updates):
        lines.append(f"- `{key}={env_updates[key]}`")
    lines += [
        "",
        "## Result",
        "",
        f"- return code: `{returncode}`",
        f"- wallclock_s: `{elapsed_s:.1f}`",
    ]
    for key in sorted(metrics):
        lines.append(f"- {key}: `{metrics[key]}`")
    lines += [
        "",
        "## Files",
        "",
        "- `train.log`",
        "- `command.sh`",
        "- `submission.json`",
    ]
    for p in snapshots:
        lines.append(f"- `{p.name}`")
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = RUNS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    source_script = Path(args.source_script)
    if not source_script.is_absolute():
        source_script = (REPO_ROOT / source_script).resolve()
    snapshot_paths = [source_script]
    for item in args.snapshot:
        path = Path(item)
        if not path.is_absolute():
            path = (REPO_ROOT / item).resolve()
        snapshot_paths.append(path)

    snapshots = snapshot_files(run_dir, snapshot_paths)

    env = os.environ.copy()
    env_updates = parse_env(args.env)
    env_updates.setdefault("RUN_ID", args.run_id)
    env.update(env_updates)
    ensure_path_tools(env, args.shim_flash_attn)
    env_updates["PATH"] = env["PATH"]
    if "PYTHONPATH" in env:
        env_updates["PYTHONPATH"] = env["PYTHONPATH"]

    log_path = run_dir / "train.log"
    command = [args.torchrun, "--standalone", f"--nproc_per_node={args.nproc_per_node}", str(source_script)]
    write_command(run_dir, env_updates, command, PARAMETER_GOLF_ROOT)

    start = time.time()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=PARAMETER_GOLF_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=args.timeout_seconds if args.timeout_seconds > 0 else None)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
            returncode = 124
            print(f"[run_local_experiment] timeout after {args.timeout_seconds}s", file=log_file)
    elapsed_s = time.time() - start

    if args.copy_logfile:
        source_log = PARAMETER_GOLF_ROOT / "logs" / f"{args.run_id}.txt"
        if source_log.exists():
            shutil.copy2(source_log, run_dir / source_log.name)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = parse_metrics(log_text)
    if timed_out:
        metrics["timeout_seconds"] = args.timeout_seconds
    write_stub_submission(run_dir, args.run_id, metrics)
    write_readme(run_dir, args.run_id, source_script, snapshots, env_updates, metrics, returncode, elapsed_s)

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
