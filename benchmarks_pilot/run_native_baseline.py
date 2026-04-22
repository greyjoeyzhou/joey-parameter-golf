#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMETER_GOLF_ROOT = REPO_ROOT / "parameter-golf"
LOCAL_BIN_DIR = REPO_ROOT / "local_bin"

REF_4090_EGPU_BASELINE_TOK_PER_SEC = 598_000.0
REF_8X_H100_TOK_PER_SEC = 1_350_000.0


@dataclass(frozen=True)
class VariantConfig:
    data_path: str
    tokenizer_path: str
    vocab_size: int


VARIANT_CONFIGS: dict[str, VariantConfig] = {
    "sp1024": VariantConfig(
        data_path="./data/datasets/fineweb10B_sp1024",
        tokenizer_path="./data/tokenizers/fineweb_1024_bpe.model",
        vocab_size=1024,
    ),
    "sp4096": VariantConfig(
        data_path="./data/datasets/fineweb10B_sp4096",
        tokenizer_path="./data/tokenizers/fineweb_4096_bpe.model",
        vocab_size=4096,
    ),
    "sp8192": VariantConfig(
        data_path="./data/datasets/fineweb10B_sp8192",
        tokenizer_path="./data/tokenizers/fineweb_8192_bpe.model",
        vocab_size=8192,
    ),
}

TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) train_loss:(?P<train_loss>\S+) "
    r"train_time:(?P<train_time_ms>\d+(?:\.\d+)?)ms step_avg:(?P<step_avg_ms>\d+(?:\.\d+)?)ms"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) val_loss:(?P<val_loss>\S+) val_bpb:(?P<val_bpb>\S+) "
    r"train_time:(?P<train_time_ms>\d+(?:\.\d+)?)ms step_avg:(?P<step_avg_ms>\d+(?:\.\d+)?)ms"
)
STOP_RE = re.compile(
    r"stopping_early: wallclock_cap train_time:(?P<train_time_ms>\d+(?:\.\d+)?)ms "
    r"step:(?P<step>\d+)/(?P<iterations>\d+)"
)
PEAK_RE = re.compile(
    r"peak memory allocated: (?P<allocated_mib>\d+) MiB reserved: (?P<reserved_mib>\d+) MiB"
)
FINAL_INT8_RE = re.compile(
    r"final_int8_zlib_roundtrip(?:_exact)? val_loss:(?P<val_loss>\S+) val_bpb:(?P<val_bpb>\S+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a native local hardware benchmark with parameter-golf/train_gpt.py")
    parser.add_argument("--variant", default="sp1024", choices=sorted(VARIANT_CONFIGS))
    parser.add_argument("--run-id", default=f"{date.today().isoformat()}_5090NativeBaseline")
    parser.add_argument("--output-dir", default=None, help="Defaults to benchmarks_pilot/<RUN_ID>")
    parser.add_argument("--torchrun", default=str(PARAMETER_GOLF_ROOT / ".venv312x" / "bin" / "torchrun"))
    parser.add_argument("--trainer", default="train_gpt.py")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-wallclock-seconds", type=float, default=0.0)
    parser.add_argument("--val-loss-every", type=int, default=2000)
    parser.add_argument("--train-log-every", type=int, default=50)
    parser.add_argument("--train-batch-tokens", type=int, default=524_288)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steady-state-min-step", type=int, default=100)
    parser.add_argument("--eager", action="store_true", help="Set TORCHDYNAMO_DISABLE=1 for an eager-mode baseline")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).resolve()
    return (REPO_ROOT / "benchmarks_pilot" / args.run_id).resolve()


def require_variant_artifacts(variant: str) -> VariantConfig:
    config = VARIANT_CONFIGS[variant]
    dataset_dir = PARAMETER_GOLF_ROOT / config.data_path.removeprefix("./")
    tokenizer_path = PARAMETER_GOLF_ROOT / config.tokenizer_path.removeprefix("./")
    if dataset_dir.exists() and tokenizer_path.exists():
        return config

    raise SystemExit(
        f"Missing local artifacts for {variant}. Expected dataset at {dataset_dir} and tokenizer at {tokenizer_path}.\n"
        f"See benchmarks_pilot/RUNBOOK.md for the current bootstrap steps.\n"
        f"Note: the local manifest in this repo currently drifts from the public sp1024 layout, so the old manifest-driven download command is not the right fallback."
    )


def build_env(args: argparse.Namespace, variant: VariantConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": args.run_id,
            "DATA_PATH": variant.data_path,
            "TOKENIZER_PATH": variant.tokenizer_path,
            "VOCAB_SIZE": str(variant.vocab_size),
            "ITERATIONS": str(args.iterations),
            "WARMUP_STEPS": str(args.warmup_steps),
            "MAX_WALLCLOCK_SECONDS": str(args.max_wallclock_seconds),
            "VAL_LOSS_EVERY": str(args.val_loss_every),
            "TRAIN_LOG_EVERY": str(args.train_log_every),
            "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
            "TRAIN_SEQ_LEN": str(args.train_seq_len),
            "SEED": str(args.seed),
            "TORCHINDUCTOR_MIX_ORDER_REDUCTION": env.get("TORCHINDUCTOR_MIX_ORDER_REDUCTION", "0"),
        }
    )
    if args.eager:
        env["TORCHDYNAMO_DISABLE"] = "1"
    if LOCAL_BIN_DIR.exists():
        env["PATH"] = f"{LOCAL_BIN_DIR}:{env['PATH']}"
    return env


def write_command_record(output_dir: Path, args: argparse.Namespace, variant: VariantConfig, env: dict[str, str]) -> None:
    command_lines = [
        f"# variant: {args.variant}",
        f"# mode: {'eager' if args.eager else 'compiled'}",
    ]
    for key in (
        "RUN_ID",
        "DATA_PATH",
        "TOKENIZER_PATH",
        "VOCAB_SIZE",
        "ITERATIONS",
        "WARMUP_STEPS",
        "MAX_WALLCLOCK_SECONDS",
        "VAL_LOSS_EVERY",
        "TRAIN_LOG_EVERY",
        "TRAIN_BATCH_TOKENS",
        "TRAIN_SEQ_LEN",
        "SEED",
        "TORCHINDUCTOR_MIX_ORDER_REDUCTION",
        "TORCHDYNAMO_DISABLE",
    ):
        if key in env:
            command_lines.append(f"export {key}={env[key]}")
    command_lines.append(f"export PATH={env['PATH']}")
    command_lines.append(f"cd {PARAMETER_GOLF_ROOT}")
    command_lines.append(f"{args.torchrun} --standalone --nproc_per_node=1 {args.trainer}")
    (output_dir / "command.sh").write_text("\n".join(command_lines) + "\n", encoding="utf-8")


def run_benchmark(args: argparse.Namespace, output_dir: Path, env: dict[str, str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_command_record(output_dir, args, VARIANT_CONFIGS[args.variant], env)
    stdout_path = output_dir / "stdout.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_file:
        subprocess.run(
            [args.torchrun, "--standalone", "--nproc_per_node=1", args.trainer],
            cwd=PARAMETER_GOLF_ROOT,
            env=env,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return stdout_path


def parse_log(log_path: Path) -> dict[str, object]:
    train_events: list[dict[str, float]] = []
    latest_val: dict[str, float] | None = None
    latest_quantized: dict[str, float] | None = None
    stop_event: dict[str, float] | None = None
    peak_memory: dict[str, int] | None = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = TRAIN_RE.search(line)
        if match:
            try:
                train_events.append(
                    {
                        "step": float(match.group("step")),
                        "iterations": float(match.group("iterations")),
                        "train_loss": float(match.group("train_loss")),
                        "train_time_ms": float(match.group("train_time_ms")),
                        "step_avg_ms": float(match.group("step_avg_ms")),
                    }
                )
            except ValueError:
                pass
            continue

        match = VAL_RE.search(line)
        if match:
            try:
                latest_val = {
                    "step": float(match.group("step")),
                    "iterations": float(match.group("iterations")),
                    "val_loss": float(match.group("val_loss")),
                    "val_bpb": float(match.group("val_bpb")),
                    "train_time_ms": float(match.group("train_time_ms")),
                    "step_avg_ms": float(match.group("step_avg_ms")),
                }
            except ValueError:
                pass
            continue

        match = STOP_RE.search(line)
        if match:
            stop_event = {
                "step": float(match.group("step")),
                "iterations": float(match.group("iterations")),
                "train_time_ms": float(match.group("train_time_ms")),
            }
            continue

        match = PEAK_RE.search(line)
        if match:
            peak_memory = {
                "allocated_mib": int(match.group("allocated_mib")),
                "reserved_mib": int(match.group("reserved_mib")),
            }
            continue

        match = FINAL_INT8_RE.search(line)
        if match:
            try:
                latest_quantized = {
                    "val_loss": float(match.group("val_loss")),
                    "val_bpb": float(match.group("val_bpb")),
                }
            except ValueError:
                pass

    return {
        "train_events": train_events,
        "latest_val": latest_val,
        "latest_quantized": latest_quantized,
        "stop_event": stop_event,
        "peak_memory": peak_memory,
    }


def summarize(args: argparse.Namespace, parsed: dict[str, object]) -> dict[str, object]:
    train_events = parsed["train_events"]
    if not train_events:
        raise RuntimeError("No training events found in benchmark log")

    intervals_ms: list[float] = []
    for prev, curr in zip(train_events, train_events[1:], strict=False):
        delta_steps = curr["step"] - prev["step"]
        delta_time_ms = curr["train_time_ms"] - prev["train_time_ms"]
        if delta_steps <= 0:
            continue
        if curr["step"] < args.steady_state_min_step:
            continue
        intervals_ms.append(delta_time_ms / delta_steps)

    steady_state_step_ms = median(intervals_ms) if intervals_ms else None
    steady_state_tok_s = None
    if steady_state_step_ms is not None and steady_state_step_ms > 0:
        steady_state_tok_s = args.train_batch_tokens / (steady_state_step_ms / 1000.0)

    ref_tok_s = REF_4090_EGPU_BASELINE_TOK_PER_SEC
    ref_step_ms = args.train_batch_tokens / ref_tok_s * 1000.0
    speedup_vs_4090 = None
    percent_faster_vs_4090 = None
    percent_of_h100 = None
    if steady_state_tok_s is not None and steady_state_tok_s > 0:
        speedup_vs_4090 = steady_state_tok_s / ref_tok_s
        percent_faster_vs_4090 = (speedup_vs_4090 - 1.0) * 100.0
        percent_of_h100 = steady_state_tok_s / REF_8X_H100_TOK_PER_SEC * 100.0

    final_train = train_events[-1]
    latest_val = parsed["latest_val"]
    latest_quantized = parsed["latest_quantized"]
    stop_event = parsed["stop_event"]
    peak_memory = parsed["peak_memory"]

    return {
        "run_id": args.run_id,
        "variant": args.variant,
        "mode": "eager" if args.eager else "compiled",
        "trainer": args.trainer,
        "iterations": args.iterations,
        "warmup_steps": args.warmup_steps,
        "train_batch_tokens": args.train_batch_tokens,
        "train_seq_len": args.train_seq_len,
        "seed": args.seed,
        "steady_state_min_step": args.steady_state_min_step,
        "final_logged_step": int(final_train["step"]),
        "final_logged_train_time_ms": final_train["train_time_ms"],
        "final_logged_step_avg_ms": final_train["step_avg_ms"],
        "steady_state_step_ms_median": steady_state_step_ms,
        "steady_state_tok_per_sec_median": steady_state_tok_s,
        "speedup_vs_4090_reference": speedup_vs_4090,
        "percent_faster_vs_4090_reference": percent_faster_vs_4090,
        "4090_reference_tok_per_sec": ref_tok_s,
        "4090_reference_step_ms_equiv": ref_step_ms,
        "percent_of_8x_h100_reference": percent_of_h100,
        "latest_val": latest_val,
        "latest_quantized": latest_quantized,
        "stop_event": stop_event,
        "peak_memory": peak_memory,
    }


def write_summary(output_dir: Path, summary: dict[str, object]) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(summary: dict[str, object]) -> None:
    print(f"run_id: {summary['run_id']}")
    print(f"variant: {summary['variant']}")
    print(f"mode: {summary['mode']}")
    print(f"final_logged_step: {summary['final_logged_step']}")
    print(f"final_logged_train_time_ms: {summary['final_logged_train_time_ms']}")
    print(f"4090_reference_tok_per_sec: {summary['4090_reference_tok_per_sec']:.0f}")
    if summary["steady_state_step_ms_median"] is None:
        print("steady_state_step_ms_median: n/a")
        print("steady_state_tok_per_sec_median: n/a")
    else:
        print(f"steady_state_step_ms_median: {summary['steady_state_step_ms_median']:.2f}")
        print(f"steady_state_tok_per_sec_median: {summary['steady_state_tok_per_sec_median']:.0f}")
        print(f"speedup_vs_4090_reference: {summary['speedup_vs_4090_reference']:.3f}x")
        print(f"percent_faster_vs_4090_reference: {summary['percent_faster_vs_4090_reference']:.1f}%")
        print(f"percent_of_8x_h100_reference: {summary['percent_of_8x_h100_reference']:.1f}%")
    latest_val = summary["latest_val"]
    if latest_val is not None:
        print(f"latest_val_bpb: {latest_val['val_bpb']:.4f}")
    peak_memory = summary["peak_memory"]
    if peak_memory is not None:
        print(f"peak_memory_allocated_mib: {peak_memory['allocated_mib']}")


def main() -> int:
    args = parse_args()
    variant = require_variant_artifacts(args.variant)
    output_dir = resolve_output_dir(args)
    env = build_env(args, variant)
    stdout_path = run_benchmark(args, output_dir, env)
    summary = summarize(args, parse_log(stdout_path))
    write_summary(output_dir, summary)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
