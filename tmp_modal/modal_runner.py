"""
Modal runner for the best non-TTT CaseOps proxy-control setting.

Usage:
    modal run tmp_modal/modal_runner.py::upload_data
    modal run tmp_modal/modal_runner.py::smoke_test
    modal run tmp_modal/modal_runner.py::main --run-id 2026-04-29_CaseOps_QKGain50_WD090_ProxyControl_Modal8x
    modal run tmp_modal/modal_runner.py::download_run --run-id <run-id>
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
BEST_RUN_ID = "2026-04-28_CaseOps_QKGain50_WD090_ProxyControl_Valid2p5h"
BEST_RUN_DIR = REPO_ROOT / "runs" / BEST_RUN_ID
SIDECAR_SCRIPT = BEST_RUN_DIR / "train_gpt_decode_sidecar.py"
SOURCE_SCRIPT = BEST_RUN_DIR / "train_gpt_decode.py"

APP_NAME = "param-golf-caseops-nonttt"
DATA_VOLUME = "param-golf-data"      # FineWeb shards + tokenizer
ARTIFACT_VOLUME = "param-golf-runs"  # logs, checkpoints, submission blobs

DEFAULT_DATA_DIR = REPO_ROOT / "parameter-golf" / "data"
DEFAULT_VOCAB_SIZE = 8192

BASE_ENV = {
    "PYTHONUNBUFFERED": "1",
    "NCCL_DEBUG": "WARN",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "DATA_DIR": "/vol/data",
    "SOURCE_TRAIN_GPT": "/workspace/train_gpt_decode.py",
}

BEST_NONTTT_ENV = {
    "EMBED_WD": "0.090",
    "GPTQ_CALIBRATION_BATCHES": "1",
    "HESSIAN_CLIP_LAMBDA": "0.15",
    "MUON_WD": "0.090",
    "NUM_LOOPS": "0",
    "PARALLEL_RESIDUAL_START": "7",
    "QK_GAIN_INIT": "5.0",
    "SKIP_GATES_ENABLED": "1",
    "SLIDING_WINDOW_ENABLED": "0",
    "TRAIN_LOG_EVERY": "50",
    "TTT_ENABLED": "0",
    "VAL_LOSS_EVERY": "0",
    "VAL_TOKEN_LIMIT": "196608",
    "VOCAB_SIZE": str(DEFAULT_VOCAB_SIZE),
    "WARMUP_STEPS": "1",
}

# ---------------------------------------------------------------
# Image: CUDA devel + torch 2.9.1/cu128 + FA3 wheel
# ---------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.9.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "numpy",
        "sentencepiece",
        "brotli",
        "packaging",
        "ninja",
        "psutil",
    )
    # The archived source script imports `flash_attn_interface` directly.
    # The cu128 + torch2.9.1 FA3 wheels expose that module path.
    .pip_install(
        "flash_attn_3",
        extra_options="--no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/",
    )
    .env(BASE_ENV)
    .add_local_file(
        str(SIDECAR_SCRIPT),
        "/workspace/train_gpt_decode_sidecar.py",
        copy=False,
    )
    .add_local_file(
        str(SOURCE_SCRIPT),
        "/workspace/train_gpt_decode.py",
        copy=False,
    )
)

app = modal.App(APP_NAME, image=image)

data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
run_vol = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)


# ---------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------
def _default_run_id(tag: str) -> str:
    date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{date_prefix}_{tag}"


def _modal_cli_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "modal", *args]


def _prepare_run_dir(run_id: str) -> Path:
    run_dir = Path("/vol/runs") / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    os.chdir("/workspace")
    logs_link = Path("/workspace/logs")
    if logs_link.is_symlink() or logs_link.exists():
        if logs_link.is_dir() and not logs_link.is_symlink():
            shutil.rmtree(logs_link)
        else:
            logs_link.unlink()
    os.symlink(str(logs_dir), str(logs_link), target_is_directory=True)
    return run_dir


def _snapshot_run_files(run_dir: Path) -> None:
    shutil.copy2("/workspace/train_gpt_decode_sidecar.py", run_dir / "train_gpt_decode_sidecar.py")
    shutil.copy2("/workspace/train_gpt_decode.py", run_dir / "train_gpt_decode.py")


def _write_command_file(run_dir: Path, cmd: list[str], env_vars: dict[str, str]) -> None:
    exports = [f"export {key}={shlex.quote(value)}" for key, value in sorted(env_vars.items())]
    lines = exports + ["cd /workspace", " ".join(shlex.quote(part) for part in cmd)]
    (run_dir / "command.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _move_outputs(run_dir: Path) -> None:
    for fname in ("final_model.pt", "final_model.int6.ptz"):
        src = Path(f"/workspace/{fname}")
        if src.exists():
            dst = run_dir / fname
            shutil.move(str(src), str(dst))
            print(f"saved: {dst}")


def _run_torchrun(nproc: int, env_overrides: dict[str, str]) -> None:
    """Launch torchrun with nproc processes inside the container."""
    env = os.environ.copy()
    env.update(BEST_NONTTT_ENV)
    env.update(env_overrides)
    run_id = env["RUN_ID"]

    run_dir = _prepare_run_dir(run_id)
    _snapshot_run_files(run_dir)

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        "--standalone",
        "/workspace/train_gpt_decode_sidecar.py",
    ]
    command_env = BASE_ENV | BEST_NONTTT_ENV | env_overrides
    _write_command_file(run_dir, cmd, {key: env[key] for key in sorted(command_env)})
    print(f"launching: {' '.join(cmd)}")
    print(f"env overrides: {env_overrides}")
    try:
        subprocess.run(cmd, env=env, check=True)
    finally:
        _move_outputs(run_dir)
        run_vol.commit()  # flush logs + artifacts even on failure


@app.function(
    gpu="H100:8",
    cpu=16.0,
    memory=128 * 1024,   # 128 GiB RAM for data loading + GPTQ Hessians
    timeout=60 * 60,     # 1 hour hard limit to prevent runaway costs
    volumes={"/vol/data": data_vol, "/vol/runs": run_vol},
)
def train(
    iterations: int = 20000,
    max_wallclock_seconds: float = 600.0,
    run_id: str | None = None,
    extra_env: dict[str, str] | None = None,
):
    overrides = {
        "ITERATIONS": str(iterations),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
    }
    overrides["RUN_ID"] = run_id or _default_run_id("CaseOps_QKGain50_WD090_ProxyControl_Modal8x")
    if extra_env:
        overrides.update(extra_env)
    _run_torchrun(nproc=8, env_overrides=overrides)


@app.function(
    gpu="H100:1",
    cpu=8.0,
    memory=64 * 1024,
    timeout=30 * 60,
    volumes={"/vol/data": data_vol, "/vol/runs": run_vol},
)
def smoke(
    iterations: int = 20000,
    max_wallclock_seconds: float = 600.0,
    run_id: str | None = None,
):
    """Single-GPU smoke test for the best non-TTT config."""
    overrides = {
        "ITERATIONS": str(iterations),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
        "RUN_ID": run_id or _default_run_id("CaseOps_QKGain50_WD090_ProxyControl_Modal1x_smoke"),
        "TRAIN_LOG_EVERY": "25",
    }
    _run_torchrun(nproc=1, env_overrides=overrides)


# ---------------------------------------------------------------
# Data upload (local -> volume). One-time operation.
# ---------------------------------------------------------------
@app.function(
    volumes={"/vol/data": data_vol},
    timeout=60 * 60,
)
def _check_data(vocab_size: int = DEFAULT_VOCAB_SIZE):
    """Inspect data layout on the volume."""
    datasets_dir = Path(f"/vol/data/datasets/fineweb10B_sp{vocab_size}")
    tokenizer_path = Path(f"/vol/data/tokenizers/fineweb_{vocab_size}_bpe.model")
    train_shards = sorted(datasets_dir.glob("fineweb_train_*.bin"))
    val_shards = sorted(datasets_dir.glob("fineweb_val_*.bin"))
    print(f"datasets_dir: {datasets_dir}")
    print(f"train shards: {len(train_shards)}")
    print(f"val shards: {len(val_shards)}")
    print(f"tokenizer present: {tokenizer_path.exists()} ({tokenizer_path})")


@app.local_entrypoint()
def upload_data(local_data_dir: str = str(DEFAULT_DATA_DIR), vocab_size: int = DEFAULT_VOCAB_SIZE):
    """
    Push one tokenizer variant from local data into the Modal Volume.
    """
    src = Path(local_data_dir).resolve()
    assert src.is_dir(), f"{src} does not exist"
    datasets_dir = src / "datasets" / f"fineweb10B_sp{vocab_size}"
    tokenizer_path = src / "tokenizers" / f"fineweb_{vocab_size}_bpe.model"
    assert datasets_dir.is_dir(), f"{datasets_dir} does not exist"
    assert tokenizer_path.is_file(), f"{tokenizer_path} does not exist"
    print(f"uploading {datasets_dir} and {tokenizer_path} -> volume {DATA_VOLUME}")
    subprocess.run(
        _modal_cli_cmd(
            "volume",
            "put",
            DATA_VOLUME,
            str(datasets_dir),
            f"/datasets/fineweb10B_sp{vocab_size}",
            "--force",
        ),
        check=True,
    )
    subprocess.run(
        _modal_cli_cmd(
            "volume",
            "put",
            DATA_VOLUME,
            str(tokenizer_path),
            f"/tokenizers/fineweb_{vocab_size}_bpe.model",
            "--force",
        ),
        check=True,
    )
    _check_data.remote(vocab_size=vocab_size)


@app.local_entrypoint()
def download_run(run_id: str, dest: str = "./runs"):
    """Pull one run directory from the artifact volume into local runs/."""
    dest_root = Path(dest).resolve()
    dest_root.mkdir(exist_ok=True, parents=True)
    dest_path = dest_root / run_id
    subprocess.run(
        _modal_cli_cmd("volume", "get", ARTIFACT_VOLUME, f"/{run_id}", str(dest_root), "--force"),
        check=True,
    )
    print(f"downloaded to {dest_path}")


@app.local_entrypoint()
def download_logs(dest: str = "./runs"):
    """Pull the full artifact volume to local runs/."""
    Path(dest).mkdir(exist_ok=True, parents=True)
    subprocess.run(
        _modal_cli_cmd("volume", "get", ARTIFACT_VOLUME, "/", dest, "--force"),
        check=True,
    )
    print(f"downloaded to {dest}")


# ---------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------
@app.local_entrypoint()
def main(
    iterations: int = 20000,
    wallclock: float = 600.0,
    run_id: str = "",
):
    """Full 8xH100 run for the best non-TTT CaseOps proxy-control config."""
    resolved_run_id = run_id or _default_run_id("CaseOps_QKGain50_WD090_ProxyControl_Modal8x")
    train.remote(
        iterations=iterations,
        max_wallclock_seconds=wallclock,
        run_id=resolved_run_id,
    )
    download_run(run_id=resolved_run_id)


@app.local_entrypoint()
def smoke_test(
    iterations: int = 20000,
    wallclock: float = 600.0,
    run_id: str = "",
):
    """Single H100 10-minute smoke path for the best non-TTT config."""
    resolved_run_id = run_id or _default_run_id("CaseOps_QKGain50_WD090_ProxyControl_Modal1x_smoke")
    smoke.remote(
        iterations=iterations,
        max_wallclock_seconds=wallclock,
        run_id=resolved_run_id,
    )
    download_run(run_id=resolved_run_id)
