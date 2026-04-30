"""
Modal runner for train_gpt_grouped_gptq.py
Usage:
    modal run modal_runner.py::main           # full training (8xH100)
    modal run modal_runner.py::smoke_test     # smoke test (1xH100, 200 steps)
    modal run modal_runner.py::upload_data    # upload local data to Volume
    modal run modal_runner.py::download_logs  # pull logs back to local
"""
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "param-golf-gpt"
DATA_VOLUME = "param-golf-data"      # FineWeb shards + tokenizer
ARTIFACT_VOLUME = "param-golf-runs"  # logs, checkpoints, submission blobs

# ---------------------------------------------------------------
# Image: CUDA 12.6 devel + torch 2.8 + FA3 prebuilt wheel
# ---------------------------------------------------------------
# Use devel image to ensure nvcc is available -- FA3 is a prebuilt wheel,
# but torch.compile sometimes invokes nvrtc when generating inductor kernels.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.8.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "numpy",
        "sentencepiece",
        "brotli",
        "packaging",
        "ninja",
        "psutil",
    )
    # FA3 prebuilt wheel -- must match cu126 + torch2.8
    .pip_install(
        "flash_attn_3",
        extra_options="--find-links https://windreamer.github.io/flash-attention3-wheels/cu126_torch280",
    )
    # Runtime environment variables
    .env({
        "PYTHONUNBUFFERED": "1",
        "NCCL_DEBUG": "WARN",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        # Match the data layout expected by the training script
        "DATA_DIR": "/vol/data",
    })
    # Inject training script last (editing it locally won't require image rebuild)
    .add_local_file(
        "train_gpt_grouped_gptq.py",
        "/workspace/train_gpt_grouped_gptq.py",
        copy=False,  # mount at runtime, don't bake into image layer
    )
)

app = modal.App(APP_NAME, image=image)

data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
run_vol = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)


# ---------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------
def _run_torchrun(nproc: int, env_overrides: dict[str, str]) -> None:
    """Launch torchrun with nproc processes inside the container."""
    env = os.environ.copy()
    env.update(env_overrides)

    # Write logs to the artifact volume
    log_dir = Path("/vol/runs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    os.chdir("/workspace")
    # Symlink so the script's default logs/ dir writes to the volume
    logs_link = Path("/workspace/logs")
    if not logs_link.exists():
        os.symlink("/vol/runs/logs", str(logs_link), target_is_directory=True)

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        "--standalone",
        "train_gpt_grouped_gptq.py",
    ]
    print(f"launching: {' '.join(cmd)}")
    print(f"env overrides: {env_overrides}")
    subprocess.run(cmd, env=env, check=True)

    # Move model files to volume (shutil.move needed for cross-device)
    import shutil
    for fname in ("final_model.pt", "final_model.int6.ptz"):
        src = Path(f"/workspace/{fname}")
        if src.exists():
            dst = Path(f"/vol/runs/{fname}")
            shutil.move(str(src), str(dst))
            print(f"saved: {dst}")

    run_vol.commit()  # flush to persistent storage immediately


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
    if run_id:
        overrides["RUN_ID"] = run_id
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
def smoke(iterations: int = 200, max_wallclock_seconds: float = 120.0):
    """Single-GPU smoke test: verify image + data + script all work."""
    overrides = {
        "ITERATIONS": str(iterations),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
        # Single GPU: grad_accum_steps = 8 // 1 = 8, already correct by default
        "RUN_ID": "smoke",
        "TRAIN_LOG_EVERY": "10",
        "VAL_LOSS_EVERY": "100",
    }
    _run_torchrun(nproc=1, env_overrides=overrides)


# ---------------------------------------------------------------
# Data upload (local -> volume). One-time operation.
# ---------------------------------------------------------------
@app.function(
    volumes={"/vol/data": data_vol},
    timeout=60 * 60,
)
def _check_data():
    """Inspect data layout on the volume."""
    import glob
    base = "/vol/data"
    print(f"contents of {base}:")
    for p in sorted(Path(base).rglob("*"))[:50]:
        print(f"  {p.relative_to(base)}  ({p.stat().st_size if p.is_file() else 'dir'})")
    train_shards = glob.glob(f"{base}/datasets/fineweb10B_sp4096/fineweb_train_*.bin")
    val_shards = glob.glob(f"{base}/datasets/fineweb10B_sp4096/fineweb_val_*.bin")
    print(f"train shards: {len(train_shards)}")
    print(f"val shards: {len(val_shards)}")


@app.local_entrypoint()
def upload_data(local_data_dir: str = "./data"):
    """
    Push local data directory to the Volume. Expected layout:
        {local_data_dir}/datasets/fineweb10B_sp4096/fineweb_train_*.bin
        {local_data_dir}/datasets/fineweb10B_sp4096/fineweb_val_*.bin
        {local_data_dir}/tokenizers/fineweb_4096_bpe.model
    `modal volume put` via CLI is faster, but this is convenient for automation.
    """
    src = Path(local_data_dir).resolve()
    assert src.is_dir(), f"{src} does not exist"
    print(f"uploading {src} -> volume {DATA_VOLUME}")
    # CLI is faster: modal volume put <volume> <local> <remote>
    subprocess.run(
        ["modal", "volume", "put", DATA_VOLUME, str(src), "/", "--force"],
        check=True,
    )
    _check_data.remote()


@app.local_entrypoint()
def download_logs(dest: str = "./runs"):
    """Pull logs and artifacts from volume to local."""
    Path(dest).mkdir(exist_ok=True, parents=True)
    subprocess.run(
        ["modal", "volume", "get", ARTIFACT_VOLUME, "/", dest, "--force"],
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
    """modal run modal_runner.py --iterations 20000 --wallclock 600"""
    train.remote(
        iterations=iterations,
        max_wallclock_seconds=wallclock,
        run_id=run_id or None,
    )
    download_logs()


@app.local_entrypoint()
def smoke_test():
    """modal run modal_runner.py::smoke_test"""
    smoke.remote()
    download_logs()
