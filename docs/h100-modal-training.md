# Running Parameter Golf on Modal (1-8xH100)

`tmp_modal/` contains a temporary Modal launcher for running a Parameter Golf training script on H100s. This doc explains how that launcher works in this repo, what needs adjusting before first use, and the current caveats.

## TL;DR

- Use `tmp_modal/modal_runner.py` as the entrypoint, not `modal_runner.py` from repo root.
- The current runner is wired to the best recent non-TTT setting: `2026-04-28_CaseOps_QKGain50_WD090_ProxyControl_Valid2p5h`.
- It exposes a `1x` H100 smoke path and an `8x` H100 full-run path.
- In this repo, local data lives under `parameter-golf/data`, and `upload_data` already defaults there.
- Artifacts are now written per-run on the Modal volume and downloaded into local `runs/<RUN_ID>/`.
- If you do not have a global `modal` binary installed, use `uvx --from modal modal ...`.

## What `tmp_modal` Currently Does

`tmp_modal/modal_runner.py` defines a Modal app that:

- builds an image from `nvidia/cuda:12.8.1-devel-ubuntu22.04`
- installs Python 3.11, `torch==2.9.1` for `cu128`, and `flash_attn_3`
- mounts the exact archived trainer snapshots from the best non-TTT run:
  - `runs/2026-04-28_CaseOps_QKGain50_WD090_ProxyControl_Valid2p5h/train_gpt_decode_sidecar.py`
  - `runs/2026-04-28_CaseOps_QKGain50_WD090_ProxyControl_Valid2p5h/train_gpt_decode.py`
- creates two persistent Modal volumes:
  - `param-golf-data` for datasets and tokenizers
  - `param-golf-runs` for logs and model artifacts
- runs `torchrun --nnodes=1 --standalone --nproc_per_node=8` for full training
- exposes a 1xH100 smoke test
- snapshots the exact trainer files plus `command.sh` into the run directory
- downloads a single run directory back into local `./runs/<RUN_ID>/`

The runner also sets:

- `DATA_DIR=/vol/data`
- `PYTHONUNBUFFERED=1`
- `NCCL_DEBUG=WARN`
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`

## Repo-Specific Adjustments

The notes in `tmp_modal/MODAL_RUNNER.md` assume a flatter project layout than this repo actually uses.

### 1. Use the right file path in commands

From repo root, run:

```bash
uvx --from modal modal run tmp_modal/modal_runner.py::main --iterations 20000 --wallclock 600 --run-id 2026-04-29_MyRun
```

not:

```bash
modal run modal_runner.py::main ...
```

### 2. Point data uploads at `parameter-golf/data`

The local entrypoint already defaults to this repo's actual data root: `./parameter-golf/data`.

Use either:

```bash
uvx --from modal modal run tmp_modal/modal_runner.py::upload_data --vocab-size 8192
```

or the raw CLI:

```bash
uvx --from modal modal volume put param-golf-data ./parameter-golf/data/datasets/fineweb10B_sp8192 /datasets/fineweb10B_sp8192
uvx --from modal modal volume put param-golf-data ./parameter-golf/data/tokenizers/fineweb_8192_bpe.model /tokenizers/fineweb_8192_bpe.model
```

### 3. Authentication is required

You need a valid Modal token before any of the above commands will work. If `modal volume ls` or `modal run` says `Token missing`, authenticate first:

```bash
uvx --from modal modal token new
```

## Choosing GPU Count (1-8)

There are two knobs that must stay aligned:

- the Modal GPU reservation, for example `gpu="H100:4"`
- the `torchrun` worker count, for example `--nproc_per_node=4`

For a run using `N` cards, both values should be `N`.

| Cards | Modal reservation | `torchrun` workers | Typical use |
|------:|-------------------|--------------------|-------------|
| 1 | `H100:1` | `1` | smoke tests, compile/debug |
| 2 | `H100:2` | `2` | cheap distributed sanity checks |
| 3 | `H100:3` | `3` | ad hoc throughput checks |
| 4 | `H100:4` | `4` | medium-cost iteration |
| 5 | `H100:5` | `5` | partial-scale experiments |
| 6 | `H100:6` | `6` | near-full-scale experiments |
| 7 | `H100:7` | `7` | near-record rehearsals |
| 8 | `H100:8` | `8` | record-style 10-minute runs |

### Current status of `tmp_modal/modal_runner.py`

Today the file only defines:

- `smoke()` on `H100:1`
- `train()` on `H100:8`

So the current checked-in helper does not yet let you choose `2` through `7` cards from the CLI.

### Recommended interface

If you generalize the runner, the natural user-facing command is:

```bash
uvx --from modal modal run tmp_modal/modal_runner.py::main \
  --gpus 4 \
  --iterations 20000 \
  --wallclock 600 \
  --run-id 2026-04-29_MyRun
```

To make that real, the runner needs to:

1. validate `1 <= gpus <= 8`
2. reserve `gpu="H100:<gpus>"` in Modal
3. call `_run_torchrun(nproc=gpus, env_overrides=...)`
4. ensure the training script's batch and accumulation logic still make sense at that `world_size`

### Important batch-size caveat

Many Parameter Golf training scripts, including the upstream baseline family, use logic shaped like:

```python
grad_accum_steps = 8 // world_size
```

That means `1`, `2`, `4`, and `8` are the cleanest choices if you want behavior closest to the 8-GPU baseline. With `3`, `5`, `6`, or `7` GPUs, integer floor division changes the effective batch schedule unless you explicitly retune the script.

For that reason:

- use `1` for smoke/debug
- use `2` or `4` for cheaper scaled-down distributed tests
- use `8` for final record-style runs
- use `3`, `5`, `6`, or `7` only if your chosen training script is written to handle those counts cleanly

## Prerequisites

Install and authenticate the Modal CLI locally:

```bash
uvx --from modal modal --version
uvx --from modal modal token new
```

The runner uses `create_if_missing=True`, so the two named volumes are created automatically on first use.

## Preparing Local Data

The runner expects the volume to contain:

- `/datasets/fineweb10B_sp<VOCAB_SIZE>/`
- `/tokenizers/fineweb_<VOCAB_SIZE>_bpe.model`

Since `DATA_DIR=/vol/data`, the training script should ultimately resolve data under `/vol/data/datasets/...` and `/vol/data/tokenizers/...`.

If you do not already have local cached data, fetch it in the upstream submodule first. Example for `sp4096`:

```bash
cd parameter-golf
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest
```

For `sp8192`, use:

```bash
cd parameter-golf
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --skip-manifest
```

## One-Time Upload to Modal

If you want to upload only the pieces needed for `sp4096`, use the raw CLI from repo root:

```bash
uvx --from modal modal volume put param-golf-data ./parameter-golf/data/datasets/fineweb10B_sp4096 /datasets/fineweb10B_sp4096
uvx --from modal modal volume put param-golf-data ./parameter-golf/data/tokenizers/fineweb_4096_bpe.model /tokenizers/fineweb_4096_bpe.model
```

For `sp8192`:

```bash
uvx --from modal modal volume put param-golf-data ./parameter-golf/data/datasets/fineweb10B_sp8192 /datasets/fineweb10B_sp8192
uvx --from modal modal volume put param-golf-data ./parameter-golf/data/tokenizers/fineweb_8192_bpe.model /tokenizers/fineweb_8192_bpe.model
```

Verify the volume contents with:

```bash
uvx --from modal modal volume ls param-golf-data /datasets
uvx --from modal modal volume ls param-golf-data /tokenizers
```

## Running a Full 8xH100 Job

The current runner is already wired to the archived best non-TTT CaseOps proxy-control trainer. Start a record-style wallclock-capped run with:

```bash
uvx --from modal modal run tmp_modal/modal_runner.py::main \
  --iterations 20000 \
  --wallclock 600 \
  --run-id 2026-04-29_MyRun
```

What happens:

1. Modal provisions one 8xH100 container.
2. The runner launches `torchrun --nproc_per_node=8` inside that container.
3. Training logs stream to your terminal.
4. The runner snapshots `train_gpt_decode_sidecar.py`, `train_gpt_decode.py`, and `command.sh` into `/vol/runs/<RUN_ID>/`.
5. The runner commits logs and artifacts to the `param-golf-runs` volume.
6. The local entrypoint downloads that single run directory into `./runs/<RUN_ID>/`.

The first run will be slower because Modal has to build the image and cold-start the container.

If you later generalize the runner to accept `--gpus`, the same command shape can be used for `1` through `8` cards, subject to the batch-size caveat above.

## Smoke Test

Use the built-in 1xH100 smoke test to validate the image, data mount, and script wiring:

```bash
uvx --from modal modal run tmp_modal/modal_runner.py::smoke_test \
  --wallclock 600 \
  --run-id 2026-04-29_CaseOps_QKGain50_WD090_ProxyControl_Modal1x_smoke
```

The smoke entrypoint currently sets:

- `ITERATIONS=20000`
- `MAX_WALLCLOCK_SECONDS=600`
- `TRAIN_LOG_EVERY=25`
- the archived best non-TTT proxy-control env overrides from `2026-04-28_CaseOps_QKGain50_WD090_ProxyControl_Valid2p5h`

If you want a `2x`, `4x`, or other intermediate-card smoke path, add a corresponding Modal function or generalize `main()` plus the underlying Modal function so the GPU reservation and `nproc` are both derived from the same `gpus` argument.

## Downloading Artifacts Manually

To re-pull the artifact volume later:

```bash
uvx --from modal modal run tmp_modal/modal_runner.py::download_run --run-id 2026-04-29_MyRun
```

Equivalent raw CLI:

```bash
uvx --from modal modal volume get param-golf-runs /2026-04-29_MyRun ./runs/2026-04-29_MyRun --force
```

The current runner writes:

- logs under `/vol/runs/<RUN_ID>/logs`
- `final_model.pt` at `/vol/runs/<RUN_ID>/final_model.pt`
- `final_model.int6.ptz` at `/vol/runs/<RUN_ID>/final_model.int6.ptz`
- trainer snapshots and `command.sh` under `/vol/runs/<RUN_ID>/`

## Changing Vocab Size or Other Env Vars

The underlying `train()` Modal function supports `extra_env`, but the current `main()` local entrypoint only exposes:

- `iterations`
- `wallclock`
- `run_id`

So if you want to set something like `VOCAB_SIZE=4096`, you currently need to either:

1. edit `tmp_modal/modal_runner.py` and hardcode the override, or
2. add another local entrypoint that forwards `extra_env` into `train.remote(...)`

Make sure the matching dataset and tokenizer are already uploaded to the `param-golf-data` volume.

## Cost and Runtime Expectations

The notes in `tmp_modal/MODAL_RUNNER.md` estimate roughly:

| Config | Typical Duration | Cost |
|--------|------------------|------|
| 8xH100 full run | ~15-20 min including startup | ~$7-10 |
| 1xH100 smoke test | ~5-10 min | ~$0.5-1 |

Those numbers depend heavily on image cache hits, current Modal pricing, and how much post-training eval or quantization the chosen script performs.

## Current Caveats

- The runner is still a temporary helper under `tmp_modal/`, not a polished repo workflow.
- It is currently specialized to the best non-TTT CaseOps proxy-control config rather than being a generic trainer launcher.
- It still only exposes `1x` and `8x` entrypoints, not arbitrary `2-7x` GPU counts.
- It snapshots the trainer and command, but it still does not auto-generate the repo-standard `README.md` and `submission.json` run metadata files.
- FlashAttention 3 wheel compatibility is tied to the exact torch and CUDA versions in the image. If you upgrade torch, update the wheel source too.
- Modal metrics can differ slightly from the official Parameter Golf environment because of driver and kernel differences.

## Recommended Next Cleanup

If this runner becomes part of the normal workflow, the first improvements should be:

1. Expose `2x`, `4x`, and general `1-8x` GPU counts from the CLI.
2. Auto-generate the repo-standard `README.md` and `submission.json` in each downloaded run.
3. Expose `extra_env` cleanly from the local entrypoints instead of requiring file edits.
4. Decide whether this runner should stay specialized to the best non-TTT CaseOps config or become a generic archived-run launcher.
