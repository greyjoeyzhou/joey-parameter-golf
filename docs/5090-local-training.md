# Running Parameter Golf on RTX 5090 (Desktop, 32GB VRAM)

The baseline `train_gpt.py` targets 8×H100 SXM. This document covers what changes on a single RTX 5090 (Blackwell, `sm_120`, 32 GB GDDR7) relative to the 4090 path. Start by reading [`4090-local-training.md`](./4090-local-training.md) — most of the reasoning transfers; this doc only notes the deltas.

## TL;DR

- **SMEM is still a concern.** Blackwell consumer (`sm_120`) keeps the ~100 KB max-dynamic-SMEM-per-block limit inherited from Ada. Apply the same `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` fix defensively. If you skip it and see the Triton OOM, you're hitting the same bug.
- **More VRAM (32 GB vs 16 GB).** Room for larger per-step token counts or wider/deeper experiments before you need grad accumulation tricks.
- **Newer toolchain required.** Blackwell (`sm_120`) needs CUDA **12.8+** and PyTorch **2.7+** (2.10+ recommended, matching the upstream `torch==2.10` expectation in `train_gpt.py`).
- **FP8 native, FP4 available.** Worth considering for QAT / low-precision experiments that are blocked on H100s today.

## Quick Start

```bash
# Environment: conda/uv env with torch 2.10+ built against CUDA 12.8+,
# sentencepiece, numpy, tqdm, huggingface-hub, datasets

# Sanity check that torch sees sm_120
python -c "import torch; print(torch.cuda.get_device_capability(0))"
# Expect: (12, 0)

# Download dataset (10 shards ≈ 1B tokens; use --train-shards 80 for full 8B)
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Train (uncapped wallclock; 10-min cap is an 8×H100 rule)
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 \
RUN_ID=baseline_5090 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=2000 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Expected: measurably faster than the 4090 (more SMs, higher memory bandwidth, GDDR7), but exact step time needs a local benchmark. Fill in the numbers in the table below after your first real run.

## The Triton Persistent-Reduction OOM (same bug, different headroom)

See the 4090 doc for the full explanation. The short version:

| GPU | Max dynamic SMEM per block | Status on upstream `train_gpt.py` |
|-----|---------------------------:|-----------------------------------|
| H100 SXM   | 228 KB | Works |
| A100       | 164 KB | Works |
| RTX 4090 (`sm_89`)  | 99 KB  | OOM (kernel needs ~112 KB) |
| **RTX 5090 (`sm_120`)** | **~100 KB (Blackwell consumer)** | **Expected to OOM without the fix** |

Blackwell's datacenter variant (GB200/GB100, `sm_100`) raises SMEM-per-SM substantially, but the **consumer GB202 die in the 5090 did not**. Until you've confirmed a clean run without it, keep `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` set.

The fix, failure mode, and all the codegen internals (`mix_order_reduction`, `override_persistent_reduction`, the `MULTI_KERNEL` Triton `cache_key` bug, the "manual RMSNorm made it worse" trap) are identical to the 4090 case. Refer to [`4090-local-training.md`](./4090-local-training.md#deep-dive-inductor-persistent-reduction-internals).

### If `MIX_ORDER_REDUCTION=0` alone is not enough on 5090

Possible (I haven't confirmed on hardware). If a residual kernel still exceeds SMEM:

1. Verify the actual per-block SMEM limit: `python -c "import torch; props = torch.cuda.get_device_properties(0); print(props.shared_memory_per_block_optin)"`.
2. If the returned limit is ≥ 112 KB, you should be fine even without the env var — but leave it on, the perf hit is ~5% and it's free insurance.
3. If a fused kernel still OOMs, the escape hatch is eager mode: `TORCHDYNAMO_DISABLE=1`. 1.6× slower on 4090, should be similar here.

## Toolchain Notes Specific to Blackwell

### PyTorch / CUDA versions

- **CUDA runtime 12.8+** is the first release with proper `sm_120` codegen. Earlier CUDA will either fall back to PTX JIT (slow, sometimes broken) or fail outright.
- **PyTorch 2.7** added `sm_120` kernels; **2.10** is what upstream `train_gpt.py` pins against. Don't try to run on older torch just to save time.
- **Triton** ships with torch; don't pip-install a separate Triton wheel on top. Blackwell codegen paths were still stabilizing as of torch 2.10, so if you see weird Triton errors, update before you debug.
- **nightly vs stable**: stable 2.10 is fine for `sm_120` as of early 2026. Fall back to nightly only if you hit a Blackwell-specific inductor bug.

### Install recipe (reference)

```bash
# Using uv (preferred for my own envs)
uv venv --python 3.11
source .venv/bin/activate
uv pip install --pre torch --index-url https://download.pytorch.org/whl/cu128
uv pip install sentencepiece numpy tqdm huggingface-hub datasets tiktoken kernels setuptools
```

Adjust to `/whl/cu129` or similar once released. The upstream `requirements.txt` doesn't pin CUDA; you pick it via the wheel index.

### NCCL

The 4090 doc's `nvidia-nccl-cu12` vs `cu13` conflict note applies here too. Blackwell just means you're more likely to be on `cu13` or later. If `import torch` prints `undefined symbol: ncclAlltoAll`:

```bash
pip uninstall -y nvidia-nccl-cu12 nvidia-nccl-cu13
pip install nvidia-nccl-cu13 --force-reinstall
```

(Or the `cu14` successor if that's current when you read this.)

## VRAM Budget (32 GB)

Upstream script uses `grad_accum_steps = 8 // world_size`, so with one 5090 you still do 8 micro-batches. Defaults are 64 sequences × 1024 tokens = 65,536 tokens per micro-batch. On 32 GB you have real headroom:

- You can likely raise `TRAIN_BATCH_TOKENS` (or the equivalent knob) to reduce the number of accumulation steps per optimizer step, cutting per-step overhead.
- Wider / deeper experimental architectures that OOM on 4090 (e.g., adding a recurrence loop over layers, doubling MLP mult) become feasible for short smoke tests.
- For full-length runs, watch for inductor kernels compiling with larger tile sizes as you scale — they can push back into the SMEM danger zone even though you have global memory to spare.

## Blackwell-specific opportunities

These aren't required, just worth knowing since the point of a 5090 is to explore things you can't easily test elsewhere:

- **Native FP8 (E4M3 / E5M2)** — same as Ada / Hopper. No new news, but confirms QAT / low-precision experiments from the leaderboard (e.g., `int6 QAT`, ternary) are directly runnable here.
- **FP4 (`e2m1`) tensor cores** — new on Blackwell. If you want to prototype sub-int6 quantization below the leaderboard's current state of the art, this is the cheapest place to do it. Note: PyTorch's FP4 support is still immature as of torch 2.10; you may need `torchao` or custom Triton.
- **TMA (Tensor Memory Accelerator)** — available on sm_100+ but not on sm_120 consumer Blackwell (same as Hopper-only on the datacenter side). Do **not** rely on TMA-based kernels from datacenter-targeted repos.

## Speed Comparison

Run the native benchmark runner first. It uses the same local trainer path as the 4090 note and reports a steady-state step-time estimate that is much more useful for a 4090 vs 5090 hardware comparison than a wallclock-capped record replay.

```bash
python3 benchmarks_pilot/run_native_baseline.py --variant sp1024
```

If you want to run the trainer directly instead, use:

```bash
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 \
RUN_ID=bench_5090 \
ITERATIONS=2000 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=2000 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

| Configuration | Step Time | 2k Steps | val_bpb @2k | Notes |
|---------------|-----------|----------|-------------|-------|
| compiled, MOR=0 | **623 ms** | **20.8 min** | **1.2965** | native `sp1024` benchmark via `run_native_baseline.py`; **841.5K tok/s** |
| eager (no compile) | TBD | TBD | — | not rerun yet on this machine |

For this repo, the more useful 4090 reference is the **1x4090 eGPU baseline from `instruction_000`**: ~598K tok/s, which is about **877 ms/step equivalent** at `TRAIN_BATCH_TOKENS=524288`. The measured 5090 native baseline here is **+40.7% faster** than that reference.

## Default Environment Variables for 5090

```bash
# Required (defensive — see SMEM table above)
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0

# Recommended defaults for local iteration
ITERATIONS=2000          # short runs for fast feedback; scale up for real runs
MAX_WALLCLOCK_SECONDS=0  # disable 10-min cap (8×H100-specific)
VAL_LOSS_EVERY=2000      # validate only at the end of short runs
TRAIN_LOG_EVERY=200      # reduce log noise
```

## Open questions to resolve on first run

- [ ] Confirm `shared_memory_per_block_optin` on your 5090 — exact number decides whether `MIX_ORDER_REDUCTION=0` is required or just defensive.
- [x] Benchmark step time vs. 4090 numbers; native compiled baseline is 623 ms/step, +40.7% vs the 1x4090 eGPU baseline.
- [ ] Verify torch 2.10 + CUDA 12.8 combination doesn't emit Blackwell-only inductor warnings.
- [ ] Try raising `TRAIN_BATCH_TOKENS` to use the extra VRAM and see if step time drops or SMEM OOM returns.
