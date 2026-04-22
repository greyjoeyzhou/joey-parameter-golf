# Benchmark Runbook: 5090 vs Competition Records

Goal: establish a **native 5090 baseline** first using the same local trainer path as the 4090 notes, then run the 3 most recent competition records (track_10min_16mb) on a single RTX 5090 as secondary compatibility replays.

## Hardware

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (sm_120, Blackwell consumer) |
| VRAM | 32 GB GDDR7 |
| Driver | 591.86 (CUDA 13.1 driver) |
| Interface | eGPU |

## Competition reference (8×H100 SXM, 10 min)

| Metric | Value |
|---|---|
| Tokens/sec | ~1.35M |
| Tokens in 600s | ~816M |
| Typical steps | 4550–5088 |
| Step time | ~130 ms |

## Current measured results

### Native baseline

| Run | Tokens/sec | Step time | Time to 816M | Comparison |
|---|---:|---:|---:|---|
| `2026-04-21_5090NativeBaseline_sp1024` | **841.5K** | **623.0 ms** | **16.2 min** | **+40.7% vs 1x4090 eGPU baseline (~598K tok/s)** |

### Record replays

| Run | Source | Status | Effective tok/s | Notes |
|---|---|---|---:|---|
| `2026-04-21_SP8192_QK5_LegalTTT` | `2026-04-06` | wallclock cap at step 47 | 29.7K | quantized val_bpb 3.0137 |
| `2026-04-21_SP8192_ParResid_TTT` | `2026-04-08` | wallclock cap at step 26 | 16.9K | quantized val_bpb 3.1522 |
| `2026-04-21_SP8192_3LayerRecur_ParResid_QK525` | `2026-04-09` | compile-bound | n/a | no completed training step after 25m27s |

## Pre-flight checklist

### 0. Primary local hardware benchmark

Use the native runner first. This is the apples-to-apples 4090 vs 5090 comparison because it stays on `parameter-golf/train_gpt.py` and reports steady-state interval step time after warmup.

```bash
python3 benchmarks_pilot/run_native_baseline.py --variant sp1024
```

Why this is the primary comparison:

- It uses the same local trainer path as `docs/4090-local-training.md`.
- It avoids conflating hardware speed with Hopper-only `flash_attn_3` assumptions in recent record scripts.
- It reports steady-state step time instead of folding compile-heavy warmup and recurrence activation into a single wallclock figure.

### 1. Python environment

```bash
cd /home/joey/Code/joey-parameter-golf/parameter-golf

# Active benchmark env in this repo is `.venv312x`
```

Verify:
```bash
.venv312x/bin/python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('gpu:', torch.cuda.get_device_name(0))
print('cap:', torch.cuda.get_device_capability(0))
print('smem_optin:', torch.cuda.get_device_properties(0).shared_memory_per_block_optin)
"
```

Expected: torch 2.7+, CUDA 12.8+, cap (12, 0), smem ~100KB.

### 2. Dataset

`sp8192` is already cached locally in this repo. `sp1024` was fetched directly from `willdepueoai/parameter-golf` for the native benchmark because the local manifest has drifted.

### 3. Key 5090 env vars (always set)

```bash
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0   # prevent SMEM OOM on sm_120 (same bug as 4090)
export MAX_WALLCLOCK_SECONDS=1200            # 20-min cap (competition has 600s for 8×H100)
export TRAIN_LOG_EVERY=50                    # step-level throughput at fine resolution
export VAL_LOSS_EVERY=500                    # periodic val during the 20-min run
export WARMUP_STEPS=1                        # only for record replays; native baseline uses 20
export GPTQ_CALIBRATION_BATCHES=1            # shorten replay post-training tail
export SEED=42
```

### 4. Permissions needed

- **Filesystem write**: `benchmarks_pilot/` (log files, metadata). ✓ No elevation needed.
- **GPU compute**: user-space CUDA via NVIDIA driver. ✓ No elevation needed (user in `video`/`render` group or driver accessible).
- **Network (dataset download only)**: HuggingFace Hub (HTTPS). ✓ Outbound only. No credentials needed for `kevclark/parameter-golf` public repo.
- **No sudo needed** for any step.

Verify GPU access: `local_bin/nvidia-smi` is used on this machine because the real `nvidia-smi` is not on PATH.

---

## Run commands

These record replays are still useful, but treat them as **secondary compatibility probes** after the native baseline above.

### Run A — SP8192 + QK5 + LegalTTT  (record: 2026-04-06)

```bash
cd /home/joey/Code/joey-parameter-golf/parameter-golf

PATH=/home/joey/Code/joey-parameter-golf/local_bin:$PATH \
PYTHONPATH=/home/joey/Code/joey-parameter-golf/local_shims \
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=500 \
WARMUP_STEPS=1 \
GPTQ_CALIBRATION_BATCHES=1 \
SEED=42 \
QK_GAIN_INIT=5.0 \
TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  .venv312x/bin/torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828/train_gpt.py \
  2>&1 | tee ../benchmarks_pilot/2026-04-21_SP8192_QK5_LegalTTT/train.log
```

### Run B — SP8192 + Parallel Residuals + Score-First TTT  (record: 2026-04-08)

```bash
cd /home/joey/Code/joey-parameter-golf/parameter-golf

PATH=/home/joey/Code/joey-parameter-golf/local_bin:$PATH \
PYTHONPATH=/home/joey/Code/joey-parameter-golf/local_shims \
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=500 \
WARMUP_STEPS=1 \
GPTQ_CALIBRATION_BATCHES=1 \
SEED=42 \
TTT_ENABLED=1 \
PARALLEL_START_LAYER=7 \
  .venv312x/bin/torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-04-08_SP8192_ParallelResid_ScoreFirstTTT/train_gpt.py \
  2>&1 | tee ../benchmarks_pilot/2026-04-21_SP8192_ParResid_TTT/train.log
```

### Run C — SP8192 + 3-Layer Recurrence + Parallel Residuals + QK 5.25 + LegalTTT  (record: 2026-04-09)

```bash
cd /home/joey/Code/joey-parameter-golf/parameter-golf

PATH=/home/joey/Code/joey-parameter-golf/local_bin:$PATH \
PYTHONPATH=/home/joey/Code/joey-parameter-golf/local_shims \
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=500 \
WARMUP_STEPS=1 \
GPTQ_CALIBRATION_BATCHES=1 \
SEED=42 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  .venv312x/bin/torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py \
  2>&1 | tee ../benchmarks_pilot/2026-04-21_SP8192_3LayerRecur_ParResid_QK525/train.log
```

---

## During training — monitoring commands

Run these in a separate terminal while training is active:

```bash
# GPU utilization and memory
watch -n 2 /home/joey/Code/joey-parameter-golf/local_bin/nvidia-smi

# Live log tail with step timing
tail -f benchmarks_pilot/2026-04-21_<RUN>/train.log | grep -E "step|val|loss|bpb|tok"

# Extract step times so far (to compute tokens/sec on the fly)
grep "step" benchmarks_pilot/2026-04-21_<RUN>/train.log | tail -20
```

---

## After training — analytics

### Extract key metrics from log

```bash
LOG=benchmarks_pilot/2026-04-21_<RUN>/train.log

# Tokens/sec (from step log lines)
grep -E "tok/s|tokens/s" $LOG | tail -5

# Step times
grep -E "ms/step|step_time" $LOG | awk '{print $NF}' | sort -n | tail -3

# val_bpb checkpoints
grep -E "val_bpb|val_loss" $LOG

# Total steps completed
grep "^step" $LOG | tail -1

# Wallclock
grep -E "wallclock|elapsed" $LOG | tail -3
```

### Compute 5090 throughput estimate

```python
# Paste this snippet (adjust from log):
step_time_ms = <median step time from log>
tokens_per_step = 64 * 1024 * 8   # 64 seqs * 1024 tok * 8 grad_accum (1 GPU)
tokens_per_sec = tokens_per_step / (step_time_ms / 1000)
time_to_816M = 816_000_000 / tokens_per_sec / 60  # minutes to match competition volume

print(f"Tokens/sec: {tokens_per_sec:,.0f}")
print(f"Minutes to reach 816M tokens (competition volume): {time_to_816M:.1f} min")
print(f"Steps in 20 min: {20*60 / (step_time_ms/1000):.0f}")
```

### Produce summary entry

After each run, extract: run_id, source_record, steps_completed, step_time_ms (median),
tokens_per_sec, val_bpb_final, wall_sec, peak_vram_gb. Append to `benchmarks_pilot/summary.tsv`.

---

## Expected throughput estimates

| GPU setup | Expected tok/sec | Time to 816M tok | Steps in 20 min (est) |
|---|---|---|---|
| 8×H100 SXM (competition) | ~1.35M | 10 min | 4550–5088 |
| 1×4090 eGPU (baseline) | ~598K | ~23 min | ~380 |
| 1×5090 eGPU native baseline | **841.5K** | **16.2 min** | **~1926 @ 623 ms/step** |

The recent-record replay rows are far slower and should not be used as the primary hardware comparison.
