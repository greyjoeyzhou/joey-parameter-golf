# 5090 Benchmark Notes

These notes now split the 5090 work into two benchmark families:

- **Native baseline**: the primary 4090 vs 5090 hardware comparison, using the same `parameter-golf/train_gpt.py` path as the local GPU docs.
- **Record replay compatibility**: secondary runs that check whether the latest leaderboard stacks survive and train sensibly on a single Blackwell GPU.

## Native Baseline

Use the native runner first:

```bash
python3 benchmarks_pilot/run_native_baseline.py --variant sp1024
```

Why this is the primary comparison:

- It uses the same trainer path as `docs/4090-local-training.md`.
- It measures steady-state interval step time after warmup instead of a wallclock-capped replay that includes compile-heavy recurrence switches.
- It stays on the native PyTorch SDPA path that already works on this 5090.

The runner writes its own benchmark directory under `benchmarks_pilot/<RUN_ID>/` with `stdout.log`, `command.sh`, and `summary.json`.

### Current native result

| Run | Path | Step time | Tokens/sec | Time to 816M tokens | Comparison | val_bpb | Peak VRAM |
|---|---|---:|---:|---:|---|---:|---:|
| `2026-04-21_5090NativeBaseline_sp1024` | `train_gpt.py` + `sp1024` | **623.0 ms/step** | **841.5K** | **16.2 min** | **+40.7% vs 1x4090 eGPU baseline (~598K tok/s)** | **1.2965** | **10191 MiB** |

This is the primary hardware comparison result. It clears the user's expected “20-30% faster than 4090” threshold.

## Record Replay Compatibility

The results below are from the recent record replays. They are **not** the primary 4090 vs 5090 comparison.

## Local Harness Adjustments

- The upstream record scripts assume Hopper `flash_attn_3`. On this Blackwell 5090, `PYTHONPATH=local_shims` injects a local `flash_attn_interface` shim backed by PyTorch SDPA so the scripts can run.
- The scripts also shell out to `nvidia-smi`. `PATH=local_bin:$PATH` injects a lightweight local compatibility command because `nvidia-smi` is not available on this machine's PATH.
- `WARMUP_STEPS=1` was used for the local benchmarks so compile warmup did not consume nearly the entire wallclock budget before measurable training started.
- `GPTQ_CALIBRATION_BATCHES=1` was used on later runs to shorten post-training quantization. Throughput comparisons below use the training-phase wallclock stop, not the GPTQ tail.

## Important Caveat

- These 5090 numbers are not apples-to-apples with the original H100 record submissions because the attention backend is different on this machine. The 5090 path is best treated as a local throughput baseline for experimentation, not a direct leaderboard-equivalent replay.

## Headline Replay Results

| Run | Source record | Effective tok/s | Time to 816M tokens | Notes |
|---|---|---:|---:|---|
| `2026-04-21_SP8192_QK5_LegalTTT` | `2026-04-06_SP8192_QK5_LegalTTT_1.0828` | 29.7K | 458.5 min | wallclock cap at step 47; quantized val_bpb 3.0137 |
| `2026-04-21_SP8192_ParResid_TTT` | `2026-04-08_SP8192_ParallelResid_ScoreFirstTTT` | 16.9K | 805.0 min | wallclock cap at step 26; quantized val_bpb 3.1522 |
| `2026-04-21_SP8192_3LayerRecur_ParResid_QK525` | `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` | n/a | n/a | compile-bound: no completed training step after 25m27s |

See `summary.tsv` for the machine-readable table.
