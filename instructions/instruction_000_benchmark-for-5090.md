# Benchmark: Running Competition Records on 1x RTX 5090

Run a **native local baseline benchmark first**, then use the most recent 3 runs under `parameter-golf/records/track_10min_16mb` as a secondary compatibility replay on our hardware (single 5090). The native baseline is the primary 4090 vs 5090 comparison; the record replays are for understanding how recent leaderboard stacks behave on a single Blackwell GPU.

## Hardware

- GPU: NVIDIA RTX 5090 eGPU (32 GB VRAM)
- See `AGENTS.md` and `docs/5090-local-training.md` for machine configs, as well as number that we need to fill.

## Compute Equivalence

The competition runs on **8x H100 SXM for 10 min**.
Measured throughput on H100 and 4090 hardware shown below.

| Setup | Measured tokens/sec | Time to reach 816M tokens |
|-------|-------------------|--------------------------|
| 8x H100 (competition) | ~1.35M | 10 min |
| 1x 4090 eGPU — baseline (9L, seq=1024) | ~598K | **~23 min** |
| 1x 4090 eGPU — heavy (7-stack, SmearGate) | ~317K | ~43 min |
| 1x 5090 eGPU — native baseline (measured) | **841.5K** | **16.2 min** |

Measure the native baseline first so the 5090 vs 4090 comparison uses the same `parameter-golf/train_gpt.py` path and a steady-state step-time metric. Then use the record replays to establish how recent SOTA configs translate to a single-GPU setup.

## Native Baseline

- Use `benchmarks_pilot/run_native_baseline.py` for the primary local hardware comparison.
- Default target: the same native `train_gpt.py` path referenced by `docs/4090-local-training.md` and `docs/5090-local-training.md`.
- Compare **steady-state interval step time after warmup**, not a 20-minute wallclock-capped record replay that includes compile-heavy architecture switches and Hopper-specific attention fallbacks.

Example:

```bash
python3 benchmarks_pilot/run_native_baseline.py --variant sp1024
```

If `sp1024` data is missing locally, seed it first using the current bootstrap notes in `benchmarks_pilot/RUNBOOK.md`.

Current measured native result:

- `2026-04-21_5090NativeBaseline_sp1024`
- `623.04 ms/step` steady-state median
- `841.5K tok/s`
- `1.2965 val_bpb @ 2k`
- **+40.7% vs the 1x4090 eGPU baseline**

## Record Replay

- After the native baseline, run the 3 most recent `track_10min_16mb` records for 20 minutes each.
- Treat those runs as compatibility / architectural behavior probes, not the primary 4090 vs 5090 hardware comparison.

Current replay outcomes on this machine:

- `2026-04-21_SP8192_QK5_LegalTTT`: 47 steps before cap, `29.7K tok/s` effective
- `2026-04-21_SP8192_ParResid_TTT`: 26 steps before cap, `16.9K tok/s` effective
- `2026-04-21_SP8192_3LayerRecur_ParResid_QK525`: compile-bound, no completed training step after `25m27s`

## Record

- record details logs and metrics along the training to help understand if the 5090 could be a good approximation for training behavior in H100
- output the result to the dir `benchmarks_pilot`, each run as a single dir named as `<run_id>`
- output the general sumamry as a `summary.tsv` under the `benchmarks_pilot` dir.

## Collect snippets

- collect what commands are needed for pre-flight, during training, after-training analytics
- collect what permissions needed during the operations
