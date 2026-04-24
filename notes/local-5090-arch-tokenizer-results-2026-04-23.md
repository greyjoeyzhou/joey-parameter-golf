# Local 1x5090 Architecture + Tokenizer Results

This note summarizes the local architecture and tokenizer sweep executed on the single RTX 5090.

Scope:
- hardware: `1x RTX 5090`
- tokenizer in architecture phase: fixed `SP8192`
- legality baseline: `no TTT`
- no custom eval-time adaptation in the baseline phase
- local runs only

Primary plan reference:
- `notes/tonight-local-5090-arch-tokenizer-plan-2026-04-22.md`

## Executive Summary

The best local no-TTT baseline architecture from this sweep is:

- `SP8192`
- no loops / recurrence
- no parallel residuals
- **skip gates enabled**

On top of that winning architecture:

- **Casefold V2 helped clearly**
- **CaseOps did not help in the same no-TTT local setting**

The practical conclusion is:

1. use the skip-gated SP8192 no-TTT base as the local architecture anchor
2. prioritize Casefold V2 over CaseOps for the next local no-TTT iteration
3. only revisit CaseOps after reintroducing stronger stack components later

## Method

Base training script family:
- `parameter-golf/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt_decode.py`

Local harness:
- `scripts/run_local_experiment.py`
- `scripts/train_gpt_decode_sidecar.py`

Why this script family:
- already supports SP8192
- exposes the structural knobs we want locally
- can be run with `TTT_ENABLED=0`
- can be cleanly adapted for CaseOps sidecar validation without rewriting the whole record stack

All architecture comparisons used:
- `VOCAB_SIZE=8192`
- `TTT_ENABLED=0`
- `WARMUP_STEPS=1`
- `GPTQ_CALIBRATION_BATCHES=1`
- local 30-minute training cap followed by quantization/eval

## Exploration Result

Exploration run:
- `runs/2026-04-22_SP8192_NoTTT_Loop45x2_Explore20m/`

Finding:
- the looped / recurrence-heavy variant is technically runnable on 1x5090
- it trains before loop activation
- once looping activates, the local path becomes too expensive to be a good local baseline

Practical interpretation:
- recurrence should not be the anchor for a local 1x5090 no-TTT baseline
- it can remain a future add-on idea, but not the local default

## Architecture Baselines

### Candidate A: plain base

Run:
- `runs/2026-04-22_SP8192_NoTTT_Base30m/`

Config:
- SP8192
- no TTT
- no loops
- no parallel residuals
- no skip gates

Results:
- throughput: about `463-468k tok/s`
- prequant `val_bpb`: `1.23678117`
- quantized `val_bpb`: `1.24227277`
- peak VRAM: `26860 MiB`

### Candidate B: skip-gated base

Run:
- `runs/2026-04-22_SP8192_NoTTT_SkipGates30m/`

Config:
- SP8192
- no TTT
- no loops
- no parallel residuals
- skip gates enabled

Results:
- throughput: about `460-461k tok/s`
- prequant `val_bpb`: `1.21760974`
- quantized `val_bpb`: `1.22334557`
- peak VRAM: `27244 MiB`

### Candidate C: parallel-residual base

Run:
- `runs/2026-04-22_SP8192_NoTTT_ParResid30m/`

Config:
- SP8192
- no TTT
- no loops
- parallel residuals from layer 7 onward
- no skip gates

Results:
- throughput: about `428-430k tok/s`
- prequant `val_bpb`: `1.25513416`
- quantized `val_bpb`: `1.26068845`
- peak VRAM: `26670 MiB`

### Architecture Ranking

By quantized `val_bpb`:

1. `SkipGates30m` -> `1.22334557`
2. `Base30m` -> `1.24227277`
3. `ParResid30m` -> `1.26068845`

Takeaways:
- lightweight skip gates clearly helped
- parallel residuals hurt both speed and quality in this local no-TTT setting
- the plain no-loop base is already decent, but skip gates are the best zero-ish-cost structural improvement from this sweep

## Tokenizer Runs On The Winning Architecture

Winning architecture held fixed:
- no loops
- no parallel residuals
- skip gates enabled
- no TTT

### Casefold V2

Run:
- `runs/2026-04-22_CasefoldV2_NoTTT_SkipGates30m_v3/`

Tokenizer/data root:
- `local_tokenizer_data/casefold_v2/` (ignored locally; not committed)

Results:
- throughput: about `460-463k tok/s`
- prequant `val_bpb`: `1.19837140`
- quantized `val_bpb`: `1.20385604`
- peak VRAM: `27244 MiB`

Delta vs winning SP8192 baseline:
- quantized improvement: `-0.01948953 bpb`

Interpretation:
- Casefold V2 gives a clear, meaningful gain in the clean local no-TTT setting
- it preserves throughput almost exactly

### CaseOps

Run:
- `runs/2026-04-22_CaseOps_NoTTT_SkipGates30m_v2/`

Tokenizer/data root:
- `local_tokenizer_data/caseops_v1/` (ignored locally; not committed)

Results:
- throughput: about `454-456k tok/s`
- prequant `val_bpb`: `1.21895652`
- quantized `val_bpb`: `1.22397362`
- peak VRAM: `27244 MiB`

Delta vs winning SP8192 baseline:
- quantized delta: `+0.00062805 bpb`

Interpretation:
- CaseOps is effectively flat to slightly worse than the plain SP8192 baseline here
- in this local no-TTT setup, CaseOps does not justify itself

## Main Findings

### 1. Best local no-TTT architecture

The best local 1x5090 architecture from this sweep is:

- SP8192
- no recurrence / loops
- no parallel residuals
- skip gates on

This is the cleanest local architecture anchor for future tokenizer experiments.

### 2. Recurrence is not a good local anchor on this machine

The recurrence-heavy candidate is not the right baseline for local 1x5090 work.

Reason:
- it reaches training before loops turn on
- after activation, it becomes too compile-expensive / throughput-poor for a simple local baseline workflow

### 3. Parallel residuals do not transfer cleanly here

Parallel residuals were clearly not worthwhile in the local no-TTT setting:
- slower than base
- worse than base on `val_bpb`

### 4. Casefold V2 is the strongest tokenizer result tonight

Casefold V2 gave the strongest practical win of the tokenizer phase:
- good delta
- almost no throughput regression
- no new local training instability

### 5. CaseOps likely needs a stronger surrounding stack

CaseOps did not help on the local no-TTT skip-gated base.

That does **not** imply CaseOps is bad in general.
It implies:
- CaseOps probably needs a stronger surrounding stack to show value
- its best known results depend on different stack choices than what we intentionally held fixed tonight

## Failed / Intermediate Runs

These remain useful and are preserved under `runs/`:

- `2026-04-22_SP8192_NoTTT_Loop45x2_smoke`
- `2026-04-22_SP8192_NoTTT_Base_smoke`
- `2026-04-22_SP8192_NoTTT_ParResid_smoke`
- `2026-04-22_SP8192_NoTTT_SkipGates_smoke`
- `2026-04-22_SP8192_NoTTT_Loop45x230m`
- `2026-04-22_CasefoldV2_NoTTT_SkipGates30m`
- `2026-04-22_CasefoldV2_NoTTT_SkipGates30m_v2`
- `2026-04-22_CaseOps_NoTTT_SkipGates30m`

Most of these failures were useful for one of three reasons:
- confirming recurrence is too heavy locally
- fixing the local tokenizer path handling
- fixing sidecar-aware validation for CaseOps

## Recommendation

Use this as the local default baseline going forward:

- `2026-04-22_SP8192_NoTTT_SkipGates30m`

Use this as the best tokenizer-upgraded local baseline so far:

- `2026-04-22_CasefoldV2_NoTTT_SkipGates30m_v3`

## Next Best Experiments

Recommended next steps in order:

1. **Casefold V2 + another small structural add-on**
   - likely the highest-EV next local step

2. **Casefold V2 + recurrence-lite**
   - only if recurrence can be made less compile-bound locally

3. **Casefold V2 + parallel residuals**
   - to test whether Casefold changes the sign of the parallel-residual delta

4. **Revisit CaseOps only after stronger stack changes**
   - do not prioritize it in the clean local no-TTT regime

## Bottom Line

Tonight's local 1x5090 result is simple:

- the best legal no-TTT SP8192 base is a **skip-gated non-recurrent base**
- **Casefold V2 improves that base materially**
- **CaseOps does not improve that base in the same regime**

That gives us a clean, actionable local research baseline for the next round.
