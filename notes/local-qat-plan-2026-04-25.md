# Local QAT Plan

This note lays out the next research branch: introducing **QAT** into our best current **local no-TTT** stack.

This is a plan, not yet an executed run report.

## Goal

Start from the current best **non-QAT** local branch and answer:

1. can a **simple legal QAT path** improve our current GPTQ-only branch?
2. if yes, what late-QAT schedule works best on 1x5090?
3. what does the QAT landscape look like across start-time, ramp, and branch interactions?

## Current Best Non-QAT Baseline

Promote this as the baseline for the QAT branch:

- tokenizer: `Casefold V2`
- `SKIP_GATES_ENABLED=1`
- `PARALLEL_RESIDUAL_START=7`
- `QK_GAIN_INIT=4.5`
- `HESSIAN_CLIP_LAMBDA=0.15`
- `NUM_LOOPS=0`
- `TTT_ENABLED=0`
- quantization: GPTQ post-training only

Current best local result from this branch:

- run: `2026-04-24_CasefoldV2_SkipGates_ParResid_QKGain45_HessClip015`
- prequant `val_bpb`: `1.19595508`
- quantized `val_bpb`: `1.20120846`

This is the branch all QAT experiments should be measured against first.

## Why QAT Now

Our local branch has already extracted obvious gains from:
- tokenizer choice
- skip gates
- parallel residuals
- small QK gain tuning
- small HessianClip tuning

The remaining gap is now mostly in the **quantization roundtrip**.

That makes QAT a reasonable next branch, because the current stack is:
- otherwise stable
- already well-tuned structurally
- still exporting through GPTQ int6 / int8

So the right next question is:

> can we shrink the quantization gap without destabilizing the branch?

## Legal QAT References In The Repo

There are already legal record-track precedents we can reuse.

### Reference 1: `2026-03-31_ParallelResiduals_MiniDepthRecurrence`

This is the most relevant starting point.

Important properties:
- legal record-track run
- no TTT
- soft-round QAT already implemented in the trainer
- late activation path already implemented

Relevant details:
- QAT is applied through `CastedLinear`
- fake-quant is only applied to 2D weights during training
- it uses a soft-round / STE-style int6 approximation
- it activates late using `late_qat_threshold`
- alpha is ramped after activation

This is the **simplest legal QAT path** to borrow.

### Reference 2: `2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6`

This record is useful as confirmation that:
- soft-round QAT + skip-gated skip connections already coexist in a legal record-track branch
- all-int6 GPTQ plus QAT is a viable regime

### What not to copy first

Do **not** start from the non-record Noisy QAT work.

Reason:
- it is tied to recurrence-heavy experiments
- it is more complex
- it is not the simplest insertion path for our current branch

For this repo, the first QAT branch should be as simple as possible.

## Proposed QAT Baseline Implementation

### Design goal

Keep the entire current best branch fixed, and add only the smallest legal QAT mechanism that is already validated upstream.

### Proposed implementation

Fork the current local source script family and add:

- **late soft-round QAT**
- applied only to quantized 2D matrix weights during training
- no embedding QAT initially
- no scalar/control-tensor QAT initially
- no eval-time adaptation
- keep GPTQ export path unchanged

### Why this is the right first QAT

It is:
- closest to legal record-track precedent
- low-surface-area
- directly targeted at our current weakness (post-training quantization gap)
- unlikely to explode code complexity immediately

### What stays fixed in the initial QAT branch

- `Casefold V2`
- `SkipGates=1`
- `PARALLEL_RESIDUAL_START=7`
- `QK_GAIN_INIT=4.5`
- `HESSIAN_CLIP_LAMBDA=0.15`
- `NUM_LOOPS=0`
- `TTT_ENABLED=0`
- GPTQ calibration/export path

## Phase 1: Build The Simplest QAT Baseline

### Objective

Create one local trainer fork from the current best branch with:
- late soft-round int6 QAT on 2D weights only
- a simple activation knob such as `QAT_FRACTION`
- a simple ramp knob such as `QAT_RAMP_STEPS`

### Minimal config surface

Start with these QAT knobs only:
- `QAT_ENABLED=1`
- `QAT_FRACTION`
- `QAT_RAMP_STEPS`

Optional but not required in v1:
- `QAT_TARGET`
- `QAT_ALPHA_INIT`
- `QAT_ALPHA_FINAL`

### Success criteria for Phase 1

- branch runs stably on 1x5090
- throughput does not collapse
- quantized `val_bpb` improves or at least shows a promising trend

## Phase 2: Initial Research Runs (2-3 runs)

These are the first runs after the QAT branch exists.

The purpose is not to sweep everything. It is to answer:
- does QAT help at all?
- how late should it start?
- does it fight with HessianClip?

### Run A: simplest late QAT

`CasefoldV2_ParResid_QAT10`

Config:
- baseline branch unchanged
- `QAT_FRACTION=0.10`
- `QAT_RAMP_STEPS=500`

Why:
- closest to the "late QAT" precedent
- should minimize training disruption

### Run B: longer late QAT window

`CasefoldV2_ParResid_QAT20`

Config:
- same as Run A
- `QAT_FRACTION=0.20`

Why:
- checks whether the branch needs more time to adapt to quantization

### Run C: late QAT without HessianClip

`CasefoldV2_ParResid_QAT10_NoHessClip`

Config:
- same as Run A
- `HESSIAN_CLIP_LAMBDA=0.0`

Why:
- separates "QAT helps" from "QAT + HessianClip helps"
- tells us whether the two are complementary or partially redundant

### Decision rules after the first 3 runs

- if none beats the non-QAT baseline by at least a small margin, pause the QAT branch
- if one of them improves quantized `val_bpb` cleanly, use that as the QAT base for the larger sweep
- if training becomes unstable or throughput collapses badly, simplify further before continuing

## Phase 3: 10 More Runs To Explore The Landscape

Assuming at least one of the first 3 runs is promising, run about 10 more experiments.

The point here is to map the local QAT landscape, not just find one lucky setting.

### Bucket A: QAT start-time sweep (4 runs)

These test how late is best.

1. `QAT_FRACTION=0.05`
2. `QAT_FRACTION=0.10`
3. `QAT_FRACTION=0.15`
4. `QAT_FRACTION=0.20`

Hold everything else fixed at the best result from Phase 2.

### Bucket B: QAT ramp sweep (3 runs)

These test how sharply QAT should turn on.

5. `QAT_RAMP_STEPS=250`
6. `QAT_RAMP_STEPS=500`
7. `QAT_RAMP_STEPS=1000`

Why:
- abrupt ramp may destabilize
- slow ramp may not give enough adaptation time

### Bucket C: interaction sweeps (3 runs)

These test whether the QAT branch still wants the same surrounding settings.

8. `QAT best + HESSIAN_CLIP_LAMBDA=0.15`
9. `QAT best + QK_GAIN_INIT=4.0`
10. `QAT best + QK_GAIN_INIT=5.0`

Why:
- QAT may change which surrounding non-QAT settings are optimal

## Optional Bucket D: If The Above Looks Strong

Only if the QAT branch is clearly promising.

Possible extras:
- attn/MLP-only QAT target split
- deeper branch without parallel residuals for comparison
- a clamp-related branch inspired by Issue `#775`

But these should be deferred until the simpler landscape is understood.

## Metrics To Record For Every QAT Run

At minimum:
- prequant `val_bpb`
- quantized `val_bpb`
- quantization gap (`quantized - prequant`)
- training throughput (`tok/s`)
- wallclock to stop
- total bytes
- whether the run was stable

The most important metric for the QAT branch is:

> **does QAT reduce the quantization gap without paying too much elsewhere?**

## What Counts As A Win

A QAT run is a real win if:
- quantized `val_bpb` beats `1.20120846`
- throughput stays roughly in the same range
- artifact size remains viable
- the run is stable and reproducible

If QAT only improves the prequant metric but not the quantized one, it is not solving the right problem.

## What I Expect

My current expectation is:

- the **simplest late soft-round QAT** is the right first branch
- a modest late-QAT window will help more than a very aggressive one
- the best QAT setup will likely still want `ParallelResidualStart=7`
- the quantization gain, if real, will probably be modest but meaningful

## Recommended Execution Order

### First 3 runs

1. `CasefoldV2_ParResid_QAT10`
2. `CasefoldV2_ParResid_QAT20`
3. `CasefoldV2_ParResid_QAT10_NoHessClip`

### Then the 10-run map

4. `QAT05`
5. `QAT10`
6. `QAT15`
7. `QAT20`
8. `Ramp250`
9. `Ramp500`
10. `Ramp1000`
11. `QAT + HessClip015`
12. `QAT + QKGain40`
13. `QAT + QKGain50`

## Bottom Line

The right QAT plan is:

1. start from the current best non-QAT local branch
2. add the **simplest legal late soft-round QAT** from record-track precedent
3. run 2-3 initial tests to see if QAT is positive at all
4. if yes, run about 10 more experiments to map the local landscape
5. judge success by the **quantized** metric, not by training loss or prequant metrics alone
