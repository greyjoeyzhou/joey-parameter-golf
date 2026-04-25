# Local QAT Results

This note summarizes the overnight local 1x5090 QAT batch.

Primary references:
- `notes/local-qat-plan-2026-04-25.md`
- `notes/local-5090-followup-results-2026-04-24.md`

## Goal

Start from the current best non-QAT branch and test whether a minimal legal late-QAT path can improve the quantized result.

The starting non-QAT branch was:
- `Casefold V2`
- `SkipGates`
- `ParallelResidualStart=7`
- `QK_GAIN_INIT=4.5`
- `HESSIAN_CLIP_LAMBDA=0.15`
- `NUM_LOOPS=0`
- `TTT_ENABLED=0`

Best non-QAT reference:
- run: `2026-04-24_CasefoldV2_SkipGates_ParResid_QKGain45_HessClip015`
- quantized `val_bpb`: `1.20120846`

## QAT Implementation

The local QAT branch used:
- `scripts/train_gpt_decode_qat.py`

Design choices:
- minimal wrapper over the existing `2026-04-06` trainer family
- soft-round QAT on 2D `CastedLinear` weights only
- no embedding QAT
- no scalar/control-tensor QAT
- GPTQ export path left unchanged
- late activation by wallclock fraction
- configurable ramp via `QAT_RAMP_STEPS`

This is intentionally the simplest legal-style QAT insertion path, modeled after the existing legal record-track QAT references rather than the more complex non-record noisy-QAT work.

## Important Early Finding

The first 10-minute QAT smoke looked terrible, but a matched 10-minute non-QAT control looked almost equally terrible.

That showed the short-run smoke metric was dominated by **short-run EMA lag**, not by a broken QAT implementation.

So the overnight 30-minute batch is the first meaningful QAT result set.

## Overnight Batch Status

Batch files:
- `runs/2026-04-25_QATBatch/status.tsv`
- `runs/2026-04-25_QATBatch/queue.log`

All 10 queued runs completed successfully.

## Runs Executed

### 1. `QAT10`

Run:
- `runs/2026-04-25_CasefoldV2_ParResid_QAT10/`

Settings:
- `QAT_FRACTION=0.10`
- `QAT_RAMP_STEPS=500`
- `QK_GAIN_INIT=4.5`
- `HESSIAN_CLIP_LAMBDA=0.15`

Results:
- prequant `val_bpb`: `1.20182672`
- quantized `val_bpb`: `1.20725563`

Interpretation:
- bad baseline QAT setting
- confirms that simply turning on QAT with the old non-QAT hyperparameters is not enough

### 2. `QAT20`

Run:
- `runs/2026-04-25_CasefoldV2_ParResid_QAT20/`

Settings:
- `QAT_FRACTION=0.20`
- `QAT_RAMP_STEPS=500`
- `QK_GAIN_INIT=4.5`
- `HESSIAN_CLIP_LAMBDA=0.15`

Results:
- prequant `val_bpb`: `1.19536926`
- quantized `val_bpb`: `1.20064436`

Interpretation:
- clearly competitive
- much better than `QAT10`
- suggests that the branch needs more adaptation time than the original 10% setting gave it

### 3. `QAT10_NoHessClip`

Run:
- `runs/2026-04-25_CasefoldV2_ParResid_QAT10_NoHessClip/`

Results:
- prequant `val_bpb`: `1.19649214`
- quantized `val_bpb`: `1.20180764`

Interpretation:
- much better than plain `QAT10`
- still worse than the best non-QAT baseline
- confirms HessianClip interaction matters

### 4. `QAT05`

Run:
- `runs/2026-04-25_CasefoldV2_ParResid_QAT05/`

Results:
- prequant `val_bpb`: `1.19564281`
- quantized `val_bpb`: `1.20088355`

Interpretation:
- small but real improvement over the non-QAT baseline
- late QAT can help even with a small adaptation window if the branch is otherwise matched well

### 5. `QAT15`

Run:
- `runs/2026-04-25_CasefoldV2_ParResid_QAT15/`

Results:
- prequant `val_bpb`: `1.19582735`
- quantized `val_bpb`: `1.20114997`

Interpretation:
- slightly better than the best non-QAT branch
- not as strong as `QAT20`

### 6. `QAT10_Ramp250`

Run:
- `runs/2026-04-25_CasefoldV2_ParResid_QAT10_Ramp250/`

Results:
- prequant `val_bpb`: `1.19785027`
- quantized `val_bpb`: `1.20316933`

Interpretation:
- too abrupt
- short ramp appears worse than the more gradual alternatives

### 7. `QAT10_Ramp1000`

Run:
- `runs/2026-04-25_CasefoldV2_ParResid_QAT10_Ramp1000/`

Results:
- prequant `val_bpb`: `1.19552118`
- quantized `val_bpb`: `1.20077910`

Interpretation:
- strong result
- gentler ramp helps
- better than `QAT10`, better than the non-QAT baseline, but not the batch winner

### 8. `QAT10_QKGain40`

Run:
- `runs/2026-04-25_CasefoldV2_ParResid_QAT10_QKGain40/`

Results:
- prequant `val_bpb`: `1.19534324`
- quantized `val_bpb`: `1.20058967`

Interpretation:
- **best run in the QAT batch**
- this is the strongest evidence that QAT shifts the optimal surrounding hyperparameters

### 9. `QAT10_QKGain50`

Run:
- `runs/2026-04-25_CasefoldV2_ParResid_QAT10_QKGain50/`

Results:
- prequant `val_bpb`: `1.19626392`
- quantized `val_bpb`: `1.20155421`

Interpretation:
- worse than `QK_GAIN_INIT=4.0`
- slightly worse than the best non-QAT branch

### 10. `QAT20_NoHessClip`

Run:
- `runs/2026-04-25_CasefoldV2_ParResid_QAT20_NoHessClip/`

Results:
- prequant `val_bpb`: `1.19903094`
- quantized `val_bpb`: `1.20420794`

Interpretation:
- clearly negative
- strongest evidence in the batch that HessianClip and QAT are complementary rather than redundant

## Ranking

Quantized `val_bpb` ranking from the QAT batch:

1. `QAT10_QKGain40` -> `1.20058967`
2. `QAT20` -> `1.20064436`
3. `QAT10_Ramp1000` -> `1.20077910`
4. `QAT05` -> `1.20088355`
5. `QAT15` -> `1.20114997`
6. best non-QAT reference -> `1.20120846`
7. `QAT10_QKGain50` -> `1.20155421`
8. `QAT10_NoHessClip` -> `1.20180764`
9. `QAT10_Ramp250` -> `1.20316933`
10. `QAT20_NoHessClip` -> `1.20420794`
11. `QAT10` -> `1.20725563`

## Main Findings

### 1. QAT is real and useful on this branch

This is the main result.

The best QAT run beat the best non-QAT run:

- best non-QAT: `1.20120846`
- best QAT: `1.20058967`

Improvement:
- about `-0.00062 bpb`

That is not huge, but it is real.

### 2. QAT changes the best surrounding hyperparameters

Under non-QAT, our best branch used:
- `QK_GAIN_INIT=4.5`

Under QAT, the best run used:
- `QK_GAIN_INIT=4.0`

This is one of the most important lessons from the batch.

QAT is not just a final add-on; it changes the local optimum of the surrounding stack.

### 3. QAT needs enough adaptation time or a gentler transition

Evidence:
- `QAT20` was far better than plain `QAT10`
- `QAT10_Ramp1000` was far better than `QAT10_Ramp250`

Interpretation:
- the branch does not like abrupt or under-budget QAT
- either more QAT time or a gentler ramp improves outcomes

### 4. HessianClip still matters under QAT

Removing HessianClip hurt badly, especially at 20% QAT:

- `QAT20`: `1.20064436`
- `QAT20_NoHessClip`: `1.20420794`

Current interpretation:
- HessianClip and QAT are complementary in this branch
- QAT does not replace the need for a good post-training quantization setup

### 5. The QAT landscape is shallow but coherent

The best QAT runs are clustered quite tightly.

That suggests:
- there is a real gain available
- but not a huge cliff
- the next improvement will likely come from interaction tuning rather than one magical new switch

## Recommended New Default

Promote this to the new best local branch:

- `Casefold V2`
- `SkipGates`
- `ParallelResidualStart=7`
- `QAT_ENABLED=1`
- `QAT_FRACTION=0.10`
- `QAT_RAMP_STEPS=500`
- `QK_GAIN_INIT=4.0`
- `HESSIAN_CLIP_LAMBDA=0.15`
- no loops
- no TTT

Why:
- this is the best measured quantized result in the repo so far
- `1.20058967`

## Recommended Next Experiments

The next most valuable runs should stay in the QAT branch and refine the best region rather than reopening broader exploration.

Best next steps:

1. `QAT20 + QKGain40`
2. `QAT20 + Ramp1000 + QKGain40`
3. `QAT05 + QKGain40`
4. `QAT15 + QKGain40`
5. `QAT10 + Ramp1000 + QKGain40`

These follow naturally from the overnight pattern:
- lower QK gain helps under QAT
- gentler or longer QAT adaptation helps
- HessianClip should stay on

## Bottom Line

The overnight QAT batch changed the picture.

Before this batch, the best local branch was:
- `Casefold V2 + SkipGates + ParallelResidual + QKGain45 + HessClip015`

After this batch, the best local branch is:
- **`Casefold V2 + SkipGates + ParallelResidual + QAT + QKGain40 + HessClip015`**

So QAT is no longer just a speculative direction for this repo.
It has now produced the best local quantized result we have measured so far.
