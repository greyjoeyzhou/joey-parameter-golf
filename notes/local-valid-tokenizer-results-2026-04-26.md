# Local Valid Tokenizer Results

This note summarizes the overnight local 1x5090 batch comparing the two tokenizer options we currently regard as valid under a conservative lossless reading of the competition rules:

- stock `SP8192`
- `CaseOps`

Primary references:
- `notes/local-valid-tokenizer-comparison-2026-04-25.md`
- `notes/local-qat-structure-analysis-2026-04-25.md`
- `notes/local-5090-arch-tokenizer-results-2026-04-23.md`

## Goal

Answer a narrower question than the earlier tokenizer experiments:

> if we restrict ourselves to lossless-valid tokenizer options, does `CaseOps` still provide a real benefit over stock `SP8192` on the stronger local stacks?

This batch intentionally excludes:
- `Casefold`
- lowercase / casefold `SP10240`
- Gravity

The point here was not to find the best tokenizer in the abstract.
The point was to compare the two most practical valid options on the same ladder of local baselines.

## Comparison Matrix

We ran six stack variants, each with both tokenizers.

### Non-QAT stacks

1. `Base`
2. `SkipGates`
3. `SkipGates_ParResid`
4. `SkipGates_ParResid_QKGain45_HessClip015`

### QAT stacks

5. `SkipGates_ParResid_QAT10_QKGain40`
6. `SkipGates_ParResid_QAT20_QKGain40_MLPOnly`

That produced 12 full 30-minute runs.

## Smoke Check

Before the batch, one short run validated the riskiest combined path:

- `2026-04-25_CaseOps_SkipGates_ParResid_QAT20_QKGain40_MLPOnly_Valid_smoke`

That smoke was only a wiring check for:
- the CaseOps byte-sidecar evaluation path
- the QAT wrapper
- the new MLP-only QAT targeting

It is not part of the ranked 30-minute comparison set.

## Full Ranking

Quantized `val_bpb` ranking from the valid-tokenizer batch:

1. `CaseOps_SkipGates_ParResid_QKGain45_HessClip015` -> `1.21949691`
2. `SP8192_SkipGates_ParResid` -> `1.21983153`
3. `CaseOps_SkipGates_ParResid_QAT10_QKGain40` -> `1.21987584`
4. `CaseOps_SkipGates_ParResid_QAT20_QKGain40_MLPOnly` -> `1.21999858`
5. `CaseOps_SkipGates_ParResid` -> `1.22020667`
6. `SP8192_SkipGates_ParResid_QAT20_QKGain40_MLPOnly` -> `1.22053427`
7. `SP8192_SkipGates_ParResid_QKGain45_HessClip015` -> `1.22104359`
8. `CaseOps_SkipGates` -> `1.22121163`
9. `SP8192_SkipGates_ParResid_QAT10_QKGain40` -> `1.22144442`
10. `SP8192_SkipGates` -> `1.22267872`
11. `CaseOps_Base` -> `1.24166658`
12. `SP8192_Base` -> `1.24352652`

## Pairwise Tokenizer Deltas

CaseOps minus SP8192, same stack:

1. `Base` -> `-0.00185994`
2. `SkipGates` -> `-0.00146709`
3. `SkipGates_ParResid` -> `+0.00037514`
4. `SkipGates_ParResid_QKGain45_HessClip015` -> `-0.00154668`
5. `SkipGates_ParResid_QAT10_QKGain40` -> `-0.00156858`
6. `SkipGates_ParResid_QAT20_QKGain40_MLPOnly` -> `-0.00053569`

Summary:
- CaseOps won **5 of 6** pairwise comparisons
- the only loss was the plain `SkipGates_ParResid` stack
- the mean delta across the six stacks is about **`-0.00110 bpb`** in favor of CaseOps

## Main Findings

### 1. Yes, CaseOps still provides a real benefit overall

This is the main answer from the batch.

Under the lossless-valid comparison set:
- CaseOps won the batch overall
- CaseOps won 5 of the 6 direct tokenizer matchups
- CaseOps improved the best measured valid branch relative to the best stock-SP8192 branch

So the answer is not just "CaseOps is viable."

It is:

> **CaseOps is currently the strongest valid tokenizer option we have measured locally.**

### 2. The current best valid branch is non-QAT CaseOps with the tuned structural stack

Winner:
- `2026-04-25_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid`
- prequant `val_bpb`: `1.21457238`
- quantized `val_bpb`: `1.21949691`

This beat the best stock-SP8192 branch from the same batch:
- `2026-04-25_SP8192_SkipGates_ParResid_Valid`
- quantized `val_bpb`: `1.21983153`

Delta:
- about `-0.00033 bpb`

That margin is not huge, but it is real and it came from a clean head-to-head comparison.

### 3. CaseOps benefits are broad, but not uniform

CaseOps helped on:
- `Base`
- `SkipGates`
- `SkipGates_ParResid_QKGain45_HessClip015`
- `SkipGates_ParResid_QAT10_QKGain40`
- `SkipGates_ParResid_QAT20_QKGain40_MLPOnly`

CaseOps lost only on:
- `SkipGates_ParResid`

Interpretation:
- the effect is not universal
- the best view is that CaseOps is a modest but repeatable positive tokenizer branch once the surrounding stack is tuned reasonably well

### 4. QAT did not become the winner under the valid-tokenizer comparison

For the valid batch, the best run was still non-QAT.

That does not mean QAT is useless here.
It means:
- the structural and quantization tuning that helped the invalid Casefold branch does not transfer one-for-one to the valid-tokenizer regime
- the valid branch likely needs its own retuning, not just a direct copy of the Casefold-side QAT defaults

This is especially clear because the best CaseOps QAT runs were close, but not best:

- `CaseOps_QAT10_QKGain40` -> `1.21987584`
- `CaseOps_QAT20_QKGain40_MLPOnly` -> `1.21999858`

Both are competitive, but both trail the non-QAT tuned CaseOps winner.

### 5. The valid branch is still materially worse than the best Casefold research branch

Current best valid local result:
- `1.21949691`

Current best Casefold research branch:
- `1.20008068`

Gap:
- about `+0.01942 bpb`

That is large.

So the honest takeaway is:
- CaseOps improves the valid branch
- but the valid branch is still far behind the best lossy research branch

That means we should treat the Casefold-side research as a source of structural ideas, not as a directly shippable recipe.

## Interpretation: Why CaseOps Looks Better Now Than It Did In The Earlier Local Sweep

Earlier local CaseOps experiments looked flat to slightly negative.

This batch changed that picture.

Most likely reason:
- the earlier tests used weaker surrounding stacks
- the current comparison used stronger architecture and quantization settings
- CaseOps seems to need that stronger surrounding branch before its benefit becomes visible locally

This matches the earlier repo-level intuition that CaseOps tends to sit on top of stronger stacks rather than carrying the entire result by itself.

## Recommended New Valid Default

Promote this to the new best local valid branch:

- tokenizer: `CaseOps`
- `SKIP_GATES_ENABLED=1`
- `PARALLEL_RESIDUAL_START=7`
- `QK_GAIN_INIT=4.5`
- `HESSIAN_CLIP_LAMBDA=0.15`
- `QAT_ENABLED=0`
- `NUM_LOOPS=0`
- `TTT_ENABLED=0`

Why:
- it is the best measured valid local result so far
- it beat the stock-SP8192 comparison run cleanly
- it gives us a legally safer default than the Casefold branch

## Recommended Next Experiments

The next valid-tokenizer batch should stay on CaseOps and retune around the new winner.

Highest-value next runs:

1. `CaseOps + SkipGates + ParResid + QKGain45 + HessClip015 + QAT10`
2. `CaseOps + SkipGates + ParResid + QKGain45 + HessClip015 + QAT20_MLPOnly`
3. `CaseOps + SkipGates + ParResid + QKGain45 + HessClip015 + QAT10_Ramp1000`
4. `CaseOps + SkipGates + ParResid + QKGain40 + HessClip015`
5. `CaseOps + SkipGates + ParResid + QKGain50 + HessClip015`

Why these:
- they test whether the winning non-QAT CaseOps stack can absorb the QAT ideas successfully
- they avoid assuming the old Casefold-side QAT optimum transfers unchanged

## Bottom Line

This batch gave a much clearer answer than the earlier local CaseOps tests.

The answer is:

- **yes, CaseOps still provides benefits overall**
- **CaseOps is currently the best locally measured valid tokenizer option**
- **the best valid branch right now is a tuned non-QAT CaseOps stack**
- **QAT remains promising, but it needs retuning specifically for the valid-tokenizer regime**

That is a solid place to continue from if we want to stay on the lossless-safe side of the rules.
