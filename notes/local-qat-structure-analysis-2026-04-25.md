# Local QAT Structure Analysis

This note summarizes the local 1x5090 QAT structure batch run on 2026-04-25.

Primary references:
- `notes/local-qat-results-2026-04-25.md`
- `notes/local-qat-structure-results-2026-04-25.md`

## Goal

Keep the best current QAT schedule fixed and answer a narrower question:

> where should QAT live inside the model?

The starting point for this batch was the best simple QAT branch from the earlier sweep:

- `Casefold V2`
- `SkipGates`
- `ParallelResidualStart=7`
- `QAT_ENABLED=1`
- `QAT_FRACTION=0.20`
- `QAT_RAMP_STEPS=500`
- `QK_GAIN_INIT=4.0`
- `HESSIAN_CLIP_LAMBDA=0.15`
- `NUM_LOOPS=0`
- `TTT_ENABLED=0`

Reference winner before this batch:
- run: `2026-04-25_CasefoldV2_ParResid_QAT10_QKGain40`
- quantized `val_bpb`: `1.20058967`

## Structural QAT Knobs Added

The local QAT wrapper was extended with three placement knobs:

- `QAT_TARGET=all|attn|mlp`
- `QAT_LAYER_START`
- `QAT_LAYER_END`

This makes it possible to ask two separate questions:

1. should QAT hit attention, MLP, or both?
2. should QAT cover the whole stack or only late layers?

## Smoke Check

Before the full batch, one short smoke validated the placement machinery:

- run: `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_L7plus_smoke`

The smoke was not used for ranking because its 10-minute wallclock is too short to compare meaningfully against the 30-minute runs. Its purpose was only to confirm that module targeting and layer filtering worked correctly.

## Full Batch

All ten planned runs completed successfully.

### Runs executed

1. `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All`
2. `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_L7plus`
3. `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_L8plus`
4. `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_AttnOnly`
5. `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_Attn_L7plus`
6. `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_Attn_L8plus`
7. `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLPOnly`
8. `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLP_L7plus`
9. `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLP_L8plus`
10. `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_EarlyOnly`

## Ranking

Quantized `val_bpb` ranking from the structure batch:

1. `MLPOnly` -> `1.20008068`
2. `All_L8plus` -> `1.20025503`
3. `AttnOnly` -> `1.20061683`
4. `Attn_L8plus` -> `1.20068688`
5. `All_L7plus` -> `1.20088150`
6. `All_EarlyOnly` -> `1.20122299`
7. `Attn_L7plus` -> `1.20140407`
8. `All` -> `1.20178151`
9. `MLP_L7plus` -> `1.20192033`
10. `MLP_L8plus` -> `1.20204191`

## Main Findings

### 1. `MLPOnly` is the new best QAT structure

Winner:
- `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLPOnly`
- prequant `val_bpb`: `1.19466903`
- quantized `val_bpb`: `1.20008068`

Delta vs the prior QAT winner:
- previous best: `1.20058967`
- new best: `1.20008068`
- improvement: about `-0.00051 bpb`

This is a modest gain, but it is clean and measured.

### 2. Full-stack QAT was worse than family-targeted QAT

The naive full-coverage run underperformed badly:

- `All`: `1.20178151`
- `MLPOnly`: `1.20008068`
- `AttnOnly`: `1.20061683`

That means the issue is not simply whether attention QAT is useful or not.
The more important finding is that applying the same QAT schedule to all matrix families at once is worse than targeting one family.

### 3. Late narrowing helped for `all` and `attn`

Evidence:

- `All`: `1.20178151`
- `All_L7plus`: `1.20088150`
- `All_L8plus`: `1.20025503`

and:

- `AttnOnly`: `1.20061683`
- `Attn_L7plus`: `1.20140407`
- `Attn_L8plus`: `1.20068688`

Interpretation:
- if QAT touches attention at all, it seems safer to keep that pressure near the top of the stack
- the strongest all-matrix variant was also the narrowest late-stack one

### 4. MLP QAT wanted broad coverage, not late narrowing

Evidence:

- `MLPOnly`: `1.20008068`
- `MLP_L7plus`: `1.20192033`
- `MLP_L8plus`: `1.20204191`

Interpretation:
- MLP adaptation appears to benefit from broad stack coverage
- unlike attention, MLP QAT did not want to be restricted to only the last few layers

### 5. Early-only QAT did not win

`All_EarlyOnly` landed at `1.20122299`.

That is better than `All`, but clearly worse than the top structure runs.

Interpretation:
- the gain is not coming from a purely early-stack effect
- the earlier working intuition still holds: QAT pressure needs to be placed where it helps the final quantized model most

## Hypothesis: Why `MLPOnly` Beat `All`

My current best hypothesis is that `MLPOnly` captures most of the quantization benefit while avoiding the most optimization damage.

### 1. MLP covers most of the quantized matrix budget

Per block, the MLP path contains the largest matrices in this trainer family.

So even if attention also matters, MLP-only QAT may already cover the majority of the quantization error budget that the model needs to adapt to.

### 2. Attention is a more fragile place to inject fake-quant noise

Quantization error in the MLP mainly perturbs feedforward transforms.

Quantization error in attention perturbs:
- query/key geometry
- attention routing
- value mixing
- the interaction with RoPE and `q_gain`

That means equal fake-quant pressure may be much more disruptive in attention than in MLP.

### 3. A single global QAT schedule is probably too blunt for both families

This branch used one shared setting for:
- `QAT_FRACTION`
- `QAT_RAMP_STEPS`
- `QAT_ALPHA_INIT`
- `QAT_ALPHA_FINAL`

The results suggest that the best schedule for MLP is not the same as the best schedule for attention.

Evidence for that view:
- MLP wanted broad coverage
- attention preferred either narrow late coverage or no joint QAT pressure from MLP at all
- `All` was worse than both `MLPOnly` and `AttnOnly`

### 4. The interaction seems to be the real problem

If `All` were better than both single-family runs, then the conclusion would be "everywhere helps."

But that is not what happened.

Instead:
- `MLPOnly` was best
- `AttnOnly` was competitive
- `All` was clearly worse

That points to a harmful interaction between simultaneous QAT pressure on attention and MLP under a short local adaptation window.

## Recommended New Default

Promote this to the new best local QAT branch:

- `Casefold V2`
- `SkipGates`
- `ParallelResidualStart=7`
- `QAT_ENABLED=1`
- `QAT_FRACTION=0.20`
- `QAT_RAMP_STEPS=500`
- `QAT_TARGET=mlp`
- `QAT_LAYER_START=0`
- `QAT_LAYER_END=10`
- `QK_GAIN_INIT=4.0`
- `HESSIAN_CLIP_LAMBDA=0.15`
- `NUM_LOOPS=0`
- `TTT_ENABLED=0`

Why:
- it is now the best measured quantized local result in the repo
- it is also the cleanest structural lesson from the batch

## Recommended Next Experiments

The next batch should not go back to broad `all` QAT.

Highest-value next runs:

1. `MLPOnly + QAT15`
2. `MLPOnly + QAT10 + Ramp1000`
3. `MLPOnly + QAT20 + Ramp1000`
4. `MLPOnly + QKGain35`
5. `MLPOnly + QKGain45`

If we want one slightly more exploratory test after that, the next structural combination I would try is:

6. `MLPOnly + Attn_L8plus`

That tests the refined version of the current hypothesis:

> broad MLP QAT is useful, but attention QAT only helps when it is much narrower and later.

## Bottom Line

Today changed the QAT picture again.

The main result is no longer just "QAT helps a little."

It is now:

- **QAT structure matters**
- **MLP-only QAT is currently the best local placement**
- **full-stack QAT is too blunt for this branch**

That is a real narrowing of the search space, and a much better place to continue than generic schedule tuning alone.
