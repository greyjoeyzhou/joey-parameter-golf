# Local 1x5090 Follow-Up Results

This note summarizes the second local follow-up batch run after the first architecture + tokenizer sweep.

Primary references:
- `notes/local-5090-arch-tokenizer-results-2026-04-23.md`
- `notes/tonight-followup-plan-2026-04-24.md`

## Goal

Start from the best local branch discovered in the first sweep:

- `Casefold V2`
- `SkipGates`
- no loops
- no parallel residuals
- no TTT

Then answer three concrete questions:

1. does Hessian-aware clipping help this branch?
2. does a small QK gain sweep help?
3. do structural ideas that were bad on stock SP8192 become good once Casefold is present?

## Starting Point

Reference winner from the first batch:

- run: `2026-04-22_CasefoldV2_NoTTT_SkipGates30m_v3`
- prequant `val_bpb`: `1.19837140`
- quantized `val_bpb`: `1.20385604`

This is the baseline every follow-up run should be compared against.

## Runs Executed

### 1. HessianClip 0.15

Run:
- `runs/2026-04-24_CasefoldV2_SkipGates_HessClip015/`

Change vs baseline:
- `HESSIAN_CLIP_LAMBDA=0.15`

Results:
- prequant `val_bpb`: `1.19848597`
- quantized `val_bpb`: `1.20375957`

Interpretation:
- essentially a wash
- very slight quantized improvement over the prior baseline (`1.20385604 -> 1.20375957`)
- not enough to treat as a major positive branch by itself

## 2. QK gain 4.5

Run:
- `runs/2026-04-24_CasefoldV2_SkipGates_QKGain45/`

Change vs baseline:
- `QK_GAIN_INIT=4.5`

Results:
- prequant `val_bpb`: `1.19775288`
- quantized `val_bpb`: `1.20297049`

Interpretation:
- clean improvement over the original Casefold + SkipGates baseline
- this is the best no-parallel-residual branch in the follow-up batch

## 3. Casefold + SkipGates + ParallelResidual

Run:
- `runs/2026-04-24_CasefoldV2_SkipGates_ParResid/`

Change vs baseline:
- `PARALLEL_RESIDUAL_START=7`

Results:
- prequant `val_bpb`: `1.19661129`
- quantized `val_bpb`: `1.20199872`

Interpretation:
- this is the best result of the follow-up batch
- more importantly, it changes the qualitative conclusion from the first sweep:

> parallel residuals were bad on stock SP8192, but positive once Casefold V2 is present

This is the strongest new finding of the follow-up session.

## 4. QK gain 5.0

Run:
- `runs/2026-04-24_CasefoldV2_SkipGates_QKGain50/`

Change vs baseline:
- `QK_GAIN_INIT=5.0`

Results:
- prequant `val_bpb`: `1.19858520`
- quantized `val_bpb`: `1.20359605`

Interpretation:
- worse than `QK_GAIN_INIT=4.5`
- slightly better than the original Casefold baseline, but not the best QK-gain setting

So for this branch:
- `QK_GAIN_INIT=4.5` looks better than `5.0`

## 5. Recurrence-lite smoke

Run:
- `runs/2026-04-24_CasefoldV2_SkipGates_RecurLite_smoke/`

Change vs baseline:
- `NUM_LOOPS=1`
- looping enabled from step 0

Result:
- still too slow locally
- around `28k tok/s`
- timed out at the smoke-run limit

Interpretation:
- recurrence remains a bad local 1x5090 direction for this research loop
- even a lighter setup is still far too slow relative to the non-recurrent branches

Conclusion:
- stop recurrence work locally for now

## 6. Optional tokenizer broadening: Casefold + SP10240

Run:
- `runs/2026-04-24_CasefoldSP10240_SkipGates30m/`

Change vs baseline:
- tokenizer/dataset switched from Casefold `SP8192` to Casefold `SP10240`
- architecture stayed on the same skip-gated no-TTT no-loop base

Results:
- prequant `val_bpb`: `1.19941983`
- quantized `val_bpb`: `1.20458654`
- total bytes: `16,690,822`

Interpretation:
- worse than the Casefold SP8192 branch on quality
- worse than the best follow-up branch
- larger artifact, above the 16 MB target

Conclusion:
- for this local branch, `SP10240` should not replace `SP8192`

## Ranking

Quantized `val_bpb` ranking from the follow-up batch:

1. `CasefoldV2_SkipGates_ParResid` -> `1.20199872`
2. `CasefoldV2_SkipGates_QKGain45` -> `1.20297049`
3. `CasefoldV2_SkipGates_QKGain50` -> `1.20359605`
4. `CasefoldV2_SkipGates_HessClip015` -> `1.20375957`
5. original `CasefoldV2_NoTTT_SkipGates30m_v3` -> `1.20385604`
6. `CasefoldSP10240_SkipGates30m` -> `1.20458654`

## Main Findings

### 1. The branch improved again

The new best local branch is now:

- `Casefold V2`
- `SkipGates`
- `ParallelResidual`
- no loops
- no TTT

Quantized `val_bpb` improved from:

- `1.20385604`

to:

- `1.20199872`

### 2. Casefold changes the sign of structural ideas

This is the most important research conclusion from the follow-up run set.

From the first sweep:
- parallel residuals on stock SP8192 were clearly bad

From this sweep:
- parallel residuals on top of Casefold V2 became the best result

That means the tokenizer and structure are not independent here.

### 3. QK gain tuning helps, but less than the structural interaction

`QK_GAIN_INIT=4.5` helped.

But the gain from:
- Casefold + ParallelResidual

was larger and more important than the pure QK-gain micro-tuning.

### 4. Recurrence remains locally uncompetitive

Even the recurrence-lite smoke is far too slow on this machine for this research loop.

This does not mean recurrence is globally bad.
It means:
- recurrence is not currently a good local 1x5090 iteration path for this branch

### 5. Casefold SP10240 is not the next move here

The SP10240 casefold branch underperformed and increased artifact size.

So the best continuation is still:
- better structure on top of Casefold SP8192

not:
- larger casefold vocab

### 6. Why Casefold seems to beat CaseOps locally

This is the main tokenizer interpretation question raised by the local runs.

At a high level:

- **Casefold removes distinctions**
- **CaseOps re-encodes distinctions**

That sounds subtle, but for this competition setup it is a major difference.

#### Casefold

Casefold is a lossy normalization route.

Its strength in parameter-golf is that it directly buys:
- coverage gain
- parameter-density gain

In practice that means:
- fewer duplicated embedding rows for case variants
- more training exposure per surviving row
- fewer tokens spent on distinctions the model may not have time to learn well anyway

In the short-budget local regime, that is exactly the kind of simplification that should help.

#### CaseOps

CaseOps is lossless and bijective.

Its strength is conceptual cleanliness:
- exact reversibility
- strong original-byte accounting story
- cleaner compliance intuition

But the model does not get that for free. It now has to learn a mini control language:
- `<TITLE>the`
- `<ALLCAPS>nasa`
- `<CAPNEXT>` patterns

So even though the transformation is elegant, the learning problem can be harder.

#### Why this likely favors Casefold on our current branch

Our local branch is intentionally simple:
- no TTT
- no recurrence
- short wallclock
- small structural improvements only

That is the regime where I would expect a tokenizer that **reduces learning burden immediately** to win.

Casefold does that.
CaseOps preserves more information, but also preserves more work for the model.

#### Another important caveat: artifact quality is not equal

We are also not comparing two equally engineered tokenizer artifacts.

Casefold V2 includes:
- slot cleanup
- BPB-scored refill
- punctuation swap
- unigram / Viterbi decoding
- byte-fallback reduction

CaseOps v1 is cleaner conceptually, but less aggressively optimized as a tokenizer artifact.

So part of the observed gap may simply be:
- the Casefold tokenizer artifact is currently better tuned for this benchmark than the CaseOps artifact we tested

#### Current hypothesis

The best current interpretation is:

- **Casefold is the right default tokenizer branch for local no-TTT work**
- **CaseOps should only be revisited once the surrounding architecture is stronger**

That matches both:
- the local empirical results in this repo
- the upstream pattern where the strongest CaseOps result sits on top of a much stronger stack than the one we intentionally used locally

## Recommended New Default

Promote this branch to the new local default:

- `CasefoldV2_SkipGates_ParResid`

That is now the best local no-TTT result in the repo.

## Recommended Next Experiments

The next experiments should now continue from the new default rather than the older skip-gated base.

Best next steps:

1. `CasefoldV2_SkipGates_ParResid + QKGain45`
2. `CasefoldV2_SkipGates_ParResid + HessianClip`
3. `CaseOps` only if we specifically want to test whether the stronger structural branch rescues it

What should *not* be prioritized next:
- recurrence
- SP10240 on this branch
- reopening stock-SP8192 architecture sweeps

## Bottom Line

The follow-up batch sharpened the research direction:

- the branch to continue is no longer just `Casefold V2 + SkipGates`
- it is now **`Casefold V2 + SkipGates + ParallelResidual`**

That is the strongest local no-TTT result we have so far, and it gives a much clearer continuation path for the next batch.
