# Tonight Follow-Up Plan: Local 1x5090 After The First Sweep

This is the next-step plan after the first local 1x5090 architecture + tokenizer sweep.

Ground truth from the last session:
- best clean local architecture: `SP8192 + SkipGates + no loops + no parallel residuals + no TTT`
- best tokenizer on that architecture: `Casefold V2`
- CaseOps did not beat the same baseline in the no-TTT local regime

Primary reference notes:
- `notes/local-5090-arch-tokenizer-results-2026-04-23.md`
- `notes/tokenizer-research-plan-2026-04-21.md`

## New Default Starting Point

Promote this to the new local baseline for follow-up work:

- architecture: `SkipGates`
- tokenizer: `Casefold V2`
- no loops
- no parallel residuals
- no TTT
- no QAT

This is the best local stack discovered so far because it combines:
- the best structural result from the architecture sweep
- the best tokenizer result from the tokenizer sweep
- essentially unchanged throughput versus the stock SP8192 winner

## Goal For Tonight

Do not reopen broad exploration.

The goal tonight is narrower:

1. strengthen the new `Casefold V2 + SkipGates` default
2. test whether a small structural add-on flips positive once Casefold is present
3. defer CaseOps unless a stronger structural branch justifies revisiting it

## Working Assumptions

- hardware remains `1x RTX 5090`
- local runs only
- still no TTT tonight
- still no QAT tonight
- keep the same `2026-04-06` SP8192 script family and local harness unless a run specifically requires a tokenizer-sidecar wrapper

## Recommended Ordering

The highest-EV order is:

1. cheap positive-EV improvements on the new default
2. interaction tests between Casefold and structure
3. tokenizer broadening only after those

This matches the direction from the prior notes:
- first strengthen the best current branch
- then test whether structural ideas change sign under the better tokenizer
- only then broaden into new tokenizer branches

## Phase 1: Cheap Positive-EV Upgrades On The Winner

These should go first because they are low-risk and informative.

### Run 1. Casefold V2 + SkipGates + HessianClip

Why first:
- we explicitly disabled `HESSIAN_CLIP_LAMBDA` during the first sweep
- this is one of the easiest ways to improve quantized quality without changing the architecture family
- it targets the compression path directly, which matters in this branch because we are using GPTQ post-training quantization rather than QAT

Suggested variants:
- `HESSIAN_CLIP_LAMBDA=0.15`
- if positive, follow with `HESSIAN_CLIP_LAMBDA=0.30`

Keep fixed:
- `Casefold V2`
- `SkipGates=1`
- `NUM_LOOPS=0`
- `PARALLEL_RESIDUAL_START=-1`
- `TTT_ENABLED=0`

### Run 2. Casefold V2 + SkipGates + QK gain sweep

Why second:
- `QK_GAIN_INIT` is cheap to sweep and often changes optimization behavior noticeably in these small frontier stacks
- it is a low-surface-area change that keeps the branch interpretable

Suggested values:
- `QK_GAIN_INIT=4.5`
- `QK_GAIN_INIT=5.0`

Current baseline value:
- `QK_GAIN_INIT=4.0`

Decision rule:
- if either beats the baseline cleanly, keep the better one as the new default before moving to structural interaction tests

## Phase 2: Structural Interaction Tests Under Casefold

Tonight's first sweep showed:
- parallel residuals were bad on stock SP8192
- recurrence-heavy loops were too compile-expensive locally

But that does **not** mean they stay bad under Casefold.

### Run 3. Casefold V2 + SkipGates + ParallelResidual

Why this is important:
- we only tested parallel residuals on stock SP8192
- Casefold changes token statistics and may change whether a structural branch is worthwhile

Suggested setting:
- `PARALLEL_RESIDUAL_START=7`
- keep loops off
- keep skip gates on

This is the most important structural interaction test after the cheap Phase 1 runs.

### Run 4. Casefold V2 + SkipGates + recurrence-lite

Why this is later and conditional:
- the looped branch was the local pain point in the first sweep
- it should only be retried in a much lighter form

The goal is not "turn loops back on exactly as before".
The goal is "see whether a smaller recurrence footprint is usable locally".

Suggested approach:
- do a **10-minute smoke first**
- only run the full version if the smoke gets into real training quickly

Suggested recurrence-lite starting point:
- `NUM_LOOPS=1`
- `LOOP_START=4`
- `LOOP_END=5`
- try either:
  - `ENABLE_LOOPING_AT=0.0` if we want one upfront compile instead of a mid-run phase change
  - or a much later activation threshold if upfront looping is too slow

Success criterion:
- it must not become compile-bound for most of the wallclock

If the smoke fails, stop and do not spend a full run on it.

## Phase 3: Tokenizer Broadening

Only start this after Phase 1 and at least the parallel-residual interaction test are done.

### Run 5. Casefold + SP10240

Why this is the right tokenizer broadening move:
- the earlier notes already suggest `SP10240` only becomes really interesting when paired with casefold-style normalization
- this is the next natural tokenizer step once plain Casefold V2 is established

This should be tested on the best structural stack discovered in Phases 1-2, not on a weaker branch.

### CaseOps tonight?

Not as a priority.

Reason:
- the clean no-TTT result was flat to slightly negative relative to the SP8192 skip-gated base
- the best known CaseOps results in the notes depend on stronger surrounding stacks than what we intentionally held fixed locally

So the right policy is:
- do **not** spend an early run tonight on CaseOps again
- only revisit CaseOps if one of the stronger structural upgrades beats the current Casefold branch and we want to test whether CaseOps benefits more from that stronger stack

## Concrete Run Order For Tonight

If we want a practical 1x5090 plan, this is the order I recommend.

### Must-run set

1. `CasefoldV2_SkipGates_HessClip015`
2. `CasefoldV2_SkipGates_QKGain45`
3. `CasefoldV2_SkipGates_ParResid`

### Strongly recommended next

4. `CasefoldV2_SkipGates_QKGain50`
5. `CasefoldV2_SkipGates_RecurLite_smoke`

### Only if the above are healthy and time remains

6. `CasefoldV2_SP10240_<best-structure>`
7. `CaseOps_<best-stronger-structure>` only if we explicitly want to re-check CaseOps under a stronger branch

## Decision Rules

### If HessianClip helps

- keep the better HessianClip value as the new default
- then test structural interaction on top of that improved default

### If a QK-gain sweep helps

- update the default to the better QK gain before continuing

### If parallel residuals are still negative under Casefold

- deprioritize parallel residuals locally
- move to recurrence-lite smoke instead

### If recurrence-lite is still compile-bound

- stop recurrence work for now on 1x5090
- shift the remaining budget to tokenizer broadening or quantization tuning

### If Casefold + SP10240 beats Casefold + SP8192

- that becomes the new tokenizer branch to develop further

## What Not To Do Tonight

- do not re-open CaseOps-first exploration
- do not start a QAT branch tonight
- do not build a new fragment-refill tokenizer tonight
- do not run the heavy looped branch again in its previous form
- do not do broad stock-SP8192 structural sweeps without Casefold

## Expected Best Outcome

The most likely strong outcome tonight is:

- `Casefold V2 + SkipGates`
- plus either a modest HessianClip improvement or a slightly better `QK_GAIN_INIT`

The most likely structural upside branch after that is:

- `Casefold V2 + SkipGates + ParallelResidual`

but only if Casefold changes the sign of the parallel-residual tradeoff.

## Bottom Line

Do **not** restart from stock SP8192.

Continue from the best thing we already found:

- `Casefold V2 + SkipGates + no loops + no parallel residuals + no TTT`

Then test, in order:

1. HessianClip
2. QK gain sweep
3. ParallelResidual under Casefold
4. recurrence-lite smoke
5. Casefold + SP10240

That is the highest-EV continuation of the local research program right now.
