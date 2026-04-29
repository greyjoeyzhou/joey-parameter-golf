# Local CaseOps Lanes Summary

This note summarizes where the **valid / lossless CaseOps** branch stands after the recent local 1x5090 experiments, and how the **non-TTT** and **TTT** lanes currently compare.

Primary references:
- `notes/local-valid-tokenizer-results-2026-04-26.md`
- `notes/local-valid-tokenizer-2h-results-2026-04-26.md`
- `notes/local-caseops-nonttt-batch-2026-04-26.md`
- `notes/local-caseops-nonttt-2h-batch-2026-04-27.md`
- `runs/2026-04-27_CaseOps_SkipGates_ParResid_QKGain50_HessClip015_WD090_Valid3h/`

## Goal

After deciding to stay on the lossless-safe side of the tokenizer question, the working question became:

> how far can we push the valid CaseOps branch without TTT, and does that give us a clear reason to move into a TTT lane next?

## Valid CaseOps Progression So Far

### 1. Best 30-minute non-QAT valid run

Run:
- `2026-04-25_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid`

Result:
- prequant `val_bpb`: `1.21457238`
- quantized `val_bpb`: `1.21949691`

Interpretation:
- this established the first strong valid CaseOps baseline
- at this point, non-QAT CaseOps still beat the valid QAT variants locally

### 2. First 2-hour valid filter

Top runs:

1. `2026-04-26_CaseOps_SkipGates_ParResid_QAT10_QKGain40_Valid2h`
   - `1.11284657`
2. `2026-04-26_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid2h`
   - `1.11297258`

Interpretation:
- at roughly `4.26k` steps, the best CaseOps QAT branch narrowly beat the best CaseOps non-QAT branch
- but the margin was extremely small, so this did not yet settle the lane question decisively

### 3. 30-minute non-TTT technique sweep

Winner:
- `2026-04-26_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_WD095_Valid`
  - `1.22134413`

Interpretation:
- at 30 minutes, neither the WD sweep nor the QK-gain sweep beat the older non-QAT CaseOps baseline
- this suggested the short local filter was too noisy or too undertrained to expose the right ranking for these knobs

### 4. 2-hour non-TTT technique sweep

Winner:
- `2026-04-27_CaseOps_SkipGates_ParResid_QKGain50_HessClip015_WD090_Valid2h`
  - prequant `val_bpb`: `1.10164521`
  - quantized `val_bpb`: `1.11258473`

Interpretation:
- once the same branch was allowed to run for roughly `4.26k` steps, the non-TTT ranking changed materially
- the strongest branch was no longer the old `QKGain45` non-QAT baseline
- it became `QKGain50 + WD090`

### 5. Current best 3-hour non-TTT valid run

Run:
- `2026-04-27_CaseOps_SkipGates_ParResid_QKGain50_HessClip015_WD090_Valid3h`

Result:
- steps: `6353`
- prequant `val_bpb`: `1.08908863`
- quantized `val_bpb`: `1.10178186`

Delta vs the 2-hour version of the same branch:
- `1.11258473 -> 1.10178186`
- improvement: about `-0.01080 bpb`

Interpretation:
- the non-TTT valid CaseOps lane still had meaningful headroom left
- the earlier 2-hour result did not saturate the branch

## Current Best Branches By Lane

### Best non-TTT valid CaseOps branch

- `CaseOps`
- `SkipGates`
- `ParallelResidualStart=7`
- `QK_GAIN_INIT=5.0`
- `HESSIAN_CLIP_LAMBDA=0.15`
- `MUON_WD=0.090`
- `EMBED_WD=0.090`
- `QAT_ENABLED=0`
- `TTT_ENABLED=0`

Best measured local result:
- `1.10178186`

### Best local CaseOps branch that uses QAT

- `CaseOps`
- `SkipGates`
- `ParallelResidualStart=7`
- `QAT10`
- `QK_GAIN_INIT=4.0`
- `HESSIAN_CLIP_LAMBDA=0.15`

Best measured local result:
- `1.11284657`

Current comparison:
- best non-TTT branch is now ahead of the best local CaseOps+QAT branch by about `-0.01106 bpb`

## What The Non-TTT Lane Has Taught Us

### 1. CaseOps is a real positive tokenizer branch once the stack is strong enough

The earlier weak local results for CaseOps were misleading.

Once we moved to:
- stronger architecture
- better quantization settings
- longer runs

CaseOps became the best valid tokenizer option in the repo.

### 2. Step budget matters more than the 30-minute filter suggested

The jump from:
- ~`1060` steps
to
- ~`4260` steps
to
- ~`6350` steps

changed the ranking and the quality materially.

This is one of the biggest lessons from the recent CaseOps work.

### 3. In the non-TTT valid lane, the branch now wants stronger QK gain and modestly higher WD

The current winner is not the old:
- `QKGain45 + WD085`

It is now:
- `QKGain50 + WD090`

That is a useful shift in the local optimum.

### 4. Local recurrence is still not a practical 5090 iteration path

The recurrence smoke for the valid CaseOps branch was effectively unusable:
- only `58` steps in a 10-minute smoke
- ~`35.9 GiB` peak memory

That does **not** mean recurrence is globally bad.
It means:
- recurrence is still a poor local iteration path on this machine for this branch

## What The TTT Lane Still Has Over Us

Even after the 3-hour non-TTT gain, the public CaseOps frontier is still clearly in TTT-enabled stacks.

Relevant public anchors:
- `#1755` `CaseOps + Legal TTT` -> `1.07462`
- `#1729` `CaseOps + phased TTT stack` -> `1.0678`
- `#1738` `CaseOps + pre-quant TTT stack` -> `1.0354` (but rule-risky)

Compared with our current best non-TTT local result:
- local best: `1.10178186`

Gaps:
- vs `#1755`: about `+0.0272 bpb`
- vs `#1729`: about `+0.0340 bpb`

That is large enough that tokenizer choice alone is no longer the main story.

The remaining public gap is mostly in the **adaptation stack**.

## My Current Read On The Two Lanes

### Non-TTT lane

Pros:
- simpler
- easier to reason about
- cleaner legality surface
- still improving with more training time

Cons:
- public CaseOps frontier is not here
- remaining gap to strong CaseOps TTT runs is still substantial

Current verdict:
- worth having pushed this far
- no longer the main lane I would prioritize for large further gains

### TTT lane

Pros:
- this is where the public CaseOps wins live
- most plausible route to close the remaining `0.027-0.034 bpb` gap

Cons:
- more implementation complexity
- more compliance nuance
- for our purposes, we should stay away from the controversial pre-quant TTT route

Current verdict:
- this is the next lane I would push
- but only in the **legal score-first** form first

## Recommendation

I do **not** think we need to spend more time right now squeezing additional non-TTT micro-gains from the valid CaseOps branch.

The non-TTT lane has already done its job:
- it established the best current valid fixed-predictor base
- it taught us the branch wants `QKGain50 + WD090`
- it showed the local ranking changes materially once we approach a more realistic step budget

The next high-value move is:

> keep the current best non-TTT CaseOps branch as the base, and start a **legal score-first TTT** branch on top of it.

## Bottom Line

The current valid CaseOps story is now:

- **best non-TTT local branch:** `CaseOps + SkipGates + ParResid + QKGain50 + HessClip015 + WD090`
- **best measured local result:** `1.10178186`
- **best local CaseOps QAT result:** `1.11284657`
- **best public CaseOps results are still TTT-enabled**

So my recommendation is clear:

- stop prioritizing further non-TTT-only sweeps for now
- move the next experiments into a **legal score-first TTT lane** built on the new best non-TTT base
