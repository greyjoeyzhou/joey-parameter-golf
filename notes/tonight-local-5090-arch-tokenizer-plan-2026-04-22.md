# Tonight Plan: Local 1x5090 Architecture + Tokenizer Sweep

This is the execution plan for tonight's experiments on the local machine.

Constraints:
- hardware: **1x RTX 5090 only**
- environment: **local runs only**
- tokenizer for architecture baseline phase: **fixed `SP8192` only**
- legality baseline: **no TTT**, **no special tokenizer**, **no custom eval-time adaptation**

The goal is to first establish a legal, credible local architecture baseline, then layer tokenizer changes on top of the strongest architecture.

---

## Objectives

### 1. Architecture exploration

Explore current frontier architecture ideas and select a **legal SP8192 no-TTT baseline architecture**.

Requirements:
- stock `SP8192`
- no TTT
- no casefold / CaseOps / custom tokenizer path
- local 1x5090 runnable
- ideally not compile-bound for the entire wallclock on Blackwell

### 2. Architecture baseline set

Pick the **3 most promising architectures** after exploration and run each for **30 minutes** to build a local baseline set.

### 3. Tokenizer upgrades

Take the strongest architecture from step 2 and add tokenizer upgrades:
- Casefold V2
- CaseOps
- optionally one additional tokenizer variant if time remains

Run each tokenizer experiment for **30 minutes**.

---

## Key Decision Rules

### A. We are optimizing for a useful local research baseline, not leaderboard comparability

Because this is 1x5090 local only:
- do not chase 8xH100-only tricks as the primary baseline
- do not make TTT central to the baseline
- prefer stacks that produce actual training steps on this machine within the first few minutes

### B. Baseline must be legal and interpretable

The baseline architecture should avoid:
- TTT
- custom tokenizer transformations
- byte sidecars
- pre-quant adaptation stages

This makes the initial comparison cleaner and gives us a stable base for later tokenizer experiments.

### C. Avoid compile-bound dead ends in the baseline phase

From the earlier 5090 replay tests:
- heavier loop/recurrence stacks can spend most of the wallclock compiling
- the heaviest `SP8192` record stack never reached its first real training step on this machine

So the baseline phase should favor architectures that are:
- strong enough to matter
- simple enough to start training promptly on 1x5090

---

## What We Know From Prior Exploration

### Frontier convergence

Recent tokenizer PRs do not generally introduce a whole new architecture. They mostly place tokenizer ideas on top of an already-strong SP8192 backbone.

The current frontier splits into two families:

### Family 1: legal-ish frontier family

Examples:
- `#1670`
- `#1693`
- `#1729`
- `#1736`

Shared characteristics:
- SP8192
- `11L/512d`-ish dense transformer
- varlen attention + fused MLP lineage
- recurrence / loops
- Muon
- GPTQ-style quantization
- sometimes parallel residuals
- small gates added late
- phased or multi-phase TTT in the full PRs

This is the best source family for a legal no-TTT baseline, because the TTT can be removed while leaving a strong architecture behind.

### Family 2: stronger but riskier pre-quant-TTT family

Examples:
- `#1735`
- `#1738`

Shared characteristics:
- SP8192
- strong recurrence + parallel residuals
- QK-gain tuning
- heavy pre-quant TTT
- 8-GPU federated or parallel adaptation

This family is numerically stronger, but it is the wrong place to start for the local baseline because:
- TTT is central to the gains
- it is less clean legally
- it is more expensive and more compile-heavy on 1x5090

### Outliers we should learn from but not use directly as the baseline

#### `#755` Gravity

Useful because it isolates tokenizer effects with a plain model.

Not the right baseline because:
- it is SP1024, not SP8192
- it is mostly a tokenizer-isolation run
- it does not represent current SP8192 architecture practice

#### `#1707` casefold + SP10240 + FreqGPTQ

Useful because it is simple and no-TTT.

Not the right baseline because:
- it changes tokenizer family and vocab size
- tonight's baseline needs fixed SP8192

---

## Proposed Baseline Architecture Direction

Use the **legal-ish frontier family**, but strip it down to:
- stock `SP8192`
- no TTT
- no custom tokenizer

The baseline architecture family should therefore be built from:
- recurrence / loop depth
- optional parallel residuals
- optional lightweight gates
- Muon
- GPTQ-style quantization

without:
- phased TTT
- pre-quant TTT
- tokenizer transforms

### Recommended baseline starting point

The best initial baseline family is:

**`#1693`-style stack minus casefold minus TTT**

Reason:
- It represents the strongest reasonably clean architecture family in the tokenizer notes.
- Its gains are not purely tokenizer-based.
- It has clear, small architectural additions: recurrence, gate(s), structural tuning.
- It should be easier to downshift into a legal no-TTT local run than the heavier pre-quant-TTT family.

---

## Architecture Candidate Set For Tonight

These are the 3 architecture variants I propose comparing in the baseline phase.

All 3 use:
- stock `SP8192`
- no TTT
- no tokenizer changes
- local 1x5090

### Candidate A: structural base

Purpose:
- the cleanest strong no-TTT base

Shape:
- SP8192
- `11L/512d` frontier-style base
- varlen attention / fused MLP lineage if already in the base
- recurrence / loop depth enabled
- Muon
- GPTQ / SDClip-style quant path if part of the stack
- **no gates**
- **no TTT**

Why:
- gives us the cleanest answer to “how much do recurrence / loop-style structural changes buy locally?”

### Candidate B: structural base + parallel residuals

Purpose:
- test whether parallel residuals still help on local 1x5090 when TTT is removed

Shape:
- Candidate A
- plus parallel residual branch

Why:
- parallel residuals are one of the most common frontier ideas, but we should test them in a local no-TTT setting rather than assuming they transfer cleanly

### Candidate C: structural base + lightweight gates

Purpose:
- test whether the late frontier's small gate additions carry value without the tokenizer and TTT stack around them

Shape:
- Candidate A
- plus AttnOutGate and/or one lightweight residual/attention gate
- no TTT

Why:
- gates are cheap to add and may recover signal quality after recurrence or quantization, but their value in the clean local baseline needs to be measured directly

---

## Run Plan

## Phase 1: exploration and baseline selection

### Step 1. Docs/code exploration

Before running anything, inspect:
- local notes on tokenizer frontier
- recent SP8192 record READMEs in `parameter-golf/records/track_10min_16mb/`
- the nearest non-tokenizer architectural ancestors of `#1670/#1693/#1729/#1736`

Deliverable:
- one concrete implementation candidate for Candidate A, B, C

### Step 2. 20-minute exploration run

Run **one 20-minute exploration run** using Candidate A.

Purpose:
- verify the chosen baseline family actually trains on 1x5090
- measure compile overhead
- check whether we get real training steps soon enough for 30-minute runs to be meaningful

Success criteria:
- gets into real training promptly
- does not spend the full wallclock compiling
- stable enough to use as the architecture anchor

If Candidate A fails badly, fall back to a simpler no-gate/no-parallel-residual variant before starting the 30-minute sweep.

---

## Phase 2: three 30-minute architecture baselines

After the exploration run, execute:

### Run 1. Candidate A
- 30 minutes
- stock SP8192
- no TTT

### Run 2. Candidate B
- 30 minutes
- stock SP8192
- no TTT

### Run 3. Candidate C
- 30 minutes
- stock SP8192
- no TTT

Metrics to compare:
- whether the run gets into training quickly
- step time after warmup
- tokens/sec
- train loss trajectory
- any validation metric we can afford in the wallclock
- artifact size if we serialize
- subjective stability / fragility on 1x5090

Decision after Phase 2:
- choose the **strongest and most stable** architecture as the tokenizer base

---

## Phase 3: tokenizer upgrades on top of the winner

Take the strongest architecture from Phase 2 and keep everything else fixed.

Then run tokenizer variants for 30 minutes each.

### Tokenizer Run T1: Casefold V2

Use the strongest architecture from Phase 2 with:
- Casefold V2 tokenizer/data
- no TTT

Why first:
- highest information value
- strongest “missing combination” from the notes
- most likely to show whether V2 tokenizer engineering still matters once paired with a modern structural stack

### Tokenizer Run T2: CaseOps

Use the same winning architecture with:
- CaseOps tokenizer/data
- byte-exact accounting path if needed for the local run
- no TTT

Why second:
- cleaner legality story than casefold
- useful comparison against Casefold V2

### Tokenizer Run T3: optional extra tokenizer if time remains

Only if the first two tokenizer runs are healthy and there is time.

Priority order:
1. casefold `SP10240`
2. another CaseOps variant
3. no third tokenizer run if the first two already answer the key question

I do **not** recommend building a brand-new refill tokenizer tonight.

---

## Metrics To Capture For Every Run

At minimum:
- run ID
- architecture tag
- tokenizer tag
- wallclock budget
- tokens/sec
- stable step ms after warmup
- validation loss / bpb if run computes it
- artifact bytes if serialization happens
- notes on compile overhead and any failure mode

For tokenizer runs specifically:
- relative change vs the winning baseline architecture
- whether throughput cost is acceptable
- whether any gain looks likely to survive quantization

---

## Decision Logic Tonight

### If exploration run is compile-bound

Then simplify the architecture baseline immediately:
- remove gates first
- then remove parallel residuals if necessary
- keep recurrence only if it still reaches training promptly

### If Candidate B or C is slower but clearly stronger

Prefer the architecture that gives the best local quality/speed tradeoff, not necessarily the fastest one.

### If tokenizer gains are small but consistent

That is still a useful result. The point of tonight is to identify:
- what the right local no-TTT architecture base is
- whether tokenizer gains remain visible on top of it

---

## Most Likely Outcome I Expect

My current expectation is:
- Candidate A or C will win the architecture phase
- the heaviest recurrence + loop variants will still be too compile-heavy on 1x5090 if pushed too far
- Casefold V2 will likely show a stronger delta than raw CaseOps in the no-TTT local setting
- CaseOps may become more attractive again later if we decide to move toward stronger or more legality-focused stacks

---

## Concrete Proposed Run Order

1. Explore records/docs and choose exact Candidate A, B, C implementations.
2. Run **one 20-minute exploration run** on Candidate A.
3. If healthy, run **three 30-minute architecture baselines**:
   - A
   - B
   - C
4. Choose the best architecture.
5. Run **two 30-minute tokenizer experiments** on the winner:
   - Casefold V2
   - CaseOps
6. If there is still budget and momentum, run one optional third tokenizer experiment.

---

## Review Questions

Before execution, confirm these assumptions:

1. Is it correct that the baseline phase should avoid **all** TTT, including any phased or score-first variant?
2. For the tokenizer phase, should we also keep **no TTT**, or do you want tokenizer experiments to be allowed to add TTT later tonight?
3. Do you want the 30-minute runs to include quantization / serialization, or are we primarily optimizing for training-phase comparison?
4. For the tokenizer phase, do you want to stop at Casefold V2 and CaseOps, or explicitly include casefold `SP10240` tonight too?

---

## My Recommendation

Approve this plan with one default interpretation:

- **no TTT at all tonight**, including tokenizer runs

Reason:
- it keeps the architecture and tokenizer effects interpretable
- it avoids legality/stack confounds
- it is a better fit for 1x5090 local-only exploration

If we get a strong winner tonight, the next session can reintroduce TTT selectively.
