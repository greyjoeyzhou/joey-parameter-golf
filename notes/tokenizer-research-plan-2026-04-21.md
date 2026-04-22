# Tokenizer Research Plan

This plan turns the two tokenizer notes into an execution-oriented research roadmap for this repo.

Primary source notes:
- `notes/tokenizer-what-techniques-exist-2026-04-20.md`
- `notes/tokenizer-how-to-think-about-them-2026-04-20.md`

## Goal

Find tokenizer changes that improve `val_bpb` under actual parameter-golf constraints, not just tokenizer compression in isolation.

Specifically:
- identify which tokenizer families produce real gains on fixed training stacks
- separate tokenizer gain from stack gain
- prioritize ideas that can plausibly survive quantization and the 16 MB artifact budget

## Working Thesis

Tokenizer gain in this competition mainly comes from three mechanisms:
- coverage gain
- parameter-density gain
- segmentation disambiguation

The strongest design rule from the notes is:

> augment, do not replace

That means prioritizing Layer 1 and Layer 2 ideas:
- Layer 1: cross-byte equivalence or normalization the tokenizer cannot learn by itself
- Layer 2: correcting BPE/SP greedy segmentation mistakes

Avoid Layer 3 ideas that try to replace BPE's own competence:
- full vocab replacement
- per-letter cap markers
- high-frequency suffix markers like `<ING>`

## Research Questions

### Q1. Which existing tokenizer family gives the best standalone gain?

Compare on as fixed a stack as possible:
- stock `SP8192`
- Casefold V2
- CaseOps
- casefold `SP10240`

Key output:
- true tokenizer-only delta in `val_bpb`
- throughput impact
- artifact-size impact

### Q2. Which tokenizer family scales best when paired with stronger stacks?

After establishing tokenizer-only results, test the best candidate with:
- FreqGPTQ
- stronger legal TTT
- optionally Pre-Quant TTT where appropriate

Key output:
- whether tokenizer signal survives once the rest of the stack gets stronger

### Q3. What is the next original tokenizer frontier beyond case handling?

The notes strongly suggest the next frontier is not more case tricks but fragment ambiguity:
- `▁perman`
- `▁pict`
- `▁somet`
- `▁appar`
- `▁perpet`

Key output:
- whether fragment-driven refill or improved segmentation scoring can beat plain casefold

## Research Principles

- Keep tokenizer experiments tokenizer-heavy and architecture-light first.
- Reuse upstream artifacts and known-good records before inventing new pipelines.
- Change one dimension at a time.
- Prefer measured deltas over theory-first complexity.
- Use local 5090 runs for smoke tests and throughput checks, not final conclusions.

## Phase 0: Measurement Setup

Before starting new tokenizer work, standardize tokenizer metrics across runs.

For every tokenizer experiment, record:
- `val_bpb`
- `val_loss`
- tokens/sec
- token count on validation
- bytes/token on validation
- byte fallback rate
- embedding parameter count
- final artifact bytes

Also record tokenizer-specific diagnostics:
- top fragment tokens by NLL
- punctuation NLL
- uppercase/titlecase token behavior
- non-ASCII or byte-fallback behavior

Deliverables:
- a tokenizer experiment scoreboard
- one lightweight analysis script or notebook for per-token-class diagnostics

## Phase 1: Reproduce Existing Tokenizer Families

Goal: establish a clean baseline for known tokenizer families with minimal stack confounds.

### 1.1 Fixed-stack comparison set

Run the same or near-identical trainer stack across:
- stock `SP8192`
- Casefold V2 artifact from PR `#1585`
- CaseOps artifact from PR `#1729`
- casefold `SP10240` artifact from PR `#1707`

Recommended priority:
1. Casefold V2
2. CaseOps
3. casefold `SP10240`
4. Gravity

Rationale:
- Casefold and CaseOps are the cleanest, most transferable routes for this repo.
- `SP10240` is worth checking, but only in the casefolded setting.
- Gravity is promising but more expensive and less validated at SP8192-like scale.

### 1.2 Success criteria

At least one tokenizer family should beat stock `SP8192` clearly on `val_bpb` while staying reasonable on:
- throughput
- artifact size
- implementation complexity

## Phase 2: Pair the Best Tokenizer with Low-Risk Stack Gains

Goal: find out whether the tokenizer still matters once stronger training and quantization tricks are added.

Best first combinations:
- best tokenizer from Phase 1 + FreqGPTQ from PR `#1707`
- best tokenizer from Phase 1 + stronger legal TTT
- CaseOps + Pre-Quant TTT from PR `#1735` / `#1738`

Rules:
- only add one stack component at a time
- keep run notes explicit about what changed vs the previous run

Success criteria:
- tokenizer improvement survives when paired with stronger stack components

## Phase 3: Original Tokenizer Work

Goal: pursue the most promising new tokenizer contribution.

### 3.1 Fragment-driven refill

This is the highest-priority original direction.

Approach:
- start from top high-NLL fragments from EDA
- enumerate plausible whole-word or lower-ambiguity expansions
- score candidates by BPB improvement using a refill-style pipeline

Examples:
- `▁perman` -> `▁permanent`, `▁permanently`, `▁permanence`
- `▁pict` -> `▁picture`, `▁pictured`, `▁pictorial`
- `▁somet` -> `▁something`, `▁sometimes`, `▁somewhat`

Why this is attractive:
- directly targets the largest remaining tokenizer pain point in the notes
- stays in Layer 2 rather than inventing a new tokenizer family

### 3.2 Selective morphological refill

Only test low-frequency long-tail variants.

Do not use generic suffix markers like `<ING>`.

The acceptable version is:
- add a small number of complete low-frequency word forms if they score well under BPB

The unacceptable version is:
- rule-based suffix injection for high-frequency endings

### 3.3 Non-uniform Viterbi scoring

Test segmentation-scoring improvements without changing vocab size.

Why it is attractive:
- low implementation surface area
- keeps the tokenizer family recognizable
- may produce gain without retraining a whole tokenizer

Success criteria for Phase 3:
- beat plain casefold on the same training stack

## Phase 4: Casefold Vocab-Size Sweep

Goal: answer whether casefold shifts the optimal vocab size upward.

Suggested sweep:
- `SP8192`
- `SP10240`
- `SP12288`

Question:
- does casefold create enough parameter-density gain that a larger vocab becomes worth the embedding budget?

Important constraint:
- do not run the same sweep for vanilla mixed-case BPE unless a new reason appears; the notes already argue vanilla `SP8192` is likely near the sweet spot.

## Phase 5: Higher-Risk Exploratory Combinations

Only do this after the earlier phases.

Candidates:
- Gravity + casefold
- CaseOps + Gravity
- byte-weighted pretraining loss

These are interesting, but they are not the first place to spend effort.

## Recommended Order of Execution

1. Reproduce Casefold V2 on a fixed stack.
2. Reproduce CaseOps on the same stack.
3. Reproduce casefold `SP10240` on the same stack.
4. Apply FreqGPTQ to the best of those.
5. Apply stronger TTT to the best of those.
6. Start fragment-driven refill experiments.
7. Run the casefold vocab-size sweep.
8. Explore Gravity-on-casefold if earlier phases justify it.

## Compute Allocation

### Local 5090

Use for:
- smoke tests
- throughput sanity checks
- byte-accounting validation
- debugging tokenizer integration

Do not treat it as the final word on leaderboard-quality ranking.

### 1x H100

Use for:
- medium-cost filtering
- tokenizer-only comparison runs
- deciding which families are worth scaling up

### 8x H100

Use only for:
- top tokenizer candidates after stack choice is mostly settled
- serious submission-quality runs

## Success Criteria by Phase

### Phase 1

One tokenizer family clearly beats stock `SP8192` on fixed-stack `val_bpb`.

### Phase 2

The tokenizer gain is still visible after stronger stack components are added.

### Phase 3

A refill or segmentation idea beats plain casefold on the same stack.

### Submission Readiness

The tokenizer improvement survives:
- quantization
- artifact-size accounting
- realistic trainer stack changes

## First Concrete Steps

### Path A: fastest learning

1. Pull or reconstruct Casefold V2 artifacts.
2. Run a fixed-stack comparison against stock `SP8192`.
3. Repeat for CaseOps.
4. Repeat for casefold `SP10240`.

### Path B: fastest shot at a strong submission stack

1. Take the best result from Path A.
2. Add FreqGPTQ.
3. Add stronger TTT.
4. Re-evaluate artifact bytes and quantized `val_bpb`.

### Path C: best original research contribution

1. Build a fragment candidate list from existing EDA outputs.
2. Implement a refill scoring loop.
3. Test targeted fragment refill on top of casefold.
4. Write up what token classes improved and why.

## Recommended Immediate Focus

If the goal is practical progress rather than broad exploration, start here:

1. Casefold V2 reproduction
2. CaseOps reproduction
3. FreqGPTQ on the better of those two
4. Fragment-driven refill

Why this order:
- lowest implementation risk
- highest immediate learning value
- strongest chance of producing either a usable stack improvement or a genuinely new tokenizer result

## Things Explicitly Not Worth Doing First

- full vocab replacement
- capcode-style per-letter marker systems
- high-frequency suffix marker schemes
- large vocab sweeps without casefold
- CJK-heavy vocab work for FineWeb unless new evidence appears
- Scylla-style low-compression tokenizers for this benchmark target

## Decision Gate

Before starting any new tokenizer idea, ask:

1. Is this Layer 1 or Layer 2, not Layer 3?
2. Can eval still account for original bytes correctly?
3. Does it merge things that should share embedding exposure?
4. If it adds markers, are those markers much rarer than what they replace?

If any answer is no, do not prioritize it.

## Suggested Deliverables

- `notes/tokenizer-experiment-scoreboard.md`
- one per-token-class tokenizer analysis script or notebook
- 3-5 reproducible run directories for the first comparison set
- one short writeup on fragment ambiguity and refill ROI

## Summary

The highest-ROI tokenizer program for this repo is:
- first reproduce the strongest existing augmentation routes
- then identify the best one under a fixed stack
- then push on fragment ambiguity with refill and segmentation work

The notes strongly suggest that the next real tokenizer gain will come less from inventing another normalization trick and more from fixing the fragment-level ambiguity that stock SP/BPE still leaves behind.
