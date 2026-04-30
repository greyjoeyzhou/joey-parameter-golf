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

## Winning Baseline: Exact Setup

The winning baseline is not just "SP8192 + skip gates" in the abstract. It is a very specific localized version of the `2026-04-06` SP8192 script family with several features intentionally left on and several intentionally stripped out.

### Kept from the source script family

- `11` layers
- `d_model=512`
- `8` attention heads, `4` KV heads
- `MLP_MULT=4.0`
- tied embeddings
- `QK_GAIN_INIT=4.0`
- `xsa_last_n=11` (XSA on all layers in this script family)
- Muon with row normalization
- `MUON_WD=0.085`, `EMBED_WD=0.085`
- EMA with `EMA_DECAY=0.997`
- GPTQ quantization path
- `MATRIX_BITS=6`, `EMBED_BITS=8`
- SDClip-style quantization (`MATRIX_CLIP_SIGMAS=12.85`, `EMBED_CLIP_SIGMAS=20.0`)
- `TRAIN_SEQ_LEN=2048`
- `TRAIN_BATCH_TOKENS=786432`

### Explicitly removed or disabled for the local baseline

- `TTT_ENABLED=0`
- `NUM_LOOPS=0`
- `PARALLEL_RESIDUAL_START=-1`
- `HESSIAN_CLIP_LAMBDA=0.0`

### Explicitly enabled for the winning baseline

- `SKIP_GATES_ENABLED=1`
- `WARMUP_STEPS=1`
- `GPTQ_CALIBRATION_BATCHES=1`

### Why this baseline won

It kept the strong modern SP8192 trainer skeleton, but removed the two local-5090 troublemakers:
- recurrence / loops
- parallel residuals

Then it added one very small structural change:
- learned skip gates in the decoder skip path

That change is almost free in parameter terms. Comparing model parameter counts from the logs:

- plain base: `35,938,904`
- skip-gated base: `35,941,464`

So the winning tweak only adds `2,560` parameters.

### Local performance profile of the winning baseline

From `runs/2026-04-22_SP8192_NoTTT_SkipGates30m/`:

- steps completed before wallclock cap: `1049`
- train wallclock to stop: `1,788,213 ms`
- effective step time: about `1705 ms/step`
- throughput during training: about `460-461k tok/s`
- prequant `val_bpb`: `1.21760974`
- quantized `val_bpb`: `1.22334557`
- total submission bytes: `16,064,915`

### What this suggests we should try next on the architecture side

The baseline note should be detailed enough to support the next structural decisions. Based on tonight's sweep, the highest-EV architectural updates to test next are:

1. **Skip gates + Hessian-aware SDClip**
   - the source script already supports `HESSIAN_CLIP_LAMBDA`
   - this is a zero-byte or near-zero-byte change worth testing on the winning base

2. **Skip gates + Casefold V2 as the new default starting point**
   - this is now the best local stack discovered in this session

3. **Skip gates + a much lighter recurrence schedule**
   - if recurrence is retried, it should be a later, weaker, or less frequent version than tonight's looped setup
   - the current looped variant is too compile-heavy to be the anchor on 1x5090

4. **Skip gates + parallel residuals**
   - tonight only tested parallel residuals without skip gates
   - it is still possible that skip gates stabilize the same regime enough to change the sign of the delta

5. **QK-gain and optimizer micro-sweeps**
   - `QK_GAIN_INIT`, `MUON_WD`, and related low-surface-area knobs remain available if we want cheaper local optimization before changing larger structure

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

## Training And Quantization Stack

The architecture summary is easier to reason about if the training and compression stack is documented explicitly.

### Optimizer split

This script family is not "Muon everywhere". It uses a mixed optimizer setup:

- **Muon** for 2D matrix weights
  - attention and MLP matrices
  - `MATRIX_LR=0.02`
  - `MUON_WD=0.085`
  - row normalization enabled
- **AdamW** for token embeddings
  - `EMBED_LR=0.6`
  - `EMBED_WD=0.085`
- **AdamW** for scalar and control tensors
  - `SCALAR_LR=0.02`
  - `ADAM_WD=0.02`
  - this includes norms, scales, residual-mix parameters, `q_gain`, and `skip_gates`
- **Adam** for the untied LM head when present
  - `HEAD_LR=0.008`

So the right mental model is:

- Muon-centered training for the heavy matrix parameters
- Adam/AdamW for embeddings and low-dimensional control tensors

### EMA

All of tonight's successful runs kept EMA enabled:

- `EMA_DECAY=0.997`

The logged pre-quantization validation metric is the post-EMA result.

### Quantization path

We are **not** using QAT in this branch.

What happens instead is:

1. train in the normal floating-point path
2. apply EMA
3. collect Hessians on a short calibration set
4. run mixed GPTQ quantization
5. serialize the quantized artifact

This means:

- no fake-quantized forward pass during training
- no QAT loss shaping
- no quantization-aware training branch

### Quantization layout

The source script uses mixed post-training quantization:

- most attention and MLP matrices: **GPTQ int6**
- token embeddings: **int8**
- selected control tensors remain passthrough float tensors
  - `q_gain`
  - attention scales
  - MLP scales
  - residual-mix tensors
  - skip weights / skip gates when enabled

Compression is then applied to the quantized artifact:

- byte shuffle
- Brotli

### Why this matters for follow-up research

Because tonight's winning branch is:

- no TTT
- no QAT
- Muon-centered training
- GPTQ-style post-training quantization

the cleanest next experiments are:

1. small optimizer / QK-gain / WD changes
2. Hessian-aware clip tuning
3. small structural changes
4. tokenizer changes on top of the best structure

It is **not** yet time to introduce QAT unless we intentionally start a new branch for that.

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
- steps completed: `1049`
- effective step time: about `1705 ms/step`

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
- steps completed: `978`
- effective step time: about `1830 ms/step`

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

Artifact source:
- Hugging Face dataset: `Mikeapedia/fineweb10B-sp8192-casefold-v2`

Local setup details:
- tokenizer model copied locally as `local_tokenizer_data/casefold_v2/tokenizers/fineweb_8192_bpe.model`
- original HF filename: `fineweb_8192_bpe_casefold_refined_v2.model`
- local train shards used tonight: `fineweb_train_000000.bin` through `fineweb_train_000009.bin`
- local validation shard: `fineweb_val_000000.bin`
- no validation byte sidecar was used or required

Execution details:
- same winning architecture as the SP8192 skip-gated baseline
- still `TTT_ENABLED=0`
- still `NUM_LOOPS=0`
- still no parallel residuals
- `SLIDING_WINDOW_ENABLED=0` to keep the evaluation path simple and directly comparable through the wrapper
- the sidecar wrapper script was still used, but on this dataset it falls back to the original token-byte LUT path because no `fineweb_val_bytes_*.bin` exists

Results:
- throughput: about `460-463k tok/s`
- prequant `val_bpb`: `1.19837140`
- quantized `val_bpb`: `1.20385604`
- peak VRAM: `27244 MiB`
- steps completed: `1047`
- effective step time: about `1708 ms/step`

Delta vs winning SP8192 baseline:
- quantized improvement: `-0.01948953 bpb`

Interpretation:
- Casefold V2 gives a clear, meaningful gain in the clean local no-TTT setting
- it preserves throughput almost exactly

Additional interpretation:
- this is important because it means the gain is not coming from a slower or heavier local training regime
- in this setup, Casefold V2 looks like a clean tokenizer improvement, not a throughput tradeoff disguised as a win

### CaseOps

Run:
- `runs/2026-04-22_CaseOps_NoTTT_SkipGates30m_v2/`

Tokenizer/data root:
- `local_tokenizer_data/caseops_v1/` (ignored locally; not committed)

Artifact source:
- Hugging Face dataset: `romeerp/parameter-golf-caseops-v1`

Local setup details:
- tokenizer model copied locally as `local_tokenizer_data/caseops_v1/tokenizers/fineweb_8192_bpe.model`
- original HF filename: `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- local train shards used tonight: `fineweb_train_000000.bin` through `fineweb_train_000009.bin`
- local validation shards:
  - `fineweb_val_000000.bin`
  - `fineweb_val_bytes_000000.bin`

Wrapper details:
- CaseOps could not be run safely on the unmodified source script because the stock `fineweb_val_*.bin` glob would also pick up `fineweb_val_bytes_*.bin`
- `scripts/train_gpt_decode_sidecar.py` fixes this by:
  - loading only token validation shards into `val_tokens`
  - loading `fineweb_val_bytes_*.bin` separately as the byte denominator sidecar
  - overriding `eval_val` so `val_bpb` is computed against original bytes instead of tokenizer-piece byte heuristics
- `SLIDING_WINDOW_ENABLED=0` was kept off because only the standard validation path was patched for byte sidecar accounting in this local wrapper

Execution details:
- same winning architecture as the baseline and Casefold runs
- still `TTT_ENABLED=0`
- still `NUM_LOOPS=0`
- still no parallel residuals

Results:
- throughput: about `454-456k tok/s`
- prequant `val_bpb`: `1.21895652`
- quantized `val_bpb`: `1.22397362`
- peak VRAM: `27244 MiB`
- steps completed: `1038`
- effective step time: about `1724 ms/step`

Delta vs winning SP8192 baseline:
- quantized delta: `+0.00062805 bpb`

Interpretation:
- CaseOps is effectively flat to slightly worse than the plain SP8192 baseline here
- in this local no-TTT setup, CaseOps does not justify itself

Additional interpretation:
- the CaseOps run is still useful because it confirms the sidecar-aware local path works correctly on 1x5090
- raw train loss is much lower than the SP8192 baseline, but that should not be over-interpreted across tokenizer families
- the metric that matters here is `val_bpb` after correct original-byte accounting, and on that metric CaseOps did not beat the baseline in this regime

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
