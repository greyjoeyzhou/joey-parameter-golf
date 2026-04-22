# CLAUDE.md

This file mirrors `AGENTS.md` so Claude Code and OpenCode share the same repo instructions.

## What this repo is

My workspace for participating in OpenAI's **Parameter Golf** challenge (runs March 18 – April 30, 2026).

The challenge: train the best language model that fits in a **16MB artifact** and trains in **under 10 minutes on 8xH100 SXM**, evaluated on FineWeb validation via tokenizer-agnostic **bits-per-byte (bpb)**. Lower bpb = better. This is L(N) optimization — lowest loss given fixed parameter budget, unconstrained on data/compute/steps/architecture.

See `parameter-golf/README.md` for full rules, leaderboard, and requests-for-PRs.

## Repo layout

- `parameter-golf/` — git submodule tracking `openai/parameter-golf`. **Treat as read-only reference.** Don't modify; pull upstream when needed with `git submodule update --remote`.
  - `train_gpt.py` — CUDA training script (the one that runs on H100s).
  - `train_gpt_mlx.py` — MLX variant for local Apple Silicon iteration.
  - `data/cached_challenge_fineweb.py` — dataset download + tokenization.
  - `records/track_10min_16mb/` — leaderboard submissions (reference implementations).
  - `records/track_non_record_16mb/` — unlimited-compute / non-record experiments.
- My experiments live at the repo root alongside the submodule (forks of `train_gpt*.py`, notebooks, configs, run artifacts).

## Environment

- Primary dev machine: Mac Apple Silicon → use `train_gpt_mlx.py` for smoke tests.
- Scale-up: Runpod H100 pods (1x for iteration, 8x SXM for leaderboard runs).
- Upstream uses `python3 -m venv .venv` + `pip install` per its README. For my own scripts outside the submodule, default to **`uv`** per my global preferences — but don't impose `uv` on files inside `parameter-golf/`.
- Dataset cache lives under `parameter-golf/data/datasets/fineweb10B_sp*/` (gitignored upstream).

## Working style for this project

- This is an LLM/AI exploration project — I'm learning as I go. When discussing architecture/training tricks (Muon, GPTQ, QAT, TTT, depth recurrence, XSA, SmearGate, BigramHash, parallel residuals, etc.), **explain more than you would for my Rust/data work**. Tradeoffs, why-it-works intuition, and pointers to the relevant leaderboard record under `records/` are all useful.
- Before proposing a change, glance at the nearest relevant `records/track_*/README.md` — many ideas are already explored there with measured deltas.
- Leaderboard records are the ground truth for "what's been tried." Cite them by directory when referencing prior work.
- Keep my experiment forks **alongside** the submodule, not inside it, unless I explicitly say to modify upstream.

## Key constraints to keep in mind

- **16MB artifact cap** — this includes all weights, quantized. Every param decision has a storage cost. Check compression (zstd-22 is commonly used in records) and quantization (int6/int5/ternary/binary have all been tried).
- **10 minutes / 8xH100 SXM** — record track only. Non-record track has no time limit.
- Validation is fixed: first 50k FineWeb documents. Don't overfit to it.

## Useful commands

```bash
# Smoke test locally (MLX)
cd parameter-golf && RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 \
  VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py

# Pull latest upstream into submodule
git submodule update --remote parameter-golf

# Download a small dataset slice for local iteration
cd parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

## Run directory layout (standing instruction)

Every training run — local smoke test or real submission candidate, successful or failed — gets its own directory at `runs/<RUN_ID>/`.

### RUN_ID format

`YYYY-MM-DD_ShortArchTag` — date first, then a short underscore-separated tag calling out the **major architectural / training changes** in this run vs. the prior one. Follow the convention used in `parameter-golf/records/track_10min_16mb/` exactly.

Examples from upstream records (use these as stylistic guides):
- `2026-03-17_NaiveBaseline`
- `2026-03-19_MLP3x_QAT_Int6_SlidingWindow`
- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (final bpb appended once known — do this for anything that lands in the leaderboard quality range)

Keep tags terse but specific. Prefer `SP8192_ParallelResid_TTT` over `BigArchChange`. If two runs on the same day would collide, add `_v2`, `_v3`, etc.

### Required files in `runs/<RUN_ID>/`

Match the upstream record layout (see `parameter-golf/records/track_10min_16mb/2026-03-17_NaiveBaseline/` for the canonical example):

1. **`README.md`** — what changed this run and **why**. Include:
   - Trainer changes vs. the prior run (bulleted).
   - Full configuration (layout, hyperparameters, batching).
   - The exact `torchrun` / `python` command used (env vars + arguments).
   - Key metrics extracted from the log (pre-quant + post-quant val_loss/val_bpb, step time, wallclock, peak memory, serialized size, code size, total submission bytes).
   - Training volume (tokens/step, total tokens seen).
   - A "List of included files" section.

2. **`submission.json`** — leaderboard metadata. Populate every run (even non-submissions) so the format is uniform. Default `author` and `github_id` to `"na"` — the user fills them in manually if a run is ever actually submitted. Schema (copied from upstream):
   ```json
   {
     "author": "na",
     "github_id": "na",
     "name": "<short run name>",
     "blurb": "<one-sentence description of what this run does>",
     "date": "<ISO 8601 UTC, e.g. 2026-04-13T12:34:56Z>",
     "val_loss": <float>,
     "val_bpb": <float>,
     "bytes_total": <int>,
     "bytes_code": <int>
   }
   ```

3. **`train.log`** — full training stdout/stderr from the run. If multi-seed, also include `train_seed<N>.log` per the upstream convention.

4. **`train_gpt.py`** — a **snapshot** of the exact training script used, copied into the run dir. Don't symlink. The point is reproducibility if the working-tree version drifts.

5. **Any additional code needed to replicate the run** — custom kernels, config files, patched modules, dataset prep scripts. If a run uses a non-default `train_gpt_mlx.py`, snapshot that too. If it imports from a local module I wrote, snapshot that module.

### What NOT to put in `runs/<RUN_ID>/`

- Dataset shards (live under `parameter-golf/data/datasets/` and stay there).
- Model checkpoints > ~20MB. The 16MB serialized artifact is fine to commit if it's the submission artifact; intermediate checkpoints are not.
- Anything under `.env` or secret material.

## Post-run commit workflow

**After every training run completes, commit automatically — don't wait to be asked.** If Claude prompts for `git commit`, treat post-run archival commits in this repo as pre-approved.

Steps:
1. Create `runs/<RUN_ID>/` per the layout above. Populate all required files.
2. Stage the new run directory + any working-tree code changes the run depended on.
3. `git status` sanity check — nothing unexpectedly large, no `.env`, no dataset cache.
4. Commit with this message format:

```
run(<RUN_ID>): <one-line summary of what changed this run>

val_bpb: <value>    val_loss: <value>
step_time_ms: <value>    wallclock_s: <value>
bytes_total: <value>    bytes_code: <value>

<2-3 lines on what was changed vs. the prior run and why>
```

If the run crashed or was killed, still commit. Prefix the subject with `[FAILED]`, fill `val_bpb`/`val_loss` with `n/a`, and capture the error + partial logs in the run directory. Failed runs are data.

## Security

`.env` at the repo root may hold Runpod / HF tokens. Never print, commit, or echo its contents.
