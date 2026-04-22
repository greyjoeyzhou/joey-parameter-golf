# joey-parameter-golf

Workspace for OpenAI's Parameter Golf challenge experiments.

## Agent Setup

- `AGENTS.md` is the shared project instruction file for coding agents.
- `CLAUDE.md` mirrors the same repo context for Claude Code.
- `.claude/settings.json` contains project-level Claude permissions and is now workspace-relative.
- `.claude/settings.local.json` is machine-specific and can keep local one-off command approvals.

## Repo Layout

- `parameter-golf/` - upstream `openai/parameter-golf` submodule; treat as read-only unless explicitly asked otherwise.
- repo root - local experiment forks, configs, notebooks, and artifacts.
- `runs/` - archived run directories for every training attempt.
- `benchmarks_pilot/` - local benchmark notes and outputs.
- `instructions/` - task briefs and operating notes.

## Useful Commands

```bash
# Smoke test locally (MLX)
cd parameter-golf && RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 \
  VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py

# Pull latest upstream into submodule
git submodule update --remote parameter-golf

# Download a small dataset slice for local iteration
cd parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```
