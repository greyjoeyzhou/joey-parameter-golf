# 2026-04-22_SP8192_NoTTT_Base30m

## Source

- source script: `/home/joey/Code/joey-parameter-golf/parameter-golf/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt_decode.py`
- archived files: `train_gpt_decode.py`

## Environment

- `DATA_DIR=./data`
- `GPTQ_CALIBRATION_BATCHES=1`
- `HESSIAN_CLIP_LAMBDA=0.0`
- `MAX_WALLCLOCK_SECONDS=1800`
- `NUM_LOOPS=0`
- `PARALLEL_RESIDUAL_START=-1`
- `PATH=/usr/lib/wsl/lib:/home/joey/.opencode/bin:/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:/home/joey/.volta/bin:/home/joey/.opencode/bin:/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:/home/joey/.volta/bin:/home/joey/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin`
- `PYTHONPATH=/home/joey/Code/joey-parameter-golf/local_shims`
- `RUN_ID=2026-04-22_SP8192_NoTTT_Base30m`
- `SKIP_GATES_ENABLED=0`
- `TRAIN_LOG_EVERY=50`
- `TTT_ENABLED=0`
- `VAL_LOSS_EVERY=0`
- `VOCAB_SIZE=8192`
- `WARMUP_STEPS=1`

## Result

- return code: `124`
- wallclock_s: `2100.8`
- final_val_bpb: `1.24227277`
- peak_mem_mib: `26860`
- prequant_val_bpb: `1.23678117`
- quantized_val_bpb: `1.24227277`
- timeout_seconds: `2100.0`

## Files

- `train.log`
- `command.sh`
- `submission.json`
- `train_gpt_decode.py`
