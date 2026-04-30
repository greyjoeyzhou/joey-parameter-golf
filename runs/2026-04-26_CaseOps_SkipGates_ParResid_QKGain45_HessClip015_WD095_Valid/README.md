# 2026-04-26_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_WD095_Valid

## Source

- source script: `/home/joey/Code/joey-parameter-golf/scripts/train_gpt_decode_sidecar.py`
- archived files: `train_gpt_decode_sidecar.py`, `train_gpt_decode.py`

## Environment

- `DATA_DIR=../local_tokenizer_data/caseops_v1`
- `EMBED_WD=0.095`
- `GPTQ_CALIBRATION_BATCHES=1`
- `HESSIAN_CLIP_LAMBDA=0.15`
- `MAX_WALLCLOCK_SECONDS=1800`
- `MUON_WD=0.095`
- `NUM_LOOPS=0`
- `PARALLEL_RESIDUAL_START=7`
- `PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:/home/joey/.opencode/bin:/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:/home/joey/.volta/bin:/home/joey/.opencode/bin:/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:/home/joey/.volta/bin:/home/joey/.opencode/bin:/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:/home/joey/.volta/bin:/home/joey/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin`
- `PYTHONPATH=/home/joey/Code/joey-parameter-golf/local_shims`
- `QK_GAIN_INIT=4.5`
- `RUN_ID=2026-04-26_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_WD095_Valid`
- `SKIP_GATES_ENABLED=1`
- `SLIDING_WINDOW_ENABLED=0`
- `SOURCE_TRAIN_GPT=/home/joey/Code/joey-parameter-golf/parameter-golf/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt_decode.py`
- `TRAIN_LOG_EVERY=50`
- `TTT_ENABLED=0`
- `VAL_LOSS_EVERY=0`
- `VOCAB_SIZE=8192`
- `WARMUP_STEPS=1`

## Result

- return code: `0`
- wallclock_s: `1954.8`
- final_val_bpb: `1.22134413`
- peak_mem_mib: `27053`
- prequant_val_bpb: `1.21654617`
- quantized_val_bpb: `1.22134413`

## Files

- `train.log`
- `command.sh`
- `submission.json`
- `train_gpt_decode_sidecar.py`
- `train_gpt_decode.py`
