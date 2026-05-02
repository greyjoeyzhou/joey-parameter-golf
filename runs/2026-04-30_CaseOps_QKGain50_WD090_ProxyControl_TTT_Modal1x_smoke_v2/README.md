# 2026-04-30_CaseOps_QKGain50_WD090_ProxyControl_TTT_Modal1x_smoke_v2

## What Changed

- Second Modal smoke attempt for the conservative TTT variant on the current best non-TTT CaseOps proxy-control stack.
- Included the sidecar fallback fix so TTT works on the standard `sp8192` validation layout without byte-sidecar files.
- Kept the same cautious TTT configuration: last-three-block adaptation, one epoch, reduced LR.

## Configuration

- GPUs: `1xH100`
- `VOCAB_SIZE=8192`
- `MAX_WALLCLOCK_SECONDS=600.0`
- `ITERATIONS=20000`
- `NUM_LOOPS=0`
- `PARALLEL_RESIDUAL_START=7`
- `QK_GAIN_INIT=5.0`
- `HESSIAN_CLIP_LAMBDA=0.15`
- `MUON_WD=0.090`
- `EMBED_WD=0.090`
- `SKIP_GATES_ENABLED=1`
- `SLIDING_WINDOW_ENABLED=0`
- `TTT_ENABLED=1`
- `TTT_LR=0.001`
- `TTT_EPOCHS=1`
- `TTT_FREEZE_BLOCKS=8`
- `TTT_CHUNK_TOKENS=32768`
- `TTT_NS_STEPS=0`
- `VAL_TOKEN_LIMIT=196608`
- `TRAIN_LOG_EVERY=25`

## Command

```bash
uvx --from modal modal run tmp_modal/modal_runner.py::smoke_test_ttt \
  --wallclock 600 \
  --run-id 2026-04-30_CaseOps_QKGain50_WD090_ProxyControl_TTT_Modal1x_smoke_v2
```

## Metrics

- training stop: wallclock cap at `588.736s`
- steps completed: `752`
- training step time: `~782.6 ms/step`
- in-run val: `val_loss 3.0823`, `val_bpb 1.1907`
- pre-quant post-EMA: `val_loss 3.38020420`, `val_bpb 1.30582790`
- quantized: `val_loss 3.39410758`, `val_bpb 1.31119901`
- quantized TTT: `val_loss 3.91892557`, `val_bpb 1.51394474`
- peak memory: `25958 MiB allocated`, `26032 MiB reserved`
- quantized artifact bytes: `15989507`
- total submission bytes: `16069562`

## Outcome

- The TTT path completed successfully after the fallback fix, but materially worsened validation quality.
- This suggests the current conservative TTT settings are mechanically valid on Modal but not promising enough to justify an immediate 8xH100 run.

## Training Volume

- tokens/step: `786432`
- total tokens seen: `591396864`

## Included Files

- `README.md`
- `submission.json`
- `command.sh`
- `final_model.int6.ptz`
- `logs/2026-04-30_CaseOps_QKGain50_WD090_ProxyControl_TTT_Modal1x_smoke_v2.txt`
- `train_gpt_decode.py`
- `train_gpt_decode_sidecar.py`
