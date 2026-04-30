# 2026-04-29_CaseOps_QKGain50_WD090_ProxyControl_Modal1x_smoke

## What Changed

- First end-to-end Modal smoke test for the best known non-TTT CaseOps proxy-control setting.
- Uses the archived `2026-04-28_CaseOps_QKGain50_WD090_ProxyControl_Valid2p5h` trainer snapshot through `tmp_modal/modal_runner.py`.
- Runs the full train -> EMA -> GPTQ -> quantized eval pipeline on `1xH100` with a `600s` training wallclock cap to validate the Modal setup.

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
- `TTT_ENABLED=0`
- `VAL_TOKEN_LIMIT=196608`
- `TRAIN_LOG_EVERY=25`

## Command

```bash
uvx --from modal modal run tmp_modal/modal_runner.py::smoke_test \
  --wallclock 600 \
  --run-id 2026-04-29_CaseOps_QKGain50_WD090_ProxyControl_Modal1x_smoke
```

## Metrics

- training stop: wallclock cap at `588.211s`
- steps completed: `766`
- training step time: `~767.9 ms/step`
- in-run val: `val_loss 3.0791`, `val_bpb 1.1895`
- pre-quant post-EMA: `val_loss 3.36313876`, `val_bpb 1.29923524`
- quantized: `val_loss 3.37718042`, `val_bpb 1.30465977`
- peak memory: `25958 MiB allocated`, `26032 MiB reserved`
- quantized artifact bytes: `15989989`
- total submission bytes: `16070044`

## Training Volume

- tokens/step: `786432`
- total tokens seen: `602406912`

## Notes

- This run exists to validate the Modal pipeline, not to compare fairly against the full local 2.5h control run.
- The downloaded `final_model.pt` was intentionally left out of git because it is much larger than the repo's run-artifact size guidance.

## Included Files

- `README.md`
- `submission.json`
- `command.sh`
- `final_model.int6.ptz`
- `logs/2026-04-29_CaseOps_QKGain50_WD090_ProxyControl_Modal1x_smoke.txt`
- `train_gpt_decode.py`
- `train_gpt_decode_sidecar.py`
