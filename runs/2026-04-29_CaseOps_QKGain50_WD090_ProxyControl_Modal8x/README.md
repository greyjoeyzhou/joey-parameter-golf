# 2026-04-29_CaseOps_QKGain50_WD090_ProxyControl_Modal8x

## What Changed

- First competition-shaped Modal run for the best known non-TTT CaseOps proxy-control setting.
- Uses the archived `2026-04-28_CaseOps_QKGain50_WD090_ProxyControl_Valid2p5h` trainer snapshot through `tmp_modal/modal_runner.py`.
- Runs the full train -> EMA -> GPTQ -> quantized eval pipeline on `8xH100` with a `600s` training wallclock cap.

## Configuration

- GPUs: `8xH100`
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
- `TRAIN_LOG_EVERY=50`

## Command

```bash
uvx --from modal modal run tmp_modal/modal_runner.py::main \
  --wallclock 600 \
  --run-id 2026-04-29_CaseOps_QKGain50_WD090_ProxyControl_Modal8x
```

## Metrics

- training stop: wallclock cap at `587.993s`
- steps completed: `4133`
- training step time: `~142.3 ms/step`
- in-run val: `val_loss 2.8392`, `val_bpb 1.0968`
- pre-quant post-EMA: `val_loss 2.83647761`, `val_bpb 1.09577747`
- quantized: `val_loss 2.86931345`, `val_bpb 1.10846249`
- peak memory: `25730 MiB allocated`, `25822 MiB reserved`
- quantized artifact bytes: `15977139`
- total submission bytes: `16057194`

## Training Volume

- tokens/step: `786432`
- total tokens seen: `3250323456`

## Notes

- This matches the competition-style `8xH100` plus `600s` training setup, but the overall Modal job runtime is longer because the trainer still does post-cap EMA, serialization, GPTQ, and quantized eval work after the training clock stops.
- The downloaded `final_model.pt` was intentionally left out of git because it is much larger than the repo's run-artifact size guidance.

## Included Files

- `README.md`
- `submission.json`
- `command.sh`
- `final_model.int6.ptz`
- `logs/2026-04-29_CaseOps_QKGain50_WD090_ProxyControl_Modal8x.txt`
- `train_gpt_decode.py`
- `train_gpt_decode_sidecar.py`
