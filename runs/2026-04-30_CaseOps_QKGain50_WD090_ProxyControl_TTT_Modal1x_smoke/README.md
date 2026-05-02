# 2026-04-30_CaseOps_QKGain50_WD090_ProxyControl_TTT_Modal1x_smoke

## What Changed

- First Modal smoke attempt for a conservative TTT variant layered on the current best non-TTT CaseOps proxy-control stack.
- Reused the archived `2026-04-28_CaseOps_QKGain50_WD090_ProxyControl_Valid2p5h` source trainer through the live sidecar wrapper.
- Enabled TTT with a cautious configuration: last-three-block adaptation, fixed one-epoch updates, and reduced TTT LR.

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
  --run-id 2026-04-30_CaseOps_QKGain50_WD090_ProxyControl_TTT_Modal1x_smoke
```

## Metrics

- training stop: wallclock cap at `588.544s`
- steps completed: `785`
- training step time: `~749.7 ms/step`
- in-run val: `val_loss 3.0715`, `val_bpb 1.1866`
- pre-quant post-EMA: `val_loss 3.33773661`, `val_bpb 1.28942198`
- quantized: `val_loss 3.35178049`, `val_bpb 1.29484736`
- peak memory: `25956 MiB allocated`, `26052 MiB reserved`
- quantized artifact bytes: `15988276`
- total submission bytes: `16068331`

## Outcome

- This run failed during the TTT eval phase after the quantized baseline eval completed.
- Failure mode: the sidecar fell back to the upstream `eval_val_ttt` path on a non-sidecar validation layout and hit the `inference_mode()` / autograd rotary cache bug.

## Training Volume

- tokens/step: `786432`
- total tokens seen: `617086720`

## Included Files

- `README.md`
- `submission.json`
- `command.sh`
- `final_model.int6.ptz`
- `logs/2026-04-30_CaseOps_QKGain50_WD090_ProxyControl_TTT_Modal1x_smoke.txt`
- `train_gpt_decode.py`
- `train_gpt_decode_sidecar.py`
