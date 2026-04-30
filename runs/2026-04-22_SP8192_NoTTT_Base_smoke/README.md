# 2026-04-22_SP8192_NoTTT_Base_smoke

## Source

- source script: `parameter-golf/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt_decode.py`

## Result

- status: `timeout / interrupted during local smoke`
- purpose: verify that the no-loop SP8192 no-TTT base reaches real training on 1x5090
- outcome: successful smoke; real training steps were reached promptly, which justified the later 30-minute baseline run

## Files

- `train.log`
- `command.sh`
- `train_gpt_decode.py`
