# 2026-04-22_SP8192_NoTTT_Loop45x2_smoke

## Source

- source script: `parameter-golf/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt_decode.py`

## Result

- status: `timeout / interrupted during local smoke`
- purpose: test whether the recurrence-enabled SP8192 no-TTT base is locally viable on 1x5090
- outcome: loop warmup was reached, but the configuration was too compile-expensive to use as the local baseline anchor

## Files

- `train.log`
- `command.sh`
- `train_gpt_decode.py`
