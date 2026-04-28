# Local CaseOps Non-TTT Technique Batch

This note is updated automatically while the CaseOps non-TTT technique batch is running.

Last updated: `2026-04-27 09:27:25Z`

## Batch Status

- planned runs: `5`
- completed ok: `5`
- running: `0`
- pending: `0`
- failed/nonzero: `0`

## Reference

Current best 30-minute non-QAT CaseOps reference:
- `2026-04-25_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid` prequant `val_bpb`: `1.21457238`
- `2026-04-25_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid` quantized `val_bpb`: `1.21949691`

## Planned Runs

- `2026-04-26_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_WD090_Valid`: status `ok`, stack `WD090`
- `2026-04-26_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_WD095_Valid`: status `ok`, stack `WD095`
- `2026-04-26_CaseOps_SkipGates_ParResid_QKGain50_HessClip015_Valid`: status `ok`, stack `QKGain50`
- `2026-04-26_CaseOps_SkipGates_ParResid_QKGain525_HessClip015_Valid`: status `ok`, stack `QKGain525`
- `2026-04-26_CaseOps_SkipGates_ParResid_QKGain50_HessClip015_WD090_Valid`: status `ok`, stack `WD090_QKGain50`

## Ranking

| rank | run | stack | prequant_bpb | quantized_bpb | gap | steps |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `2026-04-26_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_WD095_Valid` | `WD095` | 1.21654617 | 1.22134413 | 0.00479796 | 1061 |
| 2 | `2026-04-26_CaseOps_SkipGates_ParResid_QKGain50_HessClip015_WD090_Valid` | `WD090_QKGain50` | 1.21659351 | 1.22142947 | 0.00483596 | 1061 |
| 3 | `2026-04-26_CaseOps_SkipGates_ParResid_QKGain50_HessClip015_Valid` | `QKGain50` | 1.21696300 | 1.22209933 | 0.00513633 | 1057 |
| 4 | `2026-04-26_CaseOps_SkipGates_ParResid_QKGain525_HessClip015_Valid` | `QKGain525` | 1.21775055 | 1.22268018 | 0.00492963 | 1054 |
| 5 | `2026-04-26_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_WD090_Valid` | `WD090` | 1.22054223 | 1.22540618 | 0.00486395 | 1037 |

## Current Takeaway

- current batch leader: `2026-04-26_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_WD095_Valid`
- stack: `WD095`
- quantized `val_bpb`: `1.22134413`
- prequant `val_bpb`: `1.21654617`
- quantization gap: `0.00479796`
- delta vs current 30m CaseOps reference: `+0.00184722 bpb`

