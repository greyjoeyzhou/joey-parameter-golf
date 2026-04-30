# Local CaseOps Non-TTT Technique Batch

This note is updated automatically while the CaseOps non-TTT technique batch is running.

Last updated: `2026-04-28 02:36:43Z`

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

- `2026-04-27_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_WD090_Valid2h`: status `ok`, stack `WD090_2h`
- `2026-04-27_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_WD095_Valid2h`: status `ok`, stack `WD095_2h`
- `2026-04-27_CaseOps_SkipGates_ParResid_QKGain50_HessClip015_Valid2h`: status `ok`, stack `QKGain50_2h`
- `2026-04-27_CaseOps_SkipGates_ParResid_QKGain525_HessClip015_Valid2h`: status `ok`, stack `QKGain525_2h`
- `2026-04-27_CaseOps_SkipGates_ParResid_QKGain50_HessClip015_WD090_Valid2h`: status `ok`, stack `WD090_QKGain50_2h`

## Ranking

| rank | run | stack | prequant_bpb | quantized_bpb | gap | steps |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `2026-04-27_CaseOps_SkipGates_ParResid_QKGain50_HessClip015_WD090_Valid2h` | `WD090_QKGain50_2h` | 1.10164521 | 1.11258473 | 0.01093952 | 4262 |
| 2 | `2026-04-27_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_WD090_Valid2h` | `WD090_2h` | 1.10167358 | 1.11266635 | 0.01099277 | 4254 |
| 3 | `2026-04-27_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_WD095_Valid2h` | `WD095_2h` | 1.10209578 | 1.11285716 | 0.01076138 | 4268 |
| 4 | `2026-04-27_CaseOps_SkipGates_ParResid_QKGain50_HessClip015_Valid2h` | `QKGain50_2h` | 1.10163368 | 1.11293534 | 0.01130166 | 4267 |
| 5 | `2026-04-27_CaseOps_SkipGates_ParResid_QKGain525_HessClip015_Valid2h` | `QKGain525_2h` | 1.10223800 | 1.11337143 | 0.01113343 | 4253 |

## Current Takeaway

- current batch leader: `2026-04-27_CaseOps_SkipGates_ParResid_QKGain50_HessClip015_WD090_Valid2h`
- stack: `WD090_QKGain50_2h`
- quantized `val_bpb`: `1.11258473`
- prequant `val_bpb`: `1.10164521`
- quantization gap: `0.01093952`
- delta vs current 30m CaseOps reference: `-0.10691218 bpb`

