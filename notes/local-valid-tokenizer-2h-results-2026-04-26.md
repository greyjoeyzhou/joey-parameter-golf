# Local Valid Tokenizer Comparison

This note is updated automatically while the valid-tokenizer batch is running.

Last updated: `2026-04-26 23:46:53Z`

## Batch Status

- planned runs: `3`
- completed ok: `3`
- running: `0`
- pending: `0`
- failed/nonzero: `0`

## Planned Runs

- `2026-04-26_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid2h`: status `ok`, tokenizer `CaseOps`, stack `SkipGates_ParResid_QKGain45_HessClip015_2h`
- `2026-04-26_SP8192_SkipGates_ParResid_Valid2h`: status `ok`, tokenizer `SP8192`, stack `SkipGates_ParResid_2h`
- `2026-04-26_CaseOps_SkipGates_ParResid_QAT10_QKGain40_Valid2h`: status `ok`, tokenizer `CaseOps`, stack `SkipGates_ParResid_QAT10_QKGain40_2h`

## Ranking

| rank | run | tokenizer | stack | prequant_bpb | quantized_bpb | gap | steps |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `2026-04-26_CaseOps_SkipGates_ParResid_QAT10_QKGain40_Valid2h` | `CaseOps` | `SkipGates_ParResid_QAT10_QKGain40_2h` | 1.10153131 | 1.11284657 | 0.01131526 | 4263 |
| 2 | `2026-04-26_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid2h` | `CaseOps` | `SkipGates_ParResid_QKGain45_HessClip015_2h` | 1.10173679 | 1.11297258 | 0.01123579 | 4262 |
| 3 | `2026-04-26_SP8192_SkipGates_ParResid_Valid2h` | `SP8192` | `SkipGates_ParResid_2h` | 1.10588898 | 1.11805923 | 0.01217025 | 4254 |

## Tokenizer Deltas By Stack

| stack | SP8192 | CaseOps | delta (CaseOps - SP8192) |
| --- | ---: | ---: | ---: |
| `SkipGates_ParResid_QKGain45_HessClip015_2h` | n/a | 1.11297258 | n/a |
| `SkipGates_ParResid_2h` | 1.11805923 | n/a | n/a |
| `SkipGates_ParResid_QAT10_QKGain40_2h` | n/a | 1.11284657 | n/a |

## Current Takeaway

- current batch leader: `2026-04-26_CaseOps_SkipGates_ParResid_QAT10_QKGain40_Valid2h`
- tokenizer: `CaseOps`
- stack: `SkipGates_ParResid_QAT10_QKGain40_2h`
- quantized `val_bpb`: `1.11284657`
- prequant `val_bpb`: `1.10153131`
- quantization gap: `0.01131526`

