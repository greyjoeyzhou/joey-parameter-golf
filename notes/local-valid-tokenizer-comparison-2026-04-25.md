# Local Valid Tokenizer Comparison

This note is updated automatically while the valid-tokenizer batch is running.

Last updated: `2026-04-26 12:47:12Z`

## Batch Status

- planned runs: `12`
- completed ok: `12`
- running: `0`
- pending: `0`
- failed/nonzero: `0`

## Planned Runs

- `2026-04-25_SP8192_Base_Valid`: status `ok`, tokenizer `SP8192`, stack `Base`
- `2026-04-25_CaseOps_Base_Valid`: status `ok`, tokenizer `CaseOps`, stack `Base`
- `2026-04-25_SP8192_SkipGates_Valid`: status `ok`, tokenizer `SP8192`, stack `SkipGates`
- `2026-04-25_CaseOps_SkipGates_Valid`: status `ok`, tokenizer `CaseOps`, stack `SkipGates`
- `2026-04-25_SP8192_SkipGates_ParResid_Valid`: status `ok`, tokenizer `SP8192`, stack `SkipGates_ParResid`
- `2026-04-25_CaseOps_SkipGates_ParResid_Valid`: status `ok`, tokenizer `CaseOps`, stack `SkipGates_ParResid`
- `2026-04-25_SP8192_SkipGates_ParResid_QKGain45_HessClip015_Valid`: status `ok`, tokenizer `SP8192`, stack `SkipGates_ParResid_QKGain45_HessClip015`
- `2026-04-25_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid`: status `ok`, tokenizer `CaseOps`, stack `SkipGates_ParResid_QKGain45_HessClip015`
- `2026-04-25_SP8192_SkipGates_ParResid_QAT10_QKGain40_Valid`: status `ok`, tokenizer `SP8192`, stack `SkipGates_ParResid_QAT10_QKGain40`
- `2026-04-25_CaseOps_SkipGates_ParResid_QAT10_QKGain40_Valid`: status `ok`, tokenizer `CaseOps`, stack `SkipGates_ParResid_QAT10_QKGain40`
- `2026-04-25_SP8192_SkipGates_ParResid_QAT20_QKGain40_MLPOnly_Valid`: status `ok`, tokenizer `SP8192`, stack `SkipGates_ParResid_QAT20_QKGain40_MLPOnly`
- `2026-04-25_CaseOps_SkipGates_ParResid_QAT20_QKGain40_MLPOnly_Valid`: status `ok`, tokenizer `CaseOps`, stack `SkipGates_ParResid_QAT20_QKGain40_MLPOnly`

## Ranking

| rank | run | tokenizer | stack | prequant_bpb | quantized_bpb | gap | steps |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `2026-04-25_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid` | `CaseOps` | `SkipGates_ParResid_QKGain45_HessClip015` | 1.21457238 | 1.21949691 | 0.00492453 | 1063 |
| 2 | `2026-04-25_SP8192_SkipGates_ParResid_Valid` | `SP8192` | `SkipGates_ParResid` | 1.21411623 | 1.21983153 | 0.00571530 | 1065 |
| 3 | `2026-04-25_CaseOps_SkipGates_ParResid_QAT10_QKGain40_Valid` | `CaseOps` | `SkipGates_ParResid_QAT10_QKGain40` | 1.21484892 | 1.21987584 | 0.00502692 | 1058 |
| 4 | `2026-04-25_CaseOps_SkipGates_ParResid_QAT20_QKGain40_MLPOnly_Valid` | `CaseOps` | `SkipGates_ParResid_QAT20_QKGain40_MLPOnly` | 1.21478246 | 1.21999858 | 0.00521612 | 1060 |
| 5 | `2026-04-25_CaseOps_SkipGates_ParResid_Valid` | `CaseOps` | `SkipGates_ParResid` | 1.21535263 | 1.22020667 | 0.00485404 | 1058 |
| 6 | `2026-04-25_SP8192_SkipGates_ParResid_QAT20_QKGain40_MLPOnly_Valid` | `SP8192` | `SkipGates_ParResid_QAT20_QKGain40_MLPOnly` | 1.21484059 | 1.22053427 | 0.00569368 | 1062 |
| 7 | `2026-04-25_SP8192_SkipGates_ParResid_QKGain45_HessClip015_Valid` | `SP8192` | `SkipGates_ParResid_QKGain45_HessClip015` | 1.21539570 | 1.22104359 | 0.00564789 | 1058 |
| 8 | `2026-04-25_CaseOps_SkipGates_Valid` | `CaseOps` | `SkipGates` | 1.21612564 | 1.22121163 | 0.00508599 | 1051 |
| 9 | `2026-04-25_SP8192_SkipGates_ParResid_QAT10_QKGain40_Valid` | `SP8192` | `SkipGates_ParResid_QAT10_QKGain40` | 1.21592187 | 1.22144442 | 0.00552255 | 1055 |
| 10 | `2026-04-25_SP8192_SkipGates_Valid` | `SP8192` | `SkipGates` | 1.21701173 | 1.22267872 | 0.00566699 | 1050 |
| 11 | `2026-04-25_CaseOps_Base_Valid` | `CaseOps` | `Base` | 1.23649766 | 1.24166658 | 0.00516892 | 1057 |
| 12 | `2026-04-25_SP8192_Base_Valid` | `SP8192` | `Base` | 1.23818867 | 1.24352652 | 0.00533785 | 1049 |

## Tokenizer Deltas By Stack

| stack | SP8192 | CaseOps | delta (CaseOps - SP8192) |
| --- | ---: | ---: | ---: |
| `Base` | 1.24352652 | 1.24166658 | -0.00185994 |
| `SkipGates` | 1.22267872 | 1.22121163 | -0.00146709 |
| `SkipGates_ParResid` | 1.21983153 | 1.22020667 | 0.00037514 |
| `SkipGates_ParResid_QKGain45_HessClip015` | 1.22104359 | 1.21949691 | -0.00154668 |
| `SkipGates_ParResid_QAT10_QKGain40` | 1.22144442 | 1.21987584 | -0.00156858 |
| `SkipGates_ParResid_QAT20_QKGain40_MLPOnly` | 1.22053427 | 1.21999858 | -0.00053569 |

## Current Takeaway

- current batch leader: `2026-04-25_CaseOps_SkipGates_ParResid_QKGain45_HessClip015_Valid`
- tokenizer: `CaseOps`
- stack: `SkipGates_ParResid_QKGain45_HessClip015`
- quantized `val_bpb`: `1.21949691`
- prequant `val_bpb`: `1.21457238`
- quantization gap: `0.00492453`

