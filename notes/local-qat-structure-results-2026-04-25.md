# Local QAT Structure Results

This note is updated automatically while the QAT structure batch is running.

Last updated: `2026-04-25 23:53:17Z`

## Batch Status

- planned runs: `10`
- completed ok: `10`
- running: `0`
- pending: `0`
- failed/nonzero: `0`

## Reference

Current best pre-structure QAT reference from the earlier batch:
- `2026-04-25_CasefoldV2_ParResid_QAT10_QKGain40` prequant `val_bpb`: `1.19534324`
- `2026-04-25_CasefoldV2_ParResid_QAT10_QKGain40` quantized `val_bpb`: `1.20058967`

## Planned Runs

- `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All`: status `ok`, target `all`, layers `0-10`
- `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_L7plus`: status `ok`, target `all`, layers `7-10`
- `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_L8plus`: status `ok`, target `all`, layers `8-10`
- `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_AttnOnly`: status `ok`, target `attn`, layers `0-10`
- `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_Attn_L7plus`: status `ok`, target `attn`, layers `7-10`
- `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_Attn_L8plus`: status `ok`, target `attn`, layers `8-10`
- `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLPOnly`: status `ok`, target `mlp`, layers `0-10`
- `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLP_L7plus`: status `ok`, target `mlp`, layers `7-10`
- `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLP_L8plus`: status `ok`, target `mlp`, layers `8-10`
- `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_EarlyOnly`: status `ok`, target `all`, layers `0-6`

## Ranking

| rank | run | target | layers | prequant_bpb | quantized_bpb | gap |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLPOnly` | `mlp` | `0-10` | 1.19466903 | 1.20008068 | 0.00541165 |
| 2 | `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_L8plus` | `all` | `8-10` | 1.19489052 | 1.20025503 | 0.00536451 |
| 3 | `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_AttnOnly` | `attn` | `0-10` | 1.19542982 | 1.20061683 | 0.00518701 |
| 4 | `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_Attn_L8plus` | `attn` | `8-10` | 1.19524249 | 1.20068688 | 0.00544439 |
| 5 | `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_L7plus` | `all` | `7-10` | 1.19557752 | 1.20088150 | 0.00530398 |
| 6 | `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_EarlyOnly` | `all` | `0-6` | 1.19580090 | 1.20122299 | 0.00542209 |
| 7 | `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_Attn_L7plus` | `attn` | `7-10` | 1.19601500 | 1.20140407 | 0.00538907 |
| 8 | `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All` | `all` | `0-10` | 1.19651223 | 1.20178151 | 0.00526928 |
| 9 | `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLP_L7plus` | `mlp` | `7-10` | 1.19656158 | 1.20192033 | 0.00535875 |
| 10 | `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLP_L8plus` | `mlp` | `8-10` | 1.19672420 | 1.20204191 | 0.00531771 |

## Current Takeaway

- current batch leader: `2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLPOnly`
- quantized `val_bpb`: `1.20008068`
- prequant `val_bpb`: `1.19466903`
- quantization gap: `0.00541165`
- delta vs `QAT10_QKGain40`: `-0.00050899 bpb`

