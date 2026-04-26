#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE_TRAIN_GPT="$ROOT/parameter-golf/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt_decode.py"
SNAPSHOT_PATH="parameter-golf/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt_decode.py"
BATCH_DIR="$ROOT/runs/2026-04-25_QATStructureBatch"
STATUS_FILE="$BATCH_DIR/status.tsv"
SUMMARY_FILE="$ROOT/notes/local-qat-structure-results-2026-04-25.md"

RUN_IDS=(
  "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All"
  "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_L7plus"
  "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_L8plus"
  "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_AttnOnly"
  "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_Attn_L7plus"
  "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_Attn_L8plus"
  "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLPOnly"
  "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLP_L7plus"
  "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLP_L8plus"
  "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_EarlyOnly"
)

mkdir -p "$BATCH_DIR"
printf 'run_id\tstatus\tstarted_at\tfinished_at\n' > "$STATUS_FILE"

BASE_ENVS=(
  "SOURCE_TRAIN_GPT=$SOURCE_TRAIN_GPT"
  "VOCAB_SIZE=8192"
  "DATA_DIR=../local_tokenizer_data/casefold_v2"
  "MAX_WALLCLOCK_SECONDS=1800"
  "WARMUP_STEPS=1"
  "TRAIN_LOG_EVERY=50"
  "VAL_LOSS_EVERY=0"
  "SLIDING_WINDOW_ENABLED=0"
  "TTT_ENABLED=0"
  "NUM_LOOPS=0"
  "PARALLEL_RESIDUAL_START=7"
  "HESSIAN_CLIP_LAMBDA=0.15"
  "SKIP_GATES_ENABLED=1"
  "QK_GAIN_INIT=4.0"
  "GPTQ_CALIBRATION_BATCHES=1"
  "QAT_ENABLED=1"
  "QAT_FRACTION=0.20"
  "QAT_RAMP_STEPS=500"
)

update_summary() {
  python3 "scripts/summarize_qat_structure_batch.py" --status-file "$STATUS_FILE" --output "$SUMMARY_FILE" "${RUN_IDS[@]}"
}

run_case() {
  local run_id="$1"
  shift
  local started_at finished_at rc
  started_at="$(date -Iseconds)"
  echo "[$started_at] START $run_id"

  local cmd=(
    python3 "scripts/run_local_experiment.py"
    --run-id "$run_id"
    --source-script "scripts/train_gpt_decode_qat.py"
    --snapshot "$SNAPSHOT_PATH"
    --shim-flash-attn
    --copy-logfile
    --timeout-seconds 2100
  )

  local kv
  for kv in "${BASE_ENVS[@]}"; do
    cmd+=(--env "$kv")
  done
  for kv in "$@"; do
    cmd+=(--env "$kv")
  done

  "${cmd[@]}"
  rc=$?
  finished_at="$(date -Iseconds)"
  if [[ $rc -eq 0 ]]; then
    printf '%s\t%s\t%s\t%s\n' "$run_id" "ok" "$started_at" "$finished_at" >> "$STATUS_FILE"
  else
    printf '%s\t%s\t%s\t%s\n' "$run_id" "rc=$rc" "$started_at" "$finished_at" >> "$STATUS_FILE"
  fi
  update_summary
  echo "[$finished_at] END $run_id rc=$rc"
  return 0
}

update_summary

run_case "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All" "QAT_TARGET=all" "QAT_LAYER_START=0" "QAT_LAYER_END=10"
run_case "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_L7plus" "QAT_TARGET=all" "QAT_LAYER_START=7" "QAT_LAYER_END=10"
run_case "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_L8plus" "QAT_TARGET=all" "QAT_LAYER_START=8" "QAT_LAYER_END=10"
run_case "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_AttnOnly" "QAT_TARGET=attn" "QAT_LAYER_START=0" "QAT_LAYER_END=10"
run_case "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_Attn_L7plus" "QAT_TARGET=attn" "QAT_LAYER_START=7" "QAT_LAYER_END=10"
run_case "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_Attn_L8plus" "QAT_TARGET=attn" "QAT_LAYER_START=8" "QAT_LAYER_END=10"
run_case "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLPOnly" "QAT_TARGET=mlp" "QAT_LAYER_START=0" "QAT_LAYER_END=10"
run_case "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLP_L7plus" "QAT_TARGET=mlp" "QAT_LAYER_START=7" "QAT_LAYER_END=10"
run_case "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_MLP_L8plus" "QAT_TARGET=mlp" "QAT_LAYER_START=8" "QAT_LAYER_END=10"
run_case "2026-04-25_CasefoldV2_ParResid_QAT20_QKGain40_All_EarlyOnly" "QAT_TARGET=all" "QAT_LAYER_START=0" "QAT_LAYER_END=6"

update_summary
echo "[$(date -Iseconds)] QAT structure batch complete"
