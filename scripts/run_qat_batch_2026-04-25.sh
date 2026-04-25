#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE_TRAIN_GPT="$ROOT/parameter-golf/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt_decode.py"
SNAPSHOT_PATH="parameter-golf/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt_decode.py"
BATCH_DIR="$ROOT/runs/2026-04-25_QATBatch"
STATUS_FILE="$BATCH_DIR/status.tsv"

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
  "QK_GAIN_INIT=4.5"
  "GPTQ_CALIBRATION_BATCHES=1"
  "QAT_ENABLED=1"
  "QAT_FRACTION=0.10"
  "QAT_RAMP_STEPS=500"
)

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
  echo "[$finished_at] END $run_id rc=$rc"
  return 0
}

run_case "2026-04-25_CasefoldV2_ParResid_QAT10"
run_case "2026-04-25_CasefoldV2_ParResid_QAT20" "QAT_FRACTION=0.20"
run_case "2026-04-25_CasefoldV2_ParResid_QAT10_NoHessClip" "HESSIAN_CLIP_LAMBDA=0.0"
run_case "2026-04-25_CasefoldV2_ParResid_QAT05" "QAT_FRACTION=0.05"
run_case "2026-04-25_CasefoldV2_ParResid_QAT15" "QAT_FRACTION=0.15"
run_case "2026-04-25_CasefoldV2_ParResid_QAT10_Ramp250" "QAT_RAMP_STEPS=250"
run_case "2026-04-25_CasefoldV2_ParResid_QAT10_Ramp1000" "QAT_RAMP_STEPS=1000"
run_case "2026-04-25_CasefoldV2_ParResid_QAT10_QKGain40" "QK_GAIN_INIT=4.0"
run_case "2026-04-25_CasefoldV2_ParResid_QAT10_QKGain50" "QK_GAIN_INIT=5.0"
run_case "2026-04-25_CasefoldV2_ParResid_QAT20_NoHessClip" "QAT_FRACTION=0.20" "HESSIAN_CLIP_LAMBDA=0.0"

echo "[$(date -Iseconds)] QAT batch complete"
