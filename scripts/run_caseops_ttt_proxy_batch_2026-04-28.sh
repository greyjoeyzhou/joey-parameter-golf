#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE_TRAIN_GPT="$ROOT/parameter-golf/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt_decode.py"
SNAPSHOT_PATH="parameter-golf/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt_decode.py"
BATCH_DIR="$ROOT/runs/2026-04-28_CaseOpsTTTProxyBatch"
STATUS_FILE="$BATCH_DIR/status.tsv"
METADATA_FILE="$BATCH_DIR/metadata.tsv"
SUMMARY_FILE="$ROOT/notes/local-caseops-ttt-proxy-batch-2026-04-28.md"

mkdir -p "$BATCH_DIR"
printf 'run_id\tstatus\tstarted_at\tfinished_at\n' > "$STATUS_FILE"
printf 'run_id\tvariant\n' > "$METADATA_FILE"

register_case() {
  printf '%s\t%s\n' "$1" "$2" >> "$METADATA_FILE"
}

register_case "2026-04-28_CaseOps_QKGain50_WD090_ProxyControl_Valid2p5h" "control"
register_case "2026-04-28_CaseOps_QKGain50_WD090_LegalTTT_Proxy_Valid2p5h" "legal_ttt"

COMMON_ENVS=(
  "SOURCE_TRAIN_GPT=$SOURCE_TRAIN_GPT"
  "DATA_DIR=../local_tokenizer_data/caseops_v1"
  "VOCAB_SIZE=8192"
  "MAX_WALLCLOCK_SECONDS=9000"
  "WARMUP_STEPS=1"
  "TRAIN_LOG_EVERY=50"
  "VAL_LOSS_EVERY=0"
  "SLIDING_WINDOW_ENABLED=0"
  "NUM_LOOPS=0"
  "GPTQ_CALIBRATION_BATCHES=1"
  "SKIP_GATES_ENABLED=1"
  "PARALLEL_RESIDUAL_START=7"
  "HESSIAN_CLIP_LAMBDA=0.15"
  "QK_GAIN_INIT=5.0"
  "MUON_WD=0.090"
  "EMBED_WD=0.090"
  "VAL_TOKEN_LIMIT=196608"
)

update_summary() {
  python3 "scripts/summarize_caseops_ttt_proxy_batch.py" --metadata-file "$METADATA_FILE" --status-file "$STATUS_FILE" --output "$SUMMARY_FILE"
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
    --source-script "scripts/train_gpt_decode_sidecar.py"
    --snapshot "$SNAPSHOT_PATH"
    --shim-flash-attn
    --copy-logfile
    --timeout-seconds 12600
  )

  local kv
  for kv in "${COMMON_ENVS[@]}"; do
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

run_case "2026-04-28_CaseOps_QKGain50_WD090_ProxyControl_Valid2p5h" "TTT_ENABLED=0"
run_case "2026-04-28_CaseOps_QKGain50_WD090_LegalTTT_Proxy_Valid2p5h" "TTT_ENABLED=1" "TTT_LR=0.005" "TTT_EPOCHS=3" "TTT_CHUNK_TOKENS=32768" "TTT_FREEZE_BLOCKS=2" "TTT_ENTROPY_HIGH=999" "TTT_ENTROPY_LOW=-999"

update_summary
echo "[$(date -Iseconds)] CaseOps TTT proxy batch complete"
