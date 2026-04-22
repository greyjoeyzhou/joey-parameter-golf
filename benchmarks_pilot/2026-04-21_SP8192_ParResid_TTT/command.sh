export PATH=/home/joey/Code/joey-parameter-golf/local_bin:$PATH
export PYTHONPATH=/home/joey/Code/joey-parameter-golf/local_shims
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
export MAX_WALLCLOCK_SECONDS=1200
export TRAIN_LOG_EVERY=50
export VAL_LOSS_EVERY=500
export WARMUP_STEPS=1
export GPTQ_CALIBRATION_BATCHES=1
export SEED=42
export TTT_ENABLED=1
export PARALLEL_START_LAYER=7
cd /home/joey/Code/joey-parameter-golf/parameter-golf
.venv312x/bin/torchrun --standalone --nproc_per_node=1 records/track_10min_16mb/2026-04-08_SP8192_ParallelResid_ScoreFirstTTT/train_gpt.py
