# variant: sp1024
# mode: compiled
export RUN_ID=2026-04-21_5090NativeBaseline_sp1024
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export ITERATIONS=2000
export WARMUP_STEPS=20
export MAX_WALLCLOCK_SECONDS=0.0
export VAL_LOSS_EVERY=2000
export TRAIN_LOG_EVERY=50
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
export SEED=1337
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
export PATH=/home/joey/Code/joey-parameter-golf/local_bin:/home/joey/.opencode/bin:/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:/home/joey/.volta/bin:/home/joey/.opencode/bin:/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:/home/joey/.volta/bin:/home/joey/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
cd /home/joey/Code/joey-parameter-golf/parameter-golf
/home/joey/Code/joey-parameter-golf/parameter-golf/.venv312x/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py
