# Modal Runner Instructions

Run `train_gpt_grouped_gptq.py` on 8xH100 via Modal.

## Prerequisites

```bash
pip install modal
modal setup
```

Volumes are auto-created on first run (`create_if_missing=True`).

## Upload Data (one-time)

The training script expects `fineweb10B_sp{VOCAB_SIZE}` shards and a matching
SentencePiece tokenizer. Upload the dataset for your vocab size:

```bash
# sp4096 (default VOCAB_SIZE)
modal volume put param-golf-data ./data/datasets/fineweb10B_sp4096 /datasets/fineweb10B_sp4096
modal volume put param-golf-data ./data/tokenizers/fineweb_4096_bpe.model /tokenizers/fineweb_4096_bpe.model
```

This uploads ~12 GB and takes several minutes depending on bandwidth.

Verify with:
```bash
modal volume ls param-golf-data /datasets/fineweb10B_sp4096/ | tail -5
modal volume ls param-golf-data /tokenizers/
```

## Training

### Full run (8xH100, 10 min wallclock)

```bash
modal run modal_runner.py::main --iterations 20000 --wallclock 600 --run-id <your-run-id>
```

This will:
1. Spin up 8xH100 (same node, NVLink)
2. Run `torchrun --nproc_per_node=8` with 600s wallclock
3. Stream logs to your terminal in real-time
4. After training: apply EMA, GPTQ quantize, eval, serialize
5. Save logs + model files to the `param-golf-runs` volume
6. Download everything to `./runs/` locally

First run builds the image (~3-5 min for torch + FA3 wheel download). Subsequent
runs use cached image (~30-60s cold start).

### Smoke test (1xH100, 200 steps)

```bash
modal run modal_runner.py::smoke_test
```

### Download logs manually

```bash
modal run modal_runner.py::download_logs
```

Or directly via CLI:
```bash
modal volume get param-golf-runs / ./runs --force
```

## Output Files

After a run, `./runs/` contains:

| File | Description |
|------|-------------|
| `logs/{run_id}.txt` | Full training log (hyperparams, losses, eval results) |
| `final_model.pt` | Full-precision model checkpoint |
| `final_model.int6.ptz` | GPTQ int6 + brotli compressed submission blob |

## Sweeping Hyperparameters

Add a sweep entrypoint to `modal_runner.py`:

```python
@app.local_entrypoint()
def sweep():
    configs = [
        {"MATRIX_LR": "0.018", "EMBED_LR": "0.5"},
        {"MATRIX_LR": "0.020", "EMBED_LR": "0.6"},
        {"MATRIX_LR": "0.022", "EMBED_LR": "0.7"},
    ]
    handles = [
        train.spawn(iterations=20000, max_wallclock_seconds=600,
                    run_id=f"sweep-{i}", extra_env=cfg)
        for i, cfg in enumerate(configs)
    ]
    for h in handles:
        h.get()
```

Each config gets its own 8xH100 container in parallel.

## Changing the Training Script

The script is injected via `add_local_file` with `copy=False`, so editing
`train_gpt_grouped_gptq.py` locally does **not** require an image rebuild.
The updated file is mounted on each run.

To use a different script, change both references in `modal_runner.py`:
- Line ~57: `.add_local_file("your_script.py", "/workspace/your_script.py", ...)`
- Line ~91: `"your_script.py"` in the torchrun command

## Changing the Tokenizer / Vocab Size

Set `VOCAB_SIZE` via `extra_env`. The script auto-constructs data paths from it:

```bash
# Example: use sp8192
modal run modal_runner.py::main --run-id test-sp8192
```

But you must also upload the matching data first:
```bash
modal volume put param-golf-data ./data/datasets/fineweb10B_sp8192 /datasets/fineweb10B_sp8192
modal volume put param-golf-data ./data/tokenizers/fineweb_8192_bpe.model /tokenizers/fineweb_8192_bpe.model
```

And pass the vocab size override. Add to `modal_runner.py` or use `extra_env`
via the Python API:
```python
train.remote(run_id="sp8192-test", extra_env={"VOCAB_SIZE": "8192"})
```

## Cost Estimate

| Config | Rate | Typical Duration | Cost |
|--------|------|-----------------|------|
| 8xH100 full run | ~$26-32/hr | ~15-20 min | ~$7-10 |
| 1xH100 smoke test | ~$3-4/hr | ~5-10 min | ~$0.5-1 |

Container auto-terminates after training completes. 1-hour hard timeout prevents
runaway costs.

## Known Pitfalls

- **FA3 wheel version**: The prebuilt wheel URL in the image must match the
  torch + CUDA version exactly. If torch is upgraded, update the `--find-links`
  URL to the matching `cu<X>_torch<Y>` directory.

- **torch.compile cache**: Each cold start recompiles inductor kernels (~30-60s).
  Not counted in the training wallclock.

- **Modal != competition grader**: val_bpb on Modal will differ by a few basis
  points from the official grader due to different driver/FA3 versions. Use Modal
  for iteration speed, do final scoring on the official environment.
