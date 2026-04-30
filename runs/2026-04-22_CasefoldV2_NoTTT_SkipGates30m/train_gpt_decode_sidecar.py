#!/usr/bin/env python3

from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "parameter-golf" / "records" / "track_10min_16mb" / "2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence" / "train_gpt_decode.py"


def load_source_module():
    source_path = Path(os.environ.get("SOURCE_TRAIN_GPT", str(DEFAULT_SOURCE))).resolve()
    spec = importlib.util.spec_from_file_location("pg_sidecar_source", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load source script: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source_path


module, SOURCE_PATH = load_source_module()


_OriginalValidationData = module.ValidationData


class ValidationDataWithSidecar(_OriginalValidationData):
    def __init__(self, h, device):
        super().__init__(h, device)
        pattern = str(Path(h.datasets_dir) / "fineweb_val_bytes_*.bin")
        files = [Path(p) for p in sorted(glob.glob(pattern))]
        if files:
            byte_counts = torch.cat([module.load_data_shard(file) for file in files]).contiguous().to(torch.int64)
            if byte_counts.numel() != self.val_tokens.numel():
                raise ValueError(
                    f"Validation byte sidecar length mismatch: tokens={self.val_tokens.numel()} bytes={byte_counts.numel()}"
                )
            self.val_byte_counts = byte_counts
        else:
            self.val_byte_counts = None


def eval_val_with_sidecar(h, device, val_data, model):
    if getattr(val_data, "val_byte_counts", None) is None:
        return module.eval_val(h, device, val_data, model)

    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, "
            f"GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * h.rank) // h.world_size
    seq_end = (total_seqs * (h.rank + 1)) // h.world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            byte_slice = val_data.val_byte_counts[raw_start + 1:raw_end].to(device=device, dtype=torch.float64, non_blocking=True)
            val_byte_count += byte_slice.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    model.train()
    return module._loss_bpb(val_loss_sum, val_token_count, val_byte_count)


module.ValidationData = ValidationDataWithSidecar
module.eval_val = eval_val_with_sidecar


def main():
    module.main()


if __name__ == "__main__":
    main()
