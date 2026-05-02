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
_OriginalEvalVal = module.eval_val
_OriginalEvalValSliding = module.eval_val_sliding
_OriginalEvalValTTT = module.eval_val_ttt


def _truncate_val_if_requested(h, val_tokens, val_byte_counts):
    limit_tokens = int(os.environ.get("VAL_TOKEN_LIMIT", "0"))
    if limit_tokens <= 0:
        return val_tokens, val_byte_counts

    usable = min(val_tokens.numel() - 1, limit_tokens)
    usable = (usable // h.eval_seq_len) * h.eval_seq_len
    if usable <= 0:
        raise ValueError(f"VAL_TOKEN_LIMIT={limit_tokens} is too small for TRAIN_SEQ_LEN={h.eval_seq_len}")

    val_tokens = val_tokens[: usable + 1]
    if val_byte_counts is not None:
        val_byte_counts = val_byte_counts[: usable]
    return val_tokens, val_byte_counts


def _sum_lut_bytes(val_data, tgt_ids, prev_ids, device):
    token_bytes = val_data.base_bytes_lut[tgt_ids].to(device=device, dtype=torch.float64, non_blocking=True)
    token_bytes += (
        val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]
    ).to(dtype=torch.float64)
    return token_bytes.sum()


class ValidationDataWithSidecar(_OriginalValidationData):
    def __init__(self, h, device):
        self.sp = module.spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}"
            )
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = module.build_sentencepiece_luts(
            self.sp, h.vocab_size, device
        )

        token_files = [
            p for p in sorted(Path(h.datasets_dir).glob("fineweb_val_*.bin")) if "_bytes_" not in p.name
        ]
        if not token_files:
            raise FileNotFoundError(f"No validation token files found under {h.datasets_dir}")
        tokens = torch.cat([module.load_data_shard(file) for file in token_files]).contiguous()
        usable = ((tokens.numel() - 1) // h.eval_seq_len) * h.eval_seq_len
        if usable <= 0:
            raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={h.eval_seq_len}")
        self.val_tokens = tokens[: usable + 1]

        files = [Path(p) for p in sorted(glob.glob(str(Path(h.datasets_dir) / "fineweb_val_bytes_*.bin")))]
        if files:
            byte_counts = torch.cat([module.load_data_shard(file) for file in files]).contiguous().to(torch.int64)
            if byte_counts.numel() < self.val_tokens.numel() - 1:
                raise ValueError(
                    f"Validation byte sidecar length mismatch: tokens={self.val_tokens.numel()} bytes={byte_counts.numel()}"
                )
            self.val_byte_counts = byte_counts[: self.val_tokens.numel() - 1]
        else:
            self.val_byte_counts = None

        self.val_tokens, self.val_byte_counts = _truncate_val_if_requested(h, self.val_tokens, self.val_byte_counts)


def eval_val_with_sidecar(h, device, val_data, model):
    if getattr(val_data, "val_byte_counts", None) is None:
        return _OriginalEvalVal(h, device, val_data, model)

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


def eval_val_sliding_with_sidecar(h, device, val_data, base_model, batch_seqs=32):
    if getattr(val_data, "val_byte_counts", None) is None:
        return _OriginalEvalValSliding(h, device, val_data, base_model, batch_seqs)

    base_model.eval()
    logits_fn = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, h.eval_stride) if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_s = total_windows * h.rank // h.world_size
    my_e = total_windows * (h.rank + 1) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x_batch)
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else context_size
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                byte_slice = val_data.val_byte_counts[ws + s:ws + wlen].to(
                    device=device, dtype=torch.float64, non_blocking=True
                )
                byte_count += byte_slice.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return module._loss_bpb(loss_sum, token_count, byte_count)


def eval_val_ttt_with_sidecar(h, device, val_data, base_model, batch_seqs=32):
    if getattr(val_data, "val_byte_counts", None) is None:
        return _OriginalEvalValTTT(h, device, val_data, base_model, batch_seqs)
    if not h.ttt_enabled:
        return eval_val_sliding_with_sidecar(h, device, val_data, base_model, batch_seqs)

    seq_len = h.eval_seq_len
    stride = h.eval_stride
    context_size = seq_len - stride
    total_tokens = val_data.val_tokens.numel() - 1
    chunk_size = h.ttt_chunk_tokens

    window_starts = [ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    chunk_boundaries = list(range(0, total_windows, max(1, chunk_size // stride)))
    if chunk_boundaries[-1] != total_windows:
        chunk_boundaries.append(total_windows)
    chunk_windows = [
        window_starts[chunk_boundaries[c]:chunk_boundaries[c + 1]]
        for c in range(len(chunk_boundaries) - 1)
    ]

    module.log(
        f"ttt: {len(chunk_windows)} chunks, {total_windows} windows, "
        f"freeze_blocks={h.ttt_freeze_blocks}, lr={h.ttt_lr}, epochs={h.ttt_epochs}"
    )

    for p in base_model.parameters():
        p.requires_grad = False
    ttt_params = []
    for i, block in enumerate(base_model.blocks):
        if i >= h.ttt_freeze_blocks:
            for p in block.parameters():
                p.requires_grad = True
                ttt_params.append(p)
    for p in base_model.final_norm.parameters():
        p.requires_grad = True
        ttt_params.append(p)

    saved_state = {id(p): p.data.clone() for p in ttt_params}

    proj_params, fc_params, other_params = [], [], []
    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        if ".proj." in name or name.endswith(".proj.weight"):
            proj_params.append(p)
        elif ".fc." in name or name.endswith(".fc.weight"):
            fc_params.append(p)
        else:
            other_params.append(p)

    ttt_lr = h.ttt_lr
    param_groups = [
        pg for pg in [
            {"params": proj_params, "lr": ttt_lr * 3.0, "_lr_mult": 3.0} if proj_params else None,
            {"params": fc_params, "lr": ttt_lr * 0.5, "_lr_mult": 0.5} if fc_params else None,
            {"params": other_params, "lr": ttt_lr, "_lr_mult": 1.0} if other_params else None,
        ] if pg is not None
    ]
    optimizer = torch.optim.SGD(param_groups, lr=ttt_lr, momentum=0.9)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    for ci, windows in enumerate(chunk_windows):
        if not windows:
            continue

        my_s = (len(windows) * h.rank) // h.world_size
        my_e = (len(windows) * (h.rank + 1)) // h.world_size
        my_windows = windows[my_s:my_e]

        base_model.eval()
        chunk_entropy = 0.0
        chunk_count = 0
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    wlens.append(wlen)
                    chunk = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk[:-1]
                    y_batch[i, :wlen] = chunk[1:]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch).float()

                for i in range(bsz):
                    wl = wlens[i]
                    if wl > 0:
                        nll = torch.nn.functional.cross_entropy(logits[i, :wl], y_batch[i, :wl], reduction="sum")
                        chunk_entropy += nll.item()
                        chunk_count += wl

                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored_nll = torch.nn.functional.cross_entropy(
                        logits[i, s:wlen], y_batch[i, s:wlen], reduction="none"
                    ).to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    byte_slice = val_data.val_byte_counts[ws + s:ws + wlen].to(
                        device=device, dtype=torch.float64, non_blocking=True
                    )
                    byte_count += byte_slice.sum()

        if chunk_count > 0:
            chunk_nll_avg = chunk_entropy / chunk_count
            if chunk_nll_avg > h.ttt_entropy_high:
                epochs = h.ttt_epochs + 1
            elif chunk_nll_avg < h.ttt_entropy_low:
                epochs = max(h.ttt_epochs - 1, 1)
            else:
                epochs = h.ttt_epochs
        else:
            epochs = h.ttt_epochs

        base_model.train()
        for _epoch in range(epochs):
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    chunk = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk[:-1]
                    y_batch[i, :wlen] = chunk[1:]

                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ttt_logits = base_model.forward_logits(x_batch)
                    ttt_loss = torch.nn.functional.cross_entropy(
                        ttt_logits.reshape(-1, ttt_logits.size(-1)),
                        y_batch.reshape(-1),
                    )
                ttt_loss.backward()

                if h.ttt_ns_steps > 0:
                    with torch.no_grad():
                        for pg in param_groups:
                            for p in pg["params"]:
                                if p.grad is None:
                                    continue
                                g = p.grad.detach().float()
                                if g.ndim == 2:
                                    g = module.zeropower_via_newtonschulz5(g, steps=h.ttt_ns_steps)
                                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                                    p.data.add_(g.to(p.dtype), alpha=-ttt_lr * pg["_lr_mult"])
                                    p.grad = None
                optimizer.step()

        if ci % 10 == 0 or ci == len(chunk_windows) - 1:
            module.log(f"ttt_chunk: {ci + 1}/{len(chunk_windows)} epochs={epochs}")

    with torch.no_grad():
        for p in ttt_params:
            p.data.copy_(saved_state[id(p)])

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    for p in base_model.parameters():
        p.requires_grad = True
    base_model.train()
    return module._loss_bpb(loss_sum, token_count, byte_count)


module.ValidationData = ValidationDataWithSidecar
module.eval_val = eval_val_with_sidecar
module.eval_val_sliding = eval_val_sliding_with_sidecar
module.eval_val_ttt = eval_val_ttt_with_sidecar


def main():
    module.main()


if __name__ == "__main__":
    main()
