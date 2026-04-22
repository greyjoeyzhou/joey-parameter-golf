import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _to_bhsd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2)


def _to_bshd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
):
    if window_size != (-1, -1):
        raise NotImplementedError("windowed attention is not supported by this local shim")
    if alibi_slopes is not None:
        raise NotImplementedError("ALiBi is not supported by this local shim")
    del deterministic

    # Prefer fused SDPA backends so the Blackwell fallback path stays as close as possible
    # to the native local trainer instead of silently dropping to math attention.
    with sdpa_kernel(
        [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    ):
        out = F.scaled_dot_product_attention(
            _to_bhsd(q),
            _to_bhsd(k),
            _to_bhsd(v),
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
            enable_gqa=q.size(2) != k.size(2),
        )
    return _to_bshd(out)


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
):
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )
