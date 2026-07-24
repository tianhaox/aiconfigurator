from __future__ import annotations

import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from flashinfer import norm

    if x.dim() == 1:
        return norm.rmsnorm(x.unsqueeze(0), weight, eps).squeeze(0)
    return norm.rmsnorm(x, weight, eps)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    from flashinfer import activation

    packed = torch.cat((gate, up), dim=-1)
    if packed.dim() == 1:
        return activation.silu_and_mul(packed.unsqueeze(0)).squeeze(0)
    return activation.silu_and_mul(packed)


def tiny_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    from flashinfer import gemm

    input_2d = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty((input_2d.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device)
    gemm.tinygemm_bf16(input_2d, weight, out, bias=bias)
    if x.dim() == 1:
        return out.squeeze(0)
    return out.reshape(*x.shape[:-1], weight.shape[0])


def apply_rope_single(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import rope

    q_rope, k_rope = rope.apply_rope(
        q.unsqueeze(0),
        k.unsqueeze(0),
        indptr,
        offsets,
        rotary_dim=q.shape[-1],
        interleave=False,
        rope_theta=rope_theta,
    )
    return q_rope.squeeze(0), k_rope.squeeze(0)


def apply_rope_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import rope

    return rope.apply_rope(
        q,
        k,
        indptr,
        offsets,
        rotary_dim=q.shape[-1],
        interleave=False,
        rope_theta=rope_theta,
    )


def single_decode_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    import flashinfer

    return flashinfer.single_decode_with_kv_cache(
        q,
        k,
        v,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        use_tensor_cores=False,
        sm_scale=sm_scale,
    )


def single_prefill_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    import flashinfer

    return flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=True,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        sm_scale=sm_scale,
    )
