# SPDX-License-Identifier: Apache-2.0
"""ResServe decode on the TRT-LLM backend's native NVFP4 cache views."""

import torch


def forward_resserve_decode(backend, q, layer, forward_batch, sinks=None):
    """Replace decode only, after the engine has appended this step's KV."""
    from flashinfer.resserve import paged_decode_residue

    if (
        not backend.is_nvfp4_kvcache
        or backend.page_size != 32
        or backend.is_xqa_impl
        or not getattr(backend, "decode_uses_native_fp4", False)
    ):
        raise ValueError("ResServe requires SM100 native NVFP4 KV with page size 32")
    if q.dtype != torch.bfloat16:
        raise ValueError("ResServe requires original BF16 queries")
    if sinks is not None or layer.sliding_window_size not in (None, -1):
        raise NotImplementedError("ResServe does not support sinks or sliding windows")
    q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
    (k, v), (ks, vs) = backend._get_nvfp4_decode_kv_cache(layer)
    pages, heads = k.shape[:2]
    key_scale, value_scale = backend._get_nvfp4_bmm_scales(layer)
    table = backend._get_layer_page_table(layer, forward_batch)
    result = paged_decode_residue(
        q,
        k,
        v,
        ks,
        vs.view(pages, heads, 8, layer.head_dim // 16, 4),
        table,
        backend.forward_metadata.cache_seqlens_int32,
        sm_scale=layer.scaling,
        key_scale=key_scale,
        value_scale=value_scale,
        max_seq_len=min(backend.max_context_len, table.shape[1] * 32),
    )
    return result.view(-1, layer.tp_q_head_num * layer.head_dim)
