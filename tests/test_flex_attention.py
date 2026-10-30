"""Regression coverage for short, grouped-query draft training attention."""

import pytest
import torch
import torch.nn.functional as F

from sensespec.models.dflash import _create_dflash_mask_mod
from sensespec.models.ops.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
def test_short_dspark_blocks_have_finite_attention_and_gradients():
    # The SenseNova DSpark smoke shape (16 anchors x 7 tokens) used to select
    # flex_decoding and fail with NoValidChoicesError on H800, before step 1.
    torch.manual_seed(42)
    device = torch.device("cuda")
    context_length, block_size, anchors_count = 128, 7, 16
    query_length = anchors_count * block_size
    kv_length = context_length + query_length
    anchors = torch.linspace(4, 100, anchors_count, device=device).long().unsqueeze(0)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    mask_mod = _create_dflash_mask_mod(anchors, keep, context_length, block_size)
    block_mask = compile_friendly_create_block_mask(
        mask_mod, 1, None, query_length, kv_length, device
    )
    query = torch.randn(
        1, query_length, 16, 256, device=device, dtype=torch.bfloat16, requires_grad=True
    ).transpose(1, 2)
    key = torch.randn(
        1, kv_length, 4, 256, device=device, dtype=torch.bfloat16, requires_grad=True
    ).transpose(1, 2)
    value = torch.randn_like(key, requires_grad=True)
    for tensor in (query, key, value):
        tensor.retain_grad()

    output = compile_friendly_flex_attention(
        query, key, value, block_mask=block_mask, enable_gqa=True
    )
    dense_mask = mask_mod(
        0,
        0,
        torch.arange(query_length, device=device).unsqueeze(1),
        torch.arange(kv_length, device=device).unsqueeze(0),
    )
    reference = F.scaled_dot_product_attention(
        query.detach().float(),
        key.detach().float(),
        value.detach().float(),
        attn_mask=dense_mask,
        enable_gqa=True,
    )
    torch.testing.assert_close(output.float(), reference, atol=0.01, rtol=0.01)
    output.float().square().mean().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().max() > 0
