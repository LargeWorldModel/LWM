"""This module contains ring attention forward and backward pass, supporting both blockwise computation and TPU-compatible fused attention.
It features blockwise computation for feedforward networks to reduce memory cost.
For more details, refer to 'RingAttention' at https://arxiv.org/abs/2305.19370 and 'Blockwise Parallel Transformers' at https://arxiv.org/abs/2310.01889.
"""

import numpy as np
import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from einops import rearrange
from functools import partial
import dataclasses
import functools
from typing import Any, NamedTuple


def _ring_attention_fwd(q, k, v, attn_bias, segment_ids, axis_name, float32_logits, blockwise_kwargs):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(q.dtype)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)
    axis_size = lax.psum(1, axis_name)
    q_block_size, kv_block_size = q_len, kv_len # assumes this function is pre-sharded inside shard_map
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]

    def scan_kv_block(carry, idx):
        prev_max_score, numerator, denominator, k, v = carry
        q_block_idx = lax.axis_index(axis_name)
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (kv_block_size // key_chunk_size)
        numerator, denominator, max_score = _blockwise_attention_fwd(q, k, v, (numerator, denominator, prev_max_score), q_chunk_idx_start, k_chunk_idx_start, bias=attn_bias, segment_ids=segment_ids, **blockwise_kwargs)
        k, v = map(lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]), (k, v))
        return (max_score, numerator, denominator, k, v), None
    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype)
    (max_score, numerator, denominator, _, _), _ = lax.scan(scan_kv_block,
        init=(prev_max_score, numerator, denominator, k, v), xs=jnp.arange(0, axis_size))
    output = numerator / rearrange(denominator, 'b h q -> b q h')[..., None]
    return output.astype(v.dtype), (output, q, k, v, attn_bias, segment_ids, denominator, max_score)

def _ring_attention_bwd(axis_name, float32_logits, blockwise_kwargs, res, g):
    del float32_logits
    output, q, k, v, attn_bias, segment_ids, denominator, max_score = res
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    axis_size = lax.psum(1, axis_name)
    dq = jnp.zeros_like(q, dtype=q.dtype)
    dk = jnp.zeros_like(k, dtype=k.dtype)
    dv = jnp.zeros_like(v, dtype=k.dtype)
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]
    q_block_size, kv_block_size = q_len, kv_len # assumes this function is pre-sharded inside shard_map
    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        q_block_idx = lax.axis_index(axis_name)
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (kv_block_size // key_chunk_size)
        dq, dk, dv = _blockwise_attention_bwd(q, k, v, g, (dq, dk, dv, output, denominator, max_score), q_chunk_idx_start, k_chunk_idx_start, bias=attn_bias, segment_ids=segment_ids, **blockwise_kwargs)
        k, v, dk, dv = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v, dk, dv))
        return (dq, dk, dv, k, v), None
    (dq, dk, dv, k, v), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, k, v), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(k.dtype)
    return dq, dk, dv, None, None

@partial(jax.custom_vjp, nondiff_argnums=[5, 6, 7])
def ring_attention(q, k, v, attn_bias, segment_ids, axis_name, float32_logits, blockwise_kwargs):
    y, _ = _ring_attention_fwd(q, k, v, attn_bias, segment_ids, axis_name, float32_logits, blockwise_kwargs)
    return y

ring_attention.defvjp(_ring_attention_fwd, _ring_attention_bwd)


def _ring_attention_standard_fwd(q, k, v, attn_mask, axis_name, float32_logits):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    batch, q_len, num_heads, _ = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(q.dtype)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)
    axis_size = lax.psum(1, axis_name)
    scale = jnp.sqrt(q.shape[-1])
    def scan_kv_block(carry, idx):
        prev_max_score, numerator, denominator, k, v = carry
        mask = lax.dynamic_slice_in_dim(attn_mask,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1)
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / scale
        attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
        max_score = jnp.maximum(prev_max_score, jnp.max(attn_weights, axis=-1))
        exp_weights = jnp.exp(attn_weights - max_score[..., None])
        correction = rearrange(jnp.exp(prev_max_score - max_score), 'b h q -> b q h')[..., None]
        numerator = numerator * correction + jnp.einsum("bhqk,bkhd->bqhd", exp_weights, v)
        denominator = denominator * jnp.exp(prev_max_score - max_score) + jnp.sum(exp_weights, axis=-1)
        k, v = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v))
        return (max_score, numerator, denominator, k, v), None
    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype)
    (max_score, numerator, denominator, _, _), _ = lax.scan(scan_kv_block,
    init=(prev_max_score, numerator, denominator, k, v), xs=jnp.arange(0, axis_size))
    output = numerator / rearrange(denominator, 'b h q -> b q h')[..., None]
    return output.astype(v.dtype), (output, q, k, v, attn_mask, numerator, denominator, max_score)

def _ring_attention_standard_bwd(axis_name, float32_logits, res, g):
    del float32_logits
    axis_size = lax.psum(1, axis_name)
    output, q, k, v, attn_mask, numerator, denominator, max_score = res
    dq = jnp.zeros_like(q, dtype=jnp.float32)
    dk = jnp.zeros_like(k, dtype=jnp.float32)
    dv = jnp.zeros_like(v, dtype=jnp.float32)
    batch, kv_len, num_heads, dim_per_head = k.shape
    scale = jnp.sqrt(q.shape[-1])
    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        mask = lax.dynamic_slice_in_dim(attn_mask,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1)
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / scale
        attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
        exp_weights = jnp.exp(attn_weights - max_score[..., None]) / denominator[..., None]
        ds = jnp.einsum("bqhd,bkhd->bhqk", g, v)
        dl = (ds - jnp.einsum("bqhd,bqhd->bhq", g, output)[..., None]) * exp_weights
        dq = dq + jnp.einsum("bhqk,bkhd->bqhd", dl, k) / scale
        dk = dk + jnp.einsum("bqhd,bhqk->bkhd", q, dl) / scale
        dv = dv + jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g)
        k, v, dk, dv = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v, dk, dv))
        return (dq, dk, dv, k, v), None
    (dq, dk, dv, k, v), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, k, v), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
    return dq, dk, dv, None

@partial(jax.custom_vjp, nondiff_argnums=[4, 5])
def ring_attention_standard(q, k, v, attn_mask, axis_name, float32_logits=True):
    y, _ = _ring_attention_standard_fwd(q, k, v, attn_mask, axis_name, float32_logits)
    return y

ring_attention_standard.defvjp(_ring_attention_standard_fwd, _ring_attention_standard_bwd)


def _blockwise_attention_fwd(q, k, v, carry, q_chunk_idx_start, k_chunk_idx_start, bias, segment_ids, causal, query_chunk_size,
                             key_chunk_size, deterministic, dropout_rng, attn_pdrop, dtype, policy, precision, prevent_cse):
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    batch, kv_len, num_heads, dim_per_head = v.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    q = q.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    k = k.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    v = v.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    q, k, v = map(lambda x: jnp.moveaxis(x, 1, 0), (q, k, v))

    numerator, denominator, max_score = carry
    numerator = numerator.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    numerator = jnp.moveaxis(numerator, 1, 0)
    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))
    denominator, max_score = map(lambda x: rearrange(x, 'b h n c -> n b h c'), (denominator, max_score))

    scale = jnp.sqrt(q.shape[-1])
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size, bias, segment_ids, deterministic,
        attn_dropout, attn_pdrop, causal, dtype)
    def scan_attention(_, scan):
        q_chunk, numerator_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan
        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, scan):
            k_chunk, value_chunk, k_chunk_idx = scan
            numerator_chunk, denominator_chunk, prev_max_score_chunk = carry
            attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q_chunk, k_chunk, precision=precision) / scale
            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            attn_weights = attn_weights + bias_chunk

            max_score_chunk = jnp.maximum(prev_max_score_chunk, jnp.max(attn_weights, axis=-1))
            max_score_chunk = lax.stop_gradient(max_score_chunk)
            exp_weights = jnp.exp(attn_weights - max_score_chunk[..., None])
            exp_values = jnp.einsum('bhqk,bkhd->bqhd', exp_weights, value_chunk, precision=precision)
            correction = rearrange(jnp.exp(prev_max_score_chunk - max_score_chunk), 'b h q -> b q h')[..., None]
            numerator_chunk = numerator_chunk * correction + exp_values
            denominator_chunk = denominator_chunk * jnp.exp(prev_max_score_chunk - max_score_chunk) + exp_weights.sum(axis=-1)
            return (numerator_chunk, denominator_chunk, max_score_chunk), None

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, k_chunk_idx = args
            skip_block = jnp.array(False)
            if causal:
                skip_block = q_chunk_idx_start + q_chunk_idx < k_chunk_idx_start + k_chunk_idx
            return jax.lax.cond(
                skip_block,
                lambda carry, args: (carry, None),
                scan_kv_block,
                carry,
                args
            )

        (numerator_chunk, denominator_chunk, max_score_chunk), _ = lax.scan(
            skip_upper_half, init=(numerator_chunk, denominator_chunk, max_score_chunk), xs=(k, v, jnp.arange(0, num_kv))
        )
        output_chunk = numerator_chunk / rearrange(denominator_chunk, 'b h q -> b q h')[..., None].astype(dtype)
        return (), (output_chunk, numerator_chunk, denominator_chunk, max_score_chunk)
    _, (_, numerator, denominator, max_score) = lax.scan(scan_attention, init=(), xs=(q, numerator, denominator, max_score, jnp.arange(0, num_q)))

    numerator = jnp.moveaxis(numerator, 1, 0)
    numerator = numerator.reshape((batch, q_len, num_heads, dim_per_head))
    denominator, max_score = map(lambda x: rearrange(x, 'n b h c -> b h n c'), (denominator, max_score))
    denominator = denominator.reshape((batch, num_heads, q_len))
    max_score = max_score.reshape((batch, num_heads, q_len))

    return numerator, denominator, max_score

def _blockwise_attention_bwd(q, k, v, g, carry, q_chunk_idx_start, k_chunk_idx_start, bias, segment_ids, causal, query_chunk_size, key_chunk_size, deterministic, dropout_rng, attn_pdrop, dtype, policy, precision, prevent_cse):
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    batch, kv_len, num_heads, dim_per_head = v.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    dq, dk, dv, output, denominator, max_score = carry

    g = g.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dq = dq.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dk = dk.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    dv = dv.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    output = output.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    g, dq, dk, dv, output = map(lambda x: jnp.moveaxis(x, 1, 0), (g, dq, dk, dv, output))

    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))
    denominator, max_score = map(lambda x: rearrange(x, 'b h n c -> n b h c'), (denominator, max_score))

    q = q.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    k = k.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    v = v.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    q, k, v = map(lambda x: jnp.moveaxis(x, 1, 0), (q, k, v))

    scale = jnp.sqrt(q.shape[-1])
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size, bias, segment_ids, deterministic,
        attn_dropout, attn_pdrop, causal, dtype)
    def scan_attention(carry, scan):
        dk, dv = carry
        q_chunk, dq_chunk, g_chunk, output_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan
        dl_part = jnp.einsum("bqhd,bqhd->bhq", g_chunk, output_chunk)[..., None]
        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, scan):
            k_chunk, value_chunk, k_chunk_idx = scan
            dq_chunk = carry
            attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q_chunk, k_chunk, precision=precision) / scale
            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            attn_weights = attn_weights + bias_chunk
            exp_weights = jnp.exp(attn_weights - max_score_chunk[..., None]) / denominator_chunk[..., None]

            ds = jnp.einsum("bqhd,bkhd->bhqk", g_chunk, value_chunk)
            dl = (ds - dl_part) * exp_weights
            dq_chunk = dq_chunk + jnp.einsum("bhqk,bkhd->bqhd", dl, k_chunk) / scale
            dk_chunk = jnp.einsum("bqhd,bhqk->bkhd", q_chunk, dl) / scale
            dv_chunk = jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g_chunk)
            return dq_chunk, (dk_chunk, dv_chunk)

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, k_chunk_idx = args
            skip_block = jnp.array(False)
            if causal:
                skip_block = q_chunk_idx_start + q_chunk_idx < k_chunk_idx_start + k_chunk_idx
            return lax.cond(
                skip_block,
                lambda carry, args: (
                    carry, (
                        jnp.zeros((batch, key_chunk_size, num_heads, dim_per_head), dtype=dk.dtype),
                        jnp.zeros((batch, key_chunk_size, num_heads, dim_per_head), dtype=dk.dtype),
                    )
                ),
                scan_kv_block,
                carry,
                args
            )

        dq_chunk, (dk_part, dv_part) = lax.scan(
            skip_upper_half, init=dq_chunk, xs=(k, v, jnp.arange(0, num_kv))
        )
        return (dk + dk_part, dv + dv_part), dq_chunk
    (dk, dv), dq = lax.scan(scan_attention, init=(dk, dv), xs=(q, dq, g, output, denominator, max_score, jnp.arange(0, num_q)))

    dq, dk, dv = map(lambda x: jnp.moveaxis(x, 1, 0), (dq, dk, dv))
    dq = dq.reshape((batch, q_len, num_heads, dim_per_head))
    dk = dk.reshape((batch, kv_len, num_heads, dim_per_head))
    dv = dv.reshape((batch, kv_len, num_heads, dim_per_head))

    return dq, dk, dv


# Blockwise feedforward network for memory-efficient training
def blockwise_ffn(remat_ffn, inputs, chunk_size, deterministic):
    inputs = rearrange(inputs, 'b (c n) d -> b c n d', c=chunk_size)
    def scan_ffn(remat_ffn, carry, hidden_states):
        outputs = remat_ffn(hidden_states, deterministic=deterministic)
        return carry, outputs
    scan_axis = inputs.ndim - 2
    _, output = nn.scan(
        scan_ffn,
        variable_broadcast="params",
        split_rngs={"params": False, "dropout": True},
        in_axes=scan_axis,
        out_axes=scan_axis,
    )(remat_ffn, None, inputs)
    output = rearrange(output, 'b c n d -> b (c n) d')
    return output


def _chunk_attention_bias(query_chunk_size, key_chunk_size,
            bias, segment_ids, deterministic, attn_dropout, attn_pdrop, causal,
            dtype, query_chunk_idx, key_chunk_idx):
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = jnp.zeros((1, 1, 1, 1), dtype=dtype)
    if bias is not None:
        chunk_bias = lax.dynamic_slice(
            bias,
            start_indices=(0, 0, 0, key_offset),
            slice_sizes=(*bias.shape[:2], min(bias.shape[-2], query_chunk_size), min(bias.shape[-1], key_chunk_size)),
        )

    if segment_ids is not None:
        q_segment_ids = lax.dynamic_slice(
            segment_ids,
            start_indices=(0, query_offset),
            slice_sizes=(segment_ids.shape[0], query_chunk_size)
        )
        k_segment_ids = lax.dynamic_slice(
            segment_ids,
            start_indices=(0, key_offset),
            slice_sizes=(segment_ids.shape[0], key_chunk_size)
        )
        segment_ids_mask = q_segment_ids[:, :, None] != k_segment_ids[:, None, :]
        segment_ids_mask = segment_ids_mask[:, None] # B1QK
        segment_ids_bias = segment_ids_mask * jnp.finfo(dtype).min
        chunk_bias += segment_ids_bias

    if causal:
        query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)
        key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)
        offset = query_offset - key_offset
        query_idx += offset
        causal_mask_value = (query_idx < key_idx) * jnp.finfo(dtype).min
        chunk_bias += causal_mask_value.reshape(1, 1, *causal_mask_value.shape)

    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_slice = lax.dynamic_slice(
            attn_dropout,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *attn_dropout.shape[:2],
                min(attn_dropout.shape[-2], query_chunk_size),
                min(attn_dropout.shape[-1], key_chunk_size),
            ),
        )
        chunk_bias += attn_dropout_slice * jnp.finfo(dtype).min
    return chunk_bias.astype(dtype)


def _ring_flash_attention_fwd_tpu(q, k, v, attn_bias, segment_ids, axis_name, float32_logits, blockwise_kwargs):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    q, k, v = map(lambda x: rearrange(x, 'b q h d -> b h q d'), [q, k, v])
    batch, num_heads, q_len, dim_per_head = q.shape
    batch, num_heads, kv_len, dim_per_head = k.shape
    attn_bias = attn_bias[:, 0, 0] # (batch, k_len)

    o = jnp.zeros((batch, num_heads, q_len, dim_per_head)).astype(q.dtype)
    l = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)
    m = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype)

    axis_size = lax.psum(1, axis_name)
    q_block_size, kv_block_size = q_len, kv_len # assumes this function is pre-sharded inside shard_map
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]
    if segment_ids is not None:
        q_segment_ids = lax.dynamic_slice_in_dim(
            segment_ids,
            lax.axis_index(axis_name) * q_len, q_len, axis=-1
        )

    block_sizes = BlockSizes(
        block_q=query_chunk_size,
        block_k_major=key_chunk_size,
        block_k=key_chunk_size,
        block_b=1,
        block_q_major_dkv=query_chunk_size,
        block_k_major_dkv=key_chunk_size,
        block_k_dkv=key_chunk_size,
        block_q_dkv=query_chunk_size,
        block_k_major_dq=key_chunk_size,
        block_k_dq=key_chunk_size,
        block_q_dq=query_chunk_size,
    )

    scale = q.shape[-1] ** -0.5
    def scan_kv_block(carry, idx):
        o, l, m, k, v = carry
        attn_bias_slice = lax.dynamic_slice_in_dim(attn_bias,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1
        )
        attn_bias_slice = None # TODO
        if segment_ids is not None:
            kv_segment_ids = lax.dynamic_slice_in_dim(
                segment_ids,
                (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1
            )
            segment_ids_slice = SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
        else:
            segment_ids_slice = None
        q_block_idx = lax.axis_index(axis_name)
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (kv_block_size // key_chunk_size)
        o, l, m = _flash_attention_fwd(
            q, k, v,
            carry=(o, l, m),
            q_chunk_idx_start=q_chunk_idx_start,
            k_chunk_idx_start=k_chunk_idx_start,
            ab=attn_bias_slice,
            segment_ids=segment_ids_slice,
            save_residuals=False,
            causal=blockwise_kwargs["causal"],
            sm_scale=scale,
            block_sizes=block_sizes,
            debug=False
        )
        k, v = map(lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]), (k, v))
        return (o, l, m, k, v), None
    (o, l, m, _, _), _ = lax.scan(scan_kv_block,
        init=(o, l, m, k, v), xs=jnp.arange(0, axis_size))
    output = rearrange(o.astype(v.dtype), 'b h q d -> b q h d')
    return output, (o, q, k, v, attn_bias, segment_ids, l, m)

def _ring_flash_attention_bwd_tpu(axis_name, float32_logits, blockwise_kwargs, res, g):
    del float32_logits
    o, q, k, v, attn_bias, segment_ids, l, m = res
    batch, num_heads, kv_len, dim_per_head = k.shape
    axis_size = lax.psum(1, axis_name)
    dq = jnp.zeros_like(q, dtype=jnp.float32)
    dk = jnp.zeros_like(k, dtype=jnp.float32)
    dv = jnp.zeros_like(v, dtype=jnp.float32)
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]
    q_block_size, kv_block_size = q.shape[2], k.shape[2] # assumes this function is pre-sharded inside shard_map
    scale = q.shape[-1] ** -0.5

    if segment_ids is not None:
        q_segment_ids = lax.dynamic_slice_in_dim(
            segment_ids,
            lax.axis_index(axis_name) * q_block_size, q_block_size, axis=-1
        )

    g = rearrange(g, 'b q h d -> b h q d')

    block_sizes = BlockSizes(
        block_q=query_chunk_size,
        block_k_major=key_chunk_size,
        block_k=key_chunk_size,
        block_b=1,
        block_q_major_dkv=query_chunk_size,
        block_k_major_dkv=key_chunk_size,
        block_k_dkv=key_chunk_size,
        block_q_dkv=query_chunk_size,
        block_k_major_dq=key_chunk_size,
        block_k_dq=key_chunk_size,
        block_q_dq=query_chunk_size,
    )

    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        attn_bias_slice = lax.dynamic_slice_in_dim(attn_bias,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1
        )
        attn_bias_slice = None # TODO
        if segment_ids is not None:
            kv_segment_ids = lax.dynamic_slice_in_dim(
                segment_ids,
                (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1
            )
            segment_ids_slice = SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
        else:
            segment_ids_slice = None
        q_block_idx = lax.axis_index(axis_name)
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (kv_block_size // key_chunk_size)
        dq_i, dk_i, dv_i, = _flash_attention_bwd(
            save_residuals=False,
            causal=blockwise_kwargs["causal"],
            sm_scale=scale,
            block_sizes=block_sizes,
            debug=False,
            q_chunk_idx_start=q_chunk_idx_start,
            k_chunk_idx_start=k_chunk_idx_start,
            residuals=(q, k, v, attn_bias_slice, segment_ids_slice, o, l, m),
            do=g
        )
        dq += dq_i
        dk += dk_i
        dv += dv_i
        k, v, dk, dv = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v, dk, dv))
        return (dq, dk, dv, k, v), None
    (dq, dk, dv, k, v), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, k, v), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
    dq, dk, dv = map(lambda x: rearrange(x, 'b h q d -> b q h d'), (dq, dk, dv))
    return dq, dk, dv, None, None

@partial(jax.custom_vjp, nondiff_argnums=[5, 6, 7])
def ring_flash_attention_tpu(q, k, v, attn_bias, segment_ids, axis_name, float32_logits, blockwise_kwargs):
    y, _ = _ring_flash_attention_fwd_tpu(q, k, v, attn_bias, segment_ids, axis_name, float32_logits, blockwise_kwargs)
    return y

ring_flash_attention_tpu.defvjp(_ring_flash_attention_fwd_tpu, _ring_flash_attention_bwd_tpu)

# TPU-compatible fused attention functions for RingAttention
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8

class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

    SegmentIds are used to generate segment mask, which prevents attention between
    different segments in the input sequence. Each array is a list of ids
    (integers).
    Only the token with the same id can attend to each other.

    Attributes:
      q: segment ids along the Q sequence.
      kv: segment ids along the KV sequence.
    """

    q: jax.Array  # [q_seq_len]
    kv: jax.Array  # [kv_seq_len]


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    block_q: int
    block_k_major: int
    block_k: int
    block_b: int

    block_q_major_dkv: int | None = None
    block_k_major_dkv: int | None = None
    block_k_dkv: int | None = None
    block_q_dkv: int | None = None

    block_k_major_dq: int | None = None
    block_k_dq: int | None = None
    block_q_dq: int | None = None

    def __post_init__(self):
        def verify_major_minor(prefix, suffix, major, minor):
            if minor > major:
                raise ValueError(
                    f"{prefix}{suffix}={minor} should be smaller than"
                    f" {prefix}_major{suffix}={major}"
                )
            if major % minor != 0:
                raise ValueError(
                    f"{prefix}{suffix}={minor} should divide"
                    f" {prefix}_major{suffix}={major}"
                )

        verify_major_minor("block_k", "", self.block_k_major, self.block_k)
        if self.block_q_major_dkv is not None and self.block_q_dkv is not None:
            verify_major_minor(
                "block_q", "_dkv", self.block_q_major_dkv, self.block_q_dkv
            )
        if self.block_k_major_dkv is not None and self.block_k_dkv is not None:
            verify_major_minor(
                "block_k", "_dkv", self.block_k_major_dkv, self.block_k_dkv
            )
        if self.block_k_major_dq is not None and self.block_k_dq is not None:
            verify_major_minor("block_k", "_dq", self.block_k_major_dq, self.block_k_dq)

    @property
    def has_backward_blocks(self) -> bool:
        backward_blocks = (
            self.block_q_major_dkv,
            self.block_k_major_dkv,
            self.block_q_dkv,
            self.block_k_dkv,
            self.block_k_major_dq,
            self.block_k_dq,
            self.block_q_dq,
        )
        return all(b is not None for b in backward_blocks)

    @classmethod
    def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model):
        del batch_size, num_heads, q_seq_len, kv_len, d_model  # Unused.
        return BlockSizes(
            block_q=128,
            block_k_major=128,
            block_k=128,
            block_b=1,
            block_q_major_dkv=128,
            block_k_major_dkv=128,
            block_k_dkv=128,
            block_q_dkv=128,
            block_k_major_dq=128,
            block_k_dq=128,
            block_q_dq=128,
        )


def _flash_attention(
    q,
    k,
    v,
    carry,
    q_chunk_idx_start,
    k_chunk_idx_start,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
    return _flash_attention_impl(
        q,
        k,
        v,
        carry,
        q_chunk_idx_start,
        k_chunk_idx_start,
        ab,
        segment_ids,
        save_residuals,
        causal,
        sm_scale,
        block_sizes.block_b,
        block_sizes.block_q,
        block_sizes.block_k_major,
        block_sizes.block_k,
        debug,
    )


def _flash_attention_fwd(
    q,
    k,
    v,
    carry,
    q_chunk_idx_start,
    k_chunk_idx_start,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported")
    o, l, m = _flash_attention(
        q,
        k,
        v,
        carry,
        q_chunk_idx_start,
        k_chunk_idx_start,
        ab,
        segment_ids,
        True,
        causal,
        sm_scale,
        block_sizes,
        debug,
    )
    return o, l, m


def _flash_attention_bwd(
    save_residuals: bool,
    causal: bool,
    sm_scale: float,
    block_sizes: BlockSizes,
    debug: bool,
    q_chunk_idx_start,
    k_chunk_idx_start,
    residuals,
    do,
):
    """VJP rule for FlashAttention."""
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported")
    (q, k, v, ab, segment_ids, o, l, m) = residuals
    if not block_sizes.has_backward_blocks:
        raise ValueError(
            "Program is being differentiated, but not all backward blocks are"
            " specified"
        )

    di = jnp.sum(
        o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1
    )  # [batch_size, num_heads, q_seq_len]

    dk, dv = _flash_attention_bwd_dkv(
        q_chunk_idx_start,
        k_chunk_idx_start,
        q,
        k,
        v,
        ab,
        segment_ids,
        l,
        m,
        do,
        di,
        block_q_major=block_sizes.block_q_major_dkv,
        block_k_major=block_sizes.block_k_major_dkv,
        block_k=block_sizes.block_k_dkv,
        block_q=block_sizes.block_q_dkv,
        sm_scale=sm_scale,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        debug=debug,
    )

    dq, ds = _flash_attention_bwd_dq(
        q_chunk_idx_start,
        k_chunk_idx_start,
        q,
        k,
        v,
        ab,
        segment_ids,
        l,
        m,
        do,
        di,
        block_q_major=block_sizes.block_q_dq,
        block_k_major=block_sizes.block_k_major_dq,
        block_k=block_sizes.block_k_dq,
        sm_scale=sm_scale,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        debug=debug,
    )
    return dq, dk, dv


MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
    # A block is considered below or on diagonal as long as the bottom left
    # corner of the block is below or on diagonal.
    return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)


def _flash_attention_kernel(
    q_idx_chunk_start, k_idx_chunk_start, q_tile_ref, *args, **kwargs
):
    block_b = q_tile_ref.shape[0]
    # If we're not going to tile the softmax, then we can avoid a bunch of VPU ops.
    if kwargs["block_k"] == kwargs["kv_seq_len"]:
        assert False
        kernel = _flash_attention_kernel_single_batch_single_step
    else:
        kernel = _flash_attention_kernel_single_batch
    for batch_idx in range(block_b):
        kernel(
            (batch_idx, 0),
            q_idx_chunk_start,
            k_idx_chunk_start,
            q_tile_ref,
            *args,
            **kwargs,
        )


def _flash_attention_kernel_single_batch(
    batch_idx: tuple[int, ...],
    q_chunk_idx_start_ref,
    k_chunk_idx_start_ref,
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    acc_tile_ref,
    l_tile_ref,
    m_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    l_ref: Any | None = None,
    m_ref: Any | None = None,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
    block_q,
):
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]
    head_dim = q_tile_ref.shape[-1]

    kv_seq_idx = pl.program_id(3)

    @pl.when(kv_seq_idx == 0)
    def start_new_sequence():
        m_scratch_ref[batch_idx] = m_tile_ref[batch_idx]
        l_scratch_ref[batch_idx] = l_tile_ref[batch_idx]
        acc_scratch_ref[batch_idx] = acc_tile_ref[batch_idx]

    q_chunk_idx_start = q_chunk_idx_start_ref[0]
    k_chunk_idx_start = k_chunk_idx_start_ref[0]

    q_seq_idx = pl.program_id(2)
    if causal:
        should_run = below_or_on_diag(
            q_seq_idx + q_chunk_idx_start,
            block_q,
            kv_seq_idx + k_chunk_idx_start,
            block_k_major,
        )
    else:
        should_run = True

    @pl.when(should_run)
    def run():
        @functools.partial(
            lax.fori_loop, 0, block_k_major // block_k, init_val=None, unroll=True
        )
        def body(i, _):
            m_prev = m_scratch_ref[batch_idx]
            l_prev = l_scratch_ref[batch_idx]
            q = q_tile_ref[batch_idx]  # [block_q, head_dim]
            start_k = i * block_k
            k = pl.load(
                k_tile_ref, (*batch_idx, pl.dslice(start_k, block_k), slice(None))
            )  # [block_k, head_dim]

            s = jax.lax.dot_general(
                q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
            )  # [block_q, block_k]

            # Add attention bias if needed.
            if ab_tile_ref is not None:
                ab = pl.load(
                    ab_tile_ref,
                    (batch_idx[0], pl.dslice(0, block_q), pl.dslice(start_k, block_k)),
                ).astype(jnp.float32)
                s += ab

            if sm_scale != 1.0:
                s *= sm_scale

            mask = None
            if q_segment_ids_tile_ref is not None:
                repeats, rem = divmod(block_k, NUM_LANES)
                if rem:
                    raise NotImplementedError(
                        f"kv block size must be a multiple of {NUM_LANES}"
                    )
                q_segment_ids = pltpu.repeat(
                    q_segment_ids_tile_ref[batch_idx[0]], repeats, axis=1
                )  # [block_q, block_k].
                kv_segment_ids = pl.load(
                    kv_segment_ids_tile_ref,
                    (batch_idx[0], pl.dslice(1), pl.dslice(start_k, block_k)),
                )  # [1, block_k].
                mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

            if causal:
                mask_shape = (block_q, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += (q_seq_idx + q_chunk_idx_start) * block_q
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += (kv_seq_idx + k_chunk_idx_start) * block_k_major + start_k
                causal_mask = col_ids <= row_ids
                mask = (
                    causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
                )

            s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

            m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [block_q, 1].
            m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].

            block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
            if rem:
                raise NotImplementedError(
                    f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
                )
            p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))

            alpha = jnp.exp(m_prev - m_next)  # Shape [block_q, 128].

            l_corr = alpha * l_prev

            l_next = jnp.sum(p, axis=1)[:, None] + l_corr  # Shape [block_q, 128]

            head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
            l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
            if rem:
                if head_dim_repeats == 0:
                    l_broadcast = lambda l: l[:, :head_dim]
                else:
                    raise NotImplementedError(
                        f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
                    )
            l_scratch_ref[batch_idx] = l_next
            m_scratch_ref[batch_idx] = m_next

            l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
            acc_scratch_ref[batch_idx] *= l_broadcast(l_corr * l_next_inv_safe)
            v = pl.load(
                v_tile_ref, (*batch_idx, pl.dslice(start_k, block_k), slice(None))
            )
            o_curr = jax.lax.dot(
                p.astype(v.dtype), v, preferred_element_type=jnp.float32
            )
            acc_scratch_ref[batch_idx] += o_curr * l_broadcast(l_next_inv_safe)

    @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def store_output():
        o_tile_ref[batch_idx] = acc_scratch_ref[batch_idx].astype(o_tile_ref.dtype)
        if l_ref is not None:
            l_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_ref.dtype)
        if m_ref is not None:
            m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)


def _flash_attention_impl(
    q,
    k,
    v,
    carry,
    q_chunk_idx_start,
    k_chunk_idx_start,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
):
    assert block_k_major == block_k, (block_k_major, block_k)
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    acc, l_prev, m_prev = carry
    l_prev, m_prev = map(
        lambda x: jnp.broadcast_to(x[..., None], (*x.shape, MIN_BLOCK_SIZE)),
        (l_prev, m_prev),
    )
    q_chunk_idx_start, k_chunk_idx_start = (
        q_chunk_idx_start[None],
        k_chunk_idx_start[None],
    )
    _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
    _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
    _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

    grid = (
        pl.cdiv(batch_size, block_b),
        num_heads,
        pl.cdiv(q_seq_len, block_q),
        kv_seq_len // block_k_major,
    )

    def q_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    def kv_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
    ):
        if causal:
            # If the kv block is skipped, prefetch the next valid kv block, i.e. the
            # 0th one to be used for the next block_q rows.
            next_kv_index = lax.select(
                below_or_on_diag(
                    q_seq_index + q_idx_ref[0],
                    block_q,
                    kv_seq_index + k_idx_ref[0],
                    block_k_major,
                ),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    def ab_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
    ):
        if causal:
            should_run = below_or_on_diag(
                q_seq_index + q_idx_ref[0],
                block_q,
                kv_seq_index + k_idx_ref[0],
                block_k_major,
            )
            next_kv_index = lax.select(should_run, kv_seq_index, 0)
        else:
            next_kv_index = kv_seq_index

        return (batch_index, 0, next_kv_index)

    def o_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    def lm_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    kernel = functools.partial(
        _flash_attention_kernel,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
        block_q=block_q,
    )
    out_shape = [jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)]
    out_specs = [pl.BlockSpec(o_index_map, (block_b, 1, block_q, head_dim))]

    if block_k != kv_seq_len:
        scratch_shape = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
        m_scratch = scratch_shape((block_b, 1, block_q, MIN_BLOCK_SIZE))
        l_scratch = scratch_shape((block_b, 1, block_q, MIN_BLOCK_SIZE))
        acc_scratch = scratch_shape((block_b, 1, block_q, head_dim))
        out_shape += [m_scratch, l_scratch, acc_scratch]
        out_specs += [
            pl.BlockSpec(lambda *_: (0, 0, 0, 0), m_scratch.shape),
            pl.BlockSpec(lambda *_: (0, 0, 0, 0), l_scratch.shape),
            pl.BlockSpec(lambda *_: (0, 0, 0, 0), acc_scratch.shape),
        ]
    else:
        assert False
        out_shape += [None, None, None]
        out_specs += [None, None, None]

    if save_residuals:
        out_specs = [
            *out_specs,
            pl.BlockSpec(lm_index_map, (block_b, 1, block_q, MIN_BLOCK_SIZE)),
            pl.BlockSpec(lm_index_map, (block_b, 1, block_q, MIN_BLOCK_SIZE)),
        ]
        l = jax.ShapeDtypeStruct(
            (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
        )
        m = jax.ShapeDtypeStruct(
            (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
        )
        out_shape = (*out_shape, l, m)

    ab_block_spec = (
        pl.BlockSpec(ab_index_map, (block_b, block_q, block_k_major))
        if ab is not None
        else None
    )

    if ab is not None:
        ab = ab[:, None].repeat(block_q, axis=1)

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(
            batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref
        ):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(
            batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
        ):
            del head_index
            if causal:
                next_kv_index = lax.select(
                    below_or_on_diag(
                        q_seq_index + q_idx_ref[0],
                        block_q,
                        kv_seq_index + k_idx_ref[0],
                        block_k_major,
                    ),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec(
            q_segment_ids_index_map, (block_b, block_q, NUM_LANES)
        )
        kv_segment_ids_spec = pl.BlockSpec(
            kv_segment_ids_index_map, (block_b, NUM_SUBLANES, block_k_major)
        )

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        pl.BlockSpec(q_index_map, (block_b, 1, block_q, head_dim)),
        pl.BlockSpec(kv_index_map, (block_b, 1, block_k_major, head_dim)),
        pl.BlockSpec(kv_index_map, (block_b, 1, block_k_major, head_dim)),
        pl.BlockSpec(q_index_map, (block_b, 1, block_q, head_dim)),
        pl.BlockSpec(lm_index_map, (block_b, 1, block_q, MIN_BLOCK_SIZE)),
        pl.BlockSpec(lm_index_map, (block_b, 1, block_q, MIN_BLOCK_SIZE)),
        ab_block_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
    ]

    o, *aux = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2, in_specs=in_specs, out_specs=out_specs, grid=grid
        ),
        debug=debug,
        mosaic_params=dict(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
    )(
        q_chunk_idx_start,
        k_chunk_idx_start,
        q,
        k,
        v,
        acc,
        l_prev,
        m_prev,
        ab,
        q_segment_ids,
        kv_segment_ids,
    )
    if save_residuals:
        l, m = (v[..., 0] for v in aux[-2:])
        return (o, l, m)
    else:
        return o


def _flash_attention_dkv_kernel(
    q_chunk_idx_start_ref,
    k_chunk_idx_start_ref,
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dk_tile_ref,
    dv_tile_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    q_seq_len: int,
    block_q: int,
    block_k: int,
):
    _, _, block_q_major, _ = q_tile_ref.shape
    _, _, block_k_major, _ = k_tile_ref.shape

    q_seq_index = pl.program_id(axis=3)
    kv_seq_index = pl.program_id(axis=2)

    q_chunk_idx_start = q_chunk_idx_start_ref[0]
    k_chunk_idx_start = k_chunk_idx_start_ref[0]

    @pl.when(q_seq_index == 0)
    def start_new_sequence():
        dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
        dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

    def q_body(j, _):
        start_q = j * block_q

        def k_body(i, _):
            start_k = i * block_k
            k = pl.load(k_tile_ref, (0, 0, pl.ds(start_k, block_k), slice(None)))
            v = pl.load(v_tile_ref, (0, 0, pl.ds(start_k, block_k), slice(None)))
            q = pl.load(
                q_tile_ref, (0, 0, pl.ds(start_q, block_q), slice(None))
            )  # [block_q, head_dim]
            l = pl.load(
                l_tile_ref, (0, 0, pl.ds(start_q, block_q), slice(None))
            )  # [block_q, 128]
            m = pl.load(
                m_tile_ref, (0, 0, pl.ds(start_q, block_q), slice(None))
            )  # [block_q, 128]
            do = pl.load(
                do_tile_ref, (0, 0, pl.ds(start_q, block_q), slice(None))
            )  # [block_q, 128]
            di = pl.load(
                di_tile_ref, (0, 0, pl.ds(start_q, block_q), slice(None))
            ).astype(
                jnp.float32
            )  # [block_q, 128]

            capped_logits = lax.dot_general(
                q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
            )  # [block_q_major, block_k]

            if ab_tile_ref is not None:
                ab = pl.load(
                    ab_tile_ref,
                    (
                        0,
                        pl.dslice(0, block_q),
                        pl.dslice(i * block_k, block_k),
                    ),
                ).astype(jnp.float32)
                capped_logits += ab

            if sm_scale != 1.0:
                capped_logits *= sm_scale

            mask = None
            if q_segment_ids_tile_ref is not None:
                repeats, rem = divmod(block_k, NUM_LANES)
                if rem:
                    raise NotImplementedError()
                q_segment_ids = pl.load(
                    q_segment_ids_tile_ref, (0, pl.ds(start_q, block_q), slice(None))
                )  # [block_q, NUM_LANES].
                q_segment_ids = pltpu.repeat(
                    q_segment_ids, repeats, axis=1
                )  # [block_q, block_k].
                kv_segment_ids = pl.load(
                    kv_segment_ids_tile_ref, (slice(None), 0, pl.ds(start_k, block_k))
                )  # [1, block_k].
                mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

            if causal:
                mask_shape = (block_q, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += (q_seq_index + q_chunk_idx_start) * block_q_major + start_q
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += (kv_seq_index + k_chunk_idx_start) * block_k_major + start_k
                causal_mask = col_ids <= row_ids
                mask = (
                    causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
                )

            capped_logits = (
                capped_logits
                if mask is None
                else capped_logits + jnp.where(mask, 0.0, mask_value)
            )

            p = jnp.exp(
                capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1)
            )
            p = p * pltpu.repeat(
                1 / l, block_k // MIN_BLOCK_SIZE, axis=1
            )  # [block_q_major, block_k_major]
            dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
            pl.store(
                dv_scratch_ref,
                (pl.ds(start_k, block_k), slice(None)),
                pl.load(dv_scratch_ref, (pl.ds(start_k, block_k), slice(None)))
                + dv.astype(dv_scratch_ref.dtype),
            )

            # di: [block_q, 128]
            # do: [block_q, head_dim]
            # v: [block_k_major, head_dim]
            dp = lax.dot_general(
                do, v, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
            )
            ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p

            if sm_scale != 1.0:
                ds = ds * sm_scale

            # ds: [block_q_major, block_k_major]
            # q: [block_q_major, head_dim]
            dk = lax.dot(ds.T.astype(do.dtype), q, preferred_element_type=jnp.float32)
            pl.store(
                dk_scratch_ref,
                (pl.ds(start_k, block_k), slice(None)),
                pl.load(dk_scratch_ref, (pl.ds(start_k, block_k), slice(None)))
                + dk.astype(dk_scratch_ref.dtype),
            )

        lax.fori_loop(0, block_k_major // block_k, k_body, None, unroll=True)

    if causal:
        should_run = below_or_on_diag(
            q_seq_index + q_chunk_idx_start,
            block_q_major,
            kv_seq_index + k_chunk_idx_start,
            block_k_major,
        )
    else:
        should_run = True

    @pl.when(should_run)
    def run():
        lax.fori_loop(0, block_q_major // block_q, q_body, None, unroll=True)

    @pl.when(q_seq_index == q_seq_len // block_q_major - 1)
    def end_of_q_sequence():
        dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref)
        dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref)


def _flash_attention_bwd_dkv(
    q_chunk_idx_start,
    k_chunk_idx_start,
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_q: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    debug: bool = False,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    q_chunk_idx_start, k_chunk_idx_start = (
        q_chunk_idx_start[None],
        k_chunk_idx_start[None],
    )
    _verify_block("block_q_major_dkv", "q_seq_len", block_q_major, q_seq_len)
    _verify_block("block_q_dkv", "q_seq_len", block_q, q_seq_len)
    _verify_block("block_k_major_dkv", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k_dkv", "kv_seq_len", block_k, kv_seq_len)

    # Broadcast out scalar values
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    # Preprocess contraction for bwd pass
    di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

    # kv index needs to be before q index since q index is the contractng
    # dimension.
    grid = (
        batch_size,
        num_heads,
        kv_seq_len // block_k_major,
        q_seq_len // block_q_major,
    )

    def qo_index_map(
        batch_index, head_index, kv_seq_index, q_seq_index, q_idx_ref, k_idx_ref
    ):
        if causal:
            # If the q block is skipped, stay at the 0th q block.
            next_q_index = lax.select(
                below_or_on_diag(
                    q_seq_index + q_idx_ref[0],
                    block_q_major,
                    kv_seq_index + k_idx_ref[0],
                    block_k_major,
                ),
                q_seq_index,
                0,
            )
        else:
            next_q_index = q_seq_index

        return (batch_index, head_index, next_q_index, 0)

    qo_spec = pl.BlockSpec(qo_index_map, (1, 1, block_q_major, head_dim))
    assert qo_spec.block_shape is not None
    assert q.ndim == len(qo_spec.block_shape)
    do_spec = qo_spec
    assert do.ndim == len(qo_spec.block_shape)

    def kv_index_map(batch_index, head_index, kv_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, kv_seq_index, 0)

    kv_spec = pl.BlockSpec(kv_index_map, (1, 1, block_k_major, head_dim))
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, _, q_seq_index, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec(lm_index_map, (1, 1, block_q_major, MIN_BLOCK_SIZE))
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    di_spec = pl.BlockSpec(qo_index_map, (1, 1, block_q_major, MIN_BLOCK_SIZE))
    assert di_spec.block_shape is not None
    assert di.ndim == len(di_spec.block_shape)

    def ab_index_map(
        batch_index, head_index, kv_seq_index, q_seq_index, q_idx_ref, k_idx_ref
    ):
        return (batch_index, 0, kv_seq_index)

    if ab is not None:
        ab = ab[:, None].repeat(block_q_major, axis=1)

    dab_spec = (
        pl.BlockSpec(ab_index_map, (1, block_q_major, block_k_major))
        if ab is not None
        else None
    )

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(
            batch_index, head_index, kv_seq_index, q_seq_index, q_idx_ref, k_idx_ref
        ):
            del head_index
            if causal:
                next_q_index = lax.select(
                    below_or_on_diag(
                        q_seq_index + q_idx_ref[0],
                        block_q_major,
                        kv_seq_index + k_idx_ref[0],
                        block_k_major,
                    ),
                    q_seq_index,
                    0,
                )
            else:
                next_q_index = q_seq_index
            return (batch_index, next_q_index, 0)

        def kv_segment_ids_index_map(
            batch_index, head_index, kv_seq_index, _, q_idx_ref, k_idx_ref
        ):
            del head_index
            return (batch_index, 0, kv_seq_index)

        q_segment_ids_spec = pl.BlockSpec(
            q_segment_ids_index_map, (1, block_q_major, NUM_LANES)
        )
        kv_segment_ids_spec = pl.BlockSpec(
            kv_segment_ids_index_map, (1, NUM_SUBLANES, block_k_major)
        )

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        qo_spec,
        kv_spec,
        kv_spec,
        dab_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
        lm_spec,
        lm_spec,
        do_spec,
        di_spec,
    ]

    out_shapes = [
        jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim), k.dtype),
        jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim), v.dtype),
        jax.ShapeDtypeStruct((block_k_major, head_dim), jnp.float32),
        jax.ShapeDtypeStruct((block_k_major, head_dim), jnp.float32),
    ]

    def dkv_index_map(batch_index, head_index, kv_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, kv_seq_index, 0)

    dkv_spec = pl.BlockSpec(dkv_index_map, (1, 1, block_k_major, head_dim))
    out_specs = [
        dkv_spec,
        dkv_spec,
        pl.BlockSpec(lambda *_: (0, 0), (block_k_major, head_dim)),
        pl.BlockSpec(lambda *_: (0, 0), (block_k_major, head_dim)),
    ]

    kernel = functools.partial(
        _flash_attention_dkv_kernel,
        block_q=block_q,
        block_k=block_k,
        sm_scale=sm_scale,
        causal=causal,
        mask_value=mask_value,
        q_seq_len=q_seq_len,
    )
    name_scope = (
        f"flash_mha_bwd_dkv_{block_q_major=}_{block_q=}_{block_k_major=}_{block_k=}"
    )
    with jax.named_scope(name_scope):
        dk, dv, _, _ = pl.pallas_call(
            kernel,
            out_shape=out_shapes,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2, in_specs=in_specs, out_specs=out_specs, grid=grid
            ),
            debug=debug,
            mosaic_params=dict(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
            ),
        )(
            q_chunk_idx_start,
            k_chunk_idx_start,
            q,
            k,
            v,
            ab,
            q_segment_ids,
            kv_segment_ids,
            l,
            m,
            do,
            di,
        )
        assert dk.shape == k.shape
        assert dv.shape == v.shape
    return dk, dv


def _flash_attention_dq_kernel(
    q_chunk_idx_start_ref,
    k_chunk_idx_start_ref,
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dq_tile_ref,
    dq_scratch_ref,
    ds_tile_ref,
    *,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    kv_seq_len: int,
    block_k: int,
):
    _, _, block_k_major, _ = k_tile_ref.shape
    _, _, block_q_major, _ = q_tile_ref.shape

    kv_seq_index = pl.program_id(axis=3)
    q_seq_index = pl.program_id(axis=2)

    q_chunk_idx_start = q_chunk_idx_start_ref[0]
    k_chunk_idx_start = k_chunk_idx_start_ref[0]

    @pl.when(kv_seq_index == 0)
    def start_new_sequence():
        dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)

    def body(i, _):
        k_slice = pl.ds(i * block_k, block_k)
        q = q_tile_ref[0, 0, :, :]
        k = pl.load(
            k_tile_ref,
            (0, 0, k_slice, slice(None)),
        )  # [block_k, head_dim]
        v = pl.load(
            v_tile_ref,
            (0, 0, k_slice, slice(None)),
        )  # [block_k, head_dim]
        l = l_tile_ref[0, 0, :, :]  # [block_q_major, 128]
        m = m_tile_ref[0, 0, :, :]  # [block_q_major, 128]
        do = do_tile_ref[0, 0, :, :]  # [block_q_major, head_dim]
        di = di_tile_ref[0, 0, :].astype(jnp.float32)  # [block_q_major, 128]

        capped_logits = jax.lax.dot_general(
            q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
        )

        if ab_tile_ref is not None:
            ab = pl.load(
                ab_tile_ref,
                (0, pl.dslice(0, block_q_major), pl.dslice(i * block_k, block_k)),
            ).astype(jnp.float32)
            capped_logits += ab

        if sm_scale != 1.0:
            capped_logits *= sm_scale

        mask = None
        if q_segment_ids_tile_ref is not None:
            repeats, rem = divmod(block_k, NUM_LANES)
            if rem:
                raise NotImplementedError(
                    f"kv block size must be a multiple of {NUM_LANES}"
                )
            q_segment_ids = pltpu.repeat(
                q_segment_ids_tile_ref[0], repeats, axis=1
            )  # [block_q, block_k].
            kv_segment_ids = pl.load(
                kv_segment_ids_tile_ref, (slice(None), 0, k_slice)
            )  # [1, block_k].
            mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

        if causal:
            mask_shape = (block_q_major, block_k)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids += (q_seq_index + q_chunk_idx_start) * block_q_major
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids += (kv_seq_index + k_chunk_idx_start) * block_k_major + i * block_k
            causal_mask = col_ids <= row_ids
            mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        capped_logits = (
            capped_logits
            if mask is None
            else capped_logits + jnp.where(mask, 0.0, mask_value)
        )

        p = jnp.exp(capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1))
        p = p * pltpu.repeat(
            1 / l, block_k // MIN_BLOCK_SIZE, axis=1
        )  # [block_q_major, block_k]

        # di: [block_q_major, 128]
        # do: [block_q_major, head_dim]
        # v: [block_k_major, head_dim]
        dp = jax.lax.dot_general(
            do,
            v,
            TRANS_B_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p

        if sm_scale != 1.0:
            ds = ds * sm_scale

        if ds_tile_ref is not None:
            pl.store(
                ds_tile_ref,
                (0, pl.dslice(None), pl.dslice(i * block_k, block_k)),
                ds.astype(ds_tile_ref.dtype),
            )

        # dp: [block_q_major, block_k]
        # k: [block_k, head_dim]
        dq_scratch_ref[:, :] += lax.dot(
            ds.astype(k.dtype),
            k,
            preferred_element_type=jnp.float32,
        ).astype(dq_scratch_ref.dtype)

    if causal:
        should_run = below_or_on_diag(
            q_seq_index + q_chunk_idx_start,
            block_q_major,
            kv_seq_index + k_chunk_idx_start,
            block_k_major,
        )
        should_not_run = lax.select(should_run, False, True)
    else:
        should_run = True
        should_not_run = False  # type: ignore

    @pl.when(should_run)
    def run():
        lax.fori_loop(0, block_k_major // block_k, body, None, unroll=True)

    @pl.when(should_not_run)
    def zero_out_ds():
        if ds_tile_ref is not None:
            ds_tile_ref[...] = jnp.zeros_like(ds_tile_ref)

    @pl.when(kv_seq_index == kv_seq_len // block_k_major - 1)
    def end_of_kv_sequence():
        dq_tile_ref[0, 0, :, :] = dq_scratch_ref[...].astype(dq_tile_ref)
        dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _flash_attention_bwd_dq(
    q_chunk_idx_start,
    k_chunk_idx_start,
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    debug: bool,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    q_chunk_idx_start, k_chunk_idx_start = (
        q_chunk_idx_start[None],
        k_chunk_idx_start[None],
    )
    _verify_block("block_q_dq", "q_seq_len", block_q_major, q_seq_len)
    _verify_block("block_k_major_dq", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k_dq", "block_k", block_k, kv_seq_len)

    # Broadcast out scalar values
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    # Preprocess contraction for bwd pass
    di = jnp.broadcast_to(di[..., None], (*di.shape, block_k_major))

    grid = (
        batch_size,
        num_heads,
        q_seq_len // block_q_major,
        kv_seq_len // block_k_major,
    )

    def qo_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    qo_spec = pl.BlockSpec(qo_index_map, (1, 1, block_q_major, head_dim))
    do_spec = qo_spec

    def kv_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
    ):
        if causal:
            # If the kv block is skipped, prefetch the next valid kv block, i.e. the
            # 0th one to be used for the next block_q rows.
            next_kv_index = lax.select(
                below_or_on_diag(
                    q_seq_index + q_idx_ref[0],
                    block_q_major,
                    kv_seq_index + k_idx_ref[0],
                    block_k_major,
                ),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    kv_spec = pl.BlockSpec(kv_index_map, (1, 1, block_k_major, head_dim))
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec(lm_index_map, (1, 1, block_q_major, MIN_BLOCK_SIZE))
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    di_spec = pl.BlockSpec(qo_index_map, (1, 1, block_q_major, MIN_BLOCK_SIZE))
    assert di_spec.block_shape is not None
    assert di.ndim == len(di_spec.block_shape)

    def ab_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
    ):
        return (batch_index, 0, kv_seq_index)

    if ab is not None:
        ab = ab[:, None].repeat(block_q_major, axis=1)

    dab_spec = (
        pl.BlockSpec(ab_index_map, (1, block_q_major, block_k_major))
        if ab is not None
        else None
    )

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(
            batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref
        ):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(
            batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
        ):
            del head_index
            if causal:
                # If the kv block is skipped, prefetch the next valid kv block, i.e. the
                # 0th one to be used for the next block_q rows.
                next_kv_index = lax.select(
                    below_or_on_diag(
                        q_seq_index + q_idx_ref[0],
                        block_q_major,
                        kv_seq_index + k_idx_ref[0],
                        block_k_major,
                    ),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec(
            q_segment_ids_index_map, (1, block_q_major, NUM_LANES)
        )
        kv_segment_ids_spec = pl.BlockSpec(
            kv_segment_ids_index_map, (1, NUM_SUBLANES, block_k_major)
        )

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        qo_spec,
        kv_spec,
        kv_spec,
        dab_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
        lm_spec,
        lm_spec,
        do_spec,
        di_spec,
    ]

    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct((block_q_major, head_dim), jnp.float32),
        jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
    ]
    dq_spec = pl.BlockSpec(qo_index_map, (1, 1, block_q_major, head_dim))
    out_specs = [
        dq_spec,
        pl.BlockSpec(lambda *_: (0, 0), (block_q_major, head_dim)),
        dab_spec,
    ]

    kernel = functools.partial(
        _flash_attention_dq_kernel,
        sm_scale=sm_scale,
        causal=causal,
        mask_value=mask_value,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    name_scope = f"flash_mha_bwd_dq_{block_q_major=}_{block_k_major=}_{block_k=}"
    with jax.named_scope(name_scope):
        dq, _, ds = pl.pallas_call(
            kernel,
            out_shape=out_shapes,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2, in_specs=in_specs, out_specs=out_specs, grid=grid
            ),
            debug=debug,
            mosaic_params=dict(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
            ),
        )(
            q_chunk_idx_start,
            k_chunk_idx_start,
            q,
            k,
            v,
            ab,
            q_segment_ids,
            kv_segment_ids,
            l,
            m,
            do,
            di,
        )

    return dq, ds


def _verify_block(block_name, dim_name, block, dim, should_divide=True):
    if block > dim:
        raise ValueError(
            f"{block_name}={block} should be smaller or equal to {dim_name}={dim}"
        )
    if should_divide and dim % block != 0:
        raise ValueError(
            f"{dim_name}={dim} should be divisible by {block_name}={block}"
        )
