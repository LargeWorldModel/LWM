import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import tempfile
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
from jax.experimental.shard_map import shard_map
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning as nn_partitioning

import sentencepiece as spm
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

from ml_collections import ConfigDict
from tux import function_args_to_config, load_pickle, open_file,  with_sharding_constraint, get_jax_mesh
from ringattention import ringattention, blockwise_feedforward, ringattention_jax, ringattention_inference


LLAMA_STANDARD_CONFIGS = {
    '200m': {
        'vocab_size': 32000,
        'hidden_size': 1024,
        'intermediate_size': 2048,
        'num_hidden_layers': 14,
        'num_attention_heads': 8,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '1b': {
        'vocab_size': 32000,
        'hidden_size': 2048,
        'intermediate_size': 5504,
        'num_hidden_layers': 22,
        'num_attention_heads': 16,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '3b': {
        'vocab_size': 32000,
        'hidden_size': 3200,
        'intermediate_size': 8640,
        'num_hidden_layers': 26,
        'num_attention_heads': 32,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '7b': {
        'vocab_size': 32000,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'max_sequence_length': 4096,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '13b': {
        'vocab_size': 32000,
        'hidden_size': 5120,
        'intermediate_size': 13824,
        'num_hidden_layers': 40,
        'num_attention_heads': 40,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '30b': {
        'vocab_size': 32000,
        'hidden_size': 6656,
        'intermediate_size': 17920,
        'num_hidden_layers': 60,
        'num_attention_heads': 52,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '65b': {
        'vocab_size': 32000,
        'hidden_size': 8192,
        'intermediate_size': 22016,
        'num_hidden_layers': 80,
        'num_attention_heads': 64,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    'debug': { # A small model for debugging
        'vocab_size': 32000,
        'hidden_size': 256,
        'intermediate_size': 256,
        'num_hidden_layers': 2,
        'num_attention_heads': 2,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
}


class LLaMAConfig(PretrainedConfig):
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_sequence_length=4096,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=1,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=False,
        scan_attention=True,
        scan_mlp=True,
        scan_query_chunk_size=1024,
        scan_key_chunk_size=1024,
        scan_mlp_chunk_size=1024,
        scan_layers=True,
        param_scan_axis=0,
        mesh_dim=None,
        theta=10000,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.scan_attention = scan_attention
        self.scan_mlp = scan_mlp
        self.scan_query_chunk_size = scan_query_chunk_size
        self.scan_key_chunk_size = scan_key_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.scan_layers = scan_layers
        self.param_scan_axis = param_scan_axis
        self.mesh_dim = mesh_dim
        self.theta = theta
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def get_default_config(cls, updates=None):
        config = function_args_to_config(cls.__init__)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    @staticmethod
    def get_jax_mesh(axis_dims):
        return get_jax_mesh(axis_dims, ('dp', 'fsdp', 'tp', 'sp'))

    @staticmethod
    def get_ranks_and_size(mesh):
        out = dict(mesh=mesh)
        mp_size = mesh.shape['tp'] * mesh.shape['sp']
        mp_node_size = max(1, mp_size // jax.local_device_count())
        dp_node_size = jax.process_count() // mp_node_size
        out.update(mp_node_size=mp_node_size,
                   dp_node_size=dp_node_size)

        dp_node_rank = jax.process_index() // mp_node_size
        mp_node_rank = jax.process_index() % mp_node_size
        out.update(dp_node_rank=dp_node_rank,
                   mp_node_rank=mp_node_rank)
        return out


    @staticmethod
    def get_partition_rules(scan_layers=False, scan_axis=0):
        """Parition rules are orderd, so that the beginning rules match first."""
        if scan_layers:
            if scan_axis == 0:
                return (
                    # embeddings
                    ("transformer/wte/embedding", PS("tp", ("fsdp", "sp"))),
                    # atention
                    ("attention/(wq|wk|wv)/kernel", PS(None, ("fsdp", "sp"), "tp")),
                    ("attention/wo/kernel", PS(None, "tp", ("fsdp", "sp"))),
                    # mlp
                    ("feed_forward/w1/kernel", PS(None, ("fsdp", "sp"), "tp")),
                    ("feed_forward/w2/kernel", PS(None, "tp", ("fsdp", "sp"))),
                    ("feed_forward/w3/kernel", PS(None, ("fsdp", "sp"), "tp")),
                    # layer norms
                    ("attention_norm/kernel", PS(None, None)),
                    ("ffn_norm/kernel", PS(None, None)),
                    # output head
                    ("transformer/ln_f/kernel", PS(None)),
                    ("lm_head/kernel", PS(("fsdp", "sp"), "tp")),
                    ('.*', PS(None)),
                )
            elif scan_axis == 1:
                return (
                    # embeddings
                    ("transformer/wte/embedding", PS("tp", ("fsdp", "sp"))),
                    # atention
                    ("attention/(wq|wk|wv)/kernel", PS(("fsdp", "sp"), None, "tp")),
                    ("attention/wo/kernel", PS("tp", None, ("fsdp", "sp"))),
                    # mlp
                    ("feed_forward/w1/kernel", PS(("fsdp", "sp"), None, "tp")),
                    ("feed_forward/w2/kernel", PS("tp", None, ("fsdp", "sp"))),
                    ("feed_forward/w3/kernel", PS(("fsdp", "sp"), None, "tp")),
                    # layer norms
                    ("attention_norm/kernel", PS(None, None)),
                    ("ffn_norm/kernel", PS(None, None)),
                    # output head
                    ("transformer/ln_f/kernel", PS(None)),
                    ("lm_head/kernel", PS(("fsdp", "sp"), "tp")),
                    ('.*', PS(None)),
                )
            else:
                raise ValueError(f"Invalid scan_axis {scan_axis}")
        else:
            return (
                # embeddings
                ("transformer/wte/embedding", PS("tp", ("fsdp", "sp"))),
                # atention
                ("attention/(wq|wk|wv)/kernel", PS(("fsdp", "sp"), "tp")),
                ("attention/wo/kernel", PS("tp", ("fsdp", "sp"))),
                # mlp
                ("feed_forward/w1/kernel", PS(("fsdp", "sp"), "tp")),
                ("feed_forward/w2/kernel", PS("tp", ("fsdp", "sp"))),
                ("feed_forward/w3/kernel", PS(("fsdp", "sp"), "tp")),
                # layer norms
                ("attention_norm/kernel", PS(None)),
                ("ffn_norm/kernel", PS(None)),
                # output head
                ("transformer/ln_f/kernel", PS(None)),
                ("lm_head/kernel", PS(("fsdp", "sp"), "tp")),
                ('.*', PS(None)),
            )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def get_frozen_param_exclusions(freeze_base):
        if freeze_base:
            return ("vte", "vision_head")
        else:
            return tuple()

    @staticmethod
    def rng_keys():
        return ('params', 'dropout')

    @classmethod
    def load_config(cls, path):
        if path in LLAMA_STANDARD_CONFIGS:
            return cls.from_dict(LLAMA_STANDARD_CONFIGS[path])
        load_type, load_path = path.split('::', 1)
        if load_type == 'pickle':
            return cls.from_dict(load_pickle(load_path)['llama_config'])
        elif load_type == 'json':
            with open_file(load_path, 'r') as fin:
                raw_config = fin.read()
            return cls.from_dict(json.loads(raw_config))
        else:
            raise ValueError(f'Unsupported load config type: {load_type}')


remat = nn_partitioning.remat

logger = logging.get_logger(__name__)


class RMSNorm(nn.Module):
    dim: int
    eps: float=1e-6
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


def precompute_freqs_cis(dim: int, max_position_embedding: int, theta: float=10000.0, dtype: jnp.dtype=jnp.float32) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(max_position_embedding) # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype=jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)


class FlaxLLaMAAttention(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.wq = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wk = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wv = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wo = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool")

        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            config.max_sequence_length,
            theta=config.theta,
            dtype=self.dtype,
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            if query.shape[1] == 1:
                mesh = LLaMAConfig.get_jax_mesh(self.config.mesh_dim)
                def fn(cached_key, cached_value, key, value, cur_index):
                    assert key.shape[1] == 1 and value.shape[1] == 1, (key.shape, value.shape)
                    sp_size = max_length // mesh.shape['sp']
                    axis_index = jax.lax.axis_index('sp')
                    cur_index = cur_index - axis_index * sp_size
                    key, value = jax.lax.cond(
                        jnp.logical_and(cur_index >= 0, cur_index < sp_size),
                        lambda: (
                            cached_key.at[:, cur_index].set(key[:, -1]),
                            cached_value.at[:, cur_index].set(value[:, -1]),
                        ),
                        lambda: (cached_key, cached_value),
                    )
                    return key, value
                fn = shard_map(
                    fn, mesh=mesh,
                    in_specs=(
                        PS(('dp', 'fsdp'), 'sp', 'tp', None),
                        PS(('dp', 'fsdp'), 'sp', 'tp', None),
                        PS(('dp', 'fsdp'), None, 'tp', None),
                        PS(('dp', 'fsdp'), None, 'tp', None),
                        PS()
                    ),
                    out_specs=(
                        PS(('dp', 'fsdp'), 'sp', 'tp', None),
                        PS(('dp', 'fsdp'), 'sp', 'tp', None)
                    ),
                    check_rep=False
                )
                key, value = fn(cached_key.value, cached_value.value, key, value, cur_index)
            else:
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        segment_ids,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

        if xq.shape[1] == 1:
            xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "tp"))
        else:
            xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), "sp", "tp"))
        xk = with_sharding_constraint(xk, PS(("dp", "fsdp"), "sp", "tp"))
        xv = with_sharding_constraint(xv, PS(("dp", "fsdp"), "sp", "tp"))

        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)

        freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype)

        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        if self.config.scan_attention and xq.shape[1] > max(self.config.scan_query_chunk_size, self.config.scan_key_chunk_size):
            # attention mask without nxn materlization, blockwise_attn will handle the rest
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

            if self.has_variable("cache", "cached_key") or init_cache:
                xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)

            # transform boolean mask into float mask
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
            attn_weights = None
            ring_attention_sharded = shard_map(
                partial(
                    ringattention,
                    axis_name="sp",
                    float32_logits=True,
                    cache_idx=None,
                    blockwise_kwargs=dict(
                        causal_block_size=1,
                        deterministic=deterministic,
                        dropout_rng=dropout_rng,
                        attn_pdrop=self.config.attn_pdrop,
                        query_chunk_size=self.config.scan_query_chunk_size,
                        key_chunk_size=self.config.scan_key_chunk_size,
                        dtype=self.dtype,
                        policy=jax.checkpoint_policies.nothing_saveable,
                        precision=self.precision,
                        prevent_cse=not self.config.scan_layers,
                    )
                ),
                mesh=LLaMAConfig.get_jax_mesh(self.config.mesh_dim),
                in_specs=(
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), None, None, None),
                    PS(("dp", "fsdp"), None),
                ),
                out_specs=PS(("dp", "fsdp"), "sp", "tp", None),
                check_rep=False
            )
            attn_output = ring_attention_sharded(xq, xk, xv, attention_bias, segment_ids)
            attn_output = with_sharding_constraint(attn_output, PS(("dp", "fsdp"), "sp", "tp", None))
        else:
            query_length, key_length = xq.shape[1], xk.shape[1]

            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = jnp.arange(max_decoder_length)[None] <= (jnp.arange(query_length) + mask_shift)[:, None]
                causal_mask = causal_mask[None, None]
                segment_mask = None
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
                if segment_ids is not None:
                    segment_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
                    segment_mask = segment_mask[:, None]
                else:
                    segment_mask = None

            batch_size = hidden_states.shape[0]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask, segment_mask)

            # During fast autoregressive decoding, we feed one position at a time,
            # and cache the keys and values step by step.
            if self.has_variable("cache", "cached_key") or init_cache:
                xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)

            q_sp_dim = None if xq.shape[1] == 1 else 'sp'
            attn_weights = None
            ring_attention_sharded = shard_map(
                partial(ringattention_inference, axis_name="sp"), mesh=LLaMAConfig.get_jax_mesh(self.config.mesh_dim),
                in_specs=(
                    PS(("dp", "fsdp"), q_sp_dim, "tp", None),
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), None, q_sp_dim, None)
                ),
                out_specs=PS(("dp", "fsdp"), q_sp_dim, "tp", None),
                check_rep=False
            )
            attn_output = ring_attention_sharded(
                xq, xk, xv, attention_mask
            )

        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxLLaMAMLP(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        config = self.config

        self.w1 = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.w2 = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.w3 = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.w2(nn.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x, deterministic=deterministic)
        return x


class FlaxLLaMABlock(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        attention_module = FlaxLLaMAAttention
        mlp_module = FlaxLLaMAMLP
        if self.config.scan_mlp:
            mlp_module = remat(
                mlp_module, static_argnums=(1,),
                policy=jax.checkpoint_policies.nothing_saveable,
                prevent_cse=not self.config.scan_layers,
            )
        self.attention = attention_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.feed_forward = mlp_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.attention_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.ffn_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        segment_ids=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        attn_outputs = self.attention(
            self.attention_norm(hidden_states),
            attention_mask,
            segment_ids,
            position_ids,
            deterministic,
            init_cache,
            output_attentions,
        )
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        feed_forward_input = self.ffn_norm(hidden_states)

        if self.config.scan_mlp and hidden_states.shape[1] >= self.config.scan_mlp_chunk_size:
            feed_forward_hidden_states = blockwise_feedforward(
                self.feed_forward,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
                pre_remat=True,
            )
        else:
            feed_forward_hidden_states = self.feed_forward(feed_forward_input, deterministic)
        feed_forward_hidden_states = with_sharding_constraint(feed_forward_hidden_states, PS(("dp", "fsdp"), None, "tp"))

        hidden_states = hidden_states + feed_forward_hidden_states

        outputs = hidden_states
        if self.config.scan_layers:
            outputs = (outputs, None)
        return outputs


class FlaxLLaMAPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LLaMAConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: LLaMAConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        segment_ids = None
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                segment_ids,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, segment_ids, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        segment_ids = None
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, segment_ids, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    @add_start_docstrings_to_model_forward("")
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        segment_ids=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            segment_ids,
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxLLaMABlockCollection(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        segment_ids=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        block = FlaxLLaMABlock
        if self.config.scan_layers:
            initializing = self.is_mutable_collection('params')
            params_spec = (
                self.config.param_scan_axis if initializing else
                nn_partitioning.ScanIn(self.config.param_scan_axis))
            cache_spec = 0
            hidden_states, _ = nn.scan(
                block,
                variable_axes={
                    'params': params_spec,
                    'cache': cache_spec,
                    'intermediates': 0
                },
                split_rngs={
                    'params': True,
                    'dropout': True
                },
                in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
                length=self.config.num_hidden_layers,
                metadata_params={nn.PARTITION_NAME: 'scan_decoder_layer'},
                )(self.config, name='scan_decoder', dtype=self.dtype, param_dtype=self.param_dtype,)(
                    hidden_states,
                    attention_mask,
                    segment_ids,
                    position_ids,
                    deterministic,
                    init_cache,
                    output_attentions,
                )
        else:
            blocks = [
                block(
                    self.config,
                    name=str(i),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ) for i in range(self.config.num_hidden_layers)
            ]
            for block in blocks:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = block(
                    hidden_states,
                    attention_mask,
                    segment_ids,
                    position_ids,
                    deterministic,
                    init_cache,
                    output_attentions,
                )
                hidden_states = layer_outputs

                if output_attentions:
                    all_attentions += (layer_outputs[1],)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxLLaMAModule(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxLLaMABlockCollection(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.ln_f = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        segment_ids,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.wte(input_ids.astype("i4"))
        assert input_embeds.shape[1] <= self.config.max_sequence_length, f"Input sequence length {input_embeds.shape[1]} larger than max supported sequence length {self.config.max_sequence_length}"

        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        outputs = self.h(
            hidden_states,
            attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )

class FlaxLLaMAForCausalLMModule(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.transformer = FlaxLLaMAModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        segment_ids=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.arange(seq_length, dtype=jnp.int32)[None].repeat(batch_size, axis=0)
        outputs = self.transformer(
            input_ids,
            attention_mask,
            segment_ids,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


@add_start_docstrings("", "")
class FlaxLLaMAForCausalLM(FlaxLLaMAPreTrainedModel):
    module_class = FlaxLLaMAForCausalLMModule

    def prepare_inputs_for_generation(
        self, input_ids, max_length,
        attention_mask: Optional[jax.Array] = None,
    ):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
