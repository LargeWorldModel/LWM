from typing import Optional
from functools import cached_property, partial
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import jax_utils
from transformers.configuration_utils import PretrainedConfig
from ml_collections import ConfigDict
from tux import function_args_to_config, open_file


class VQGAN:
    def __init__(self, vqgan_checkpoint, replicate=False):
        assert vqgan_checkpoint != ''
        self.replicate = replicate
        self.config = VQGANConfig.get_default_config()
        self.params = pickle.load(open_file(vqgan_checkpoint, 'rb'))
        if replicate:
            self.params = jax_utils.replicate(self.params)
        else:
            self.params = jax.jit(lambda x: x)(self.params)
        self.model = VQGANModel(self.config)

    def _wrap_fn(self, fn):
        if self.replicate:
            return jax.pmap(fn, devices=jax.local_devices())
        else:
            return jax.jit(fn)
    
    @cached_property
    def _encode(self):
        def fn(pixel_values, params):
            return self.model.apply(
                {'params': params}, 
                pixel_values,
                method=self.model.encode
            )
        return partial(self._wrap_fn(fn), params=self.params)
    
    @cached_property
    def _decode(self):
        def fn(encoding, params):
            return self.model.apply(
                {'params': params},
                encoding,
                method=self.model.decode
            )
        return partial(self._wrap_fn(fn), params=self.params)
    
    def encode(self, pixel_values):
        return self._encode(pixel_values)
    
    def decode(self, encoding):
        return self._decode(encoding)
    

class VQGANConfig(PretrainedConfig):
    model_type = "vqgan"

    def __init__(
        self,
        resolution=256,
        num_channels=3,
        hidden_channels=128,
        channel_mult=(1, 2, 2, 4, 6),
        num_res_blocks=2,
        attn_resolutions=(),
        no_attn_mid_block=True,
        z_channels=64,
        num_embeddings=8192,
        quantized_embed_dim=64,
        dropout=0.0,
        resample_with_conv=True,
        commitment_cost=0.25
    ):
        self.resolution = resolution
        self.num_channels = num_channels
        self.hidden_channels = hidden_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.no_attn_mid_block = no_attn_mid_block
        self.z_channels = z_channels
        self.num_embeddings = num_embeddings
        self.quantized_embed_dim = quantized_embed_dim
        self.dropout = dropout
        self.resample_with_conv = resample_with_conv
        self.commitment_cost = commitment_cost
    
    @classmethod
    def get_default_config(cls, updates=None):
        config = function_args_to_config(cls.__init__)
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        config.num_resolutions = len(config.channel_mult)
        return config
    
    @classmethod
    def load_config(cls, path):
        return cls.get_default_config(cls)

        
class VQGANModel(nn.Module):
    config: VQGANConfig

    def setup(self):
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.quantize = VectorQuantizer(
            self.config.num_embeddings, self.config.quantized_embed_dim
        )
        self.quant_conv = nn.Conv(self.config.quantized_embed_dim, [1, 1])
        self.post_quant_conv = nn.Conv(self.config.z_channels, [1, 1])
    
    def encode(self, pixel_values):
        T = None
        if len(pixel_values.shape) == 5: # video
            T = pixel_values.shape[1]
            pixel_values = pixel_values.reshape(-1, *pixel_values.shape[2:])
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        quantized_states, codebook_indices = self.quantize(hidden_states)
        if T is not None:
            quantized_states = quantized_states.reshape(-1, T, *quantized_states.shape[1:])
            codebook_indices = codebook_indices.reshape(-1, T, *codebook_indices.shape[1:])
        return quantized_states, codebook_indices

    def decode(self, encoding, is_codebook_indices=True):
        if is_codebook_indices:
            encoding = self.quantize(None, encoding)
        T = None
        if len(encoding.shape) == 5:
            T = encoding.shape[1]
            encoding = encoding.reshape(-1, *encoding.shape[2:])
        hidden_states = self.post_quant_conv(encoding)
        reconstructed_pixel_values = self.decoder(hidden_states)
        if T is not None:
            reconstructed_pixel_values = reconstructed_pixel_values.reshape(-1, T, *reconstructed_pixel_values.shape[1:])
        return jnp.clip(reconstructed_pixel_values, -1, 1)
    
    def __call__(self, pixel_values):
        encoding = self.encode(pixel_values)[1]
        recon = self.decode(encoding)
        return recon
    

class Encoder(nn.Module):
    config: VQGANConfig
    
    @nn.compact
    def __call__(self, pixel_values):
        assert pixel_values.shape[1] == pixel_values.shape[2] == self.config.resolution, pixel_values.shape
        hidden_states = nn.Conv(self.config.hidden_channels, [3, 3])(pixel_values)
        for i_level in range(self.config.num_resolutions):
            hidden_states = DownsamplingBlock(self.config, i_level)(hidden_states)
        hidden_states = MidBlock(
            self.config, self.config.no_attn_mid_block, self.config.dropout
        )(hidden_states)
        hidden_states = nn.GroupNorm()(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = nn.Conv(self.config.z_channels, [3, 3])(hidden_states)
        return hidden_states

        
class Decoder(nn.Module):
    config: VQGANConfig

    @nn.compact
    def __call__(self, hidden_states):
        hidden_states = nn.Conv(
            self.config.hidden_channels * self.config.channel_mult[self.config.num_resolutions - 1],
            [3, 3]
        )(hidden_states)
        hidden_states = MidBlock(
            self.config, self.config.no_attn_mid_block, self.config.dropout
        )(hidden_states)
        for i_level in reversed(range(self.config.num_resolutions)):
            hidden_states = UpsamplingBlock(self.config, i_level)(hidden_states)
        hidden_states = nn.GroupNorm()(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = nn.Conv(self.config.num_channels, [3, 3])(hidden_states)
        return hidden_states


class VectorQuantizer(nn.Module):
    n_e: int
    e_dim: int

    @nn.compact
    def __call__(self, z, encoding_indices=None):
        def quantize(encoding_indices):
            w = jax.device_put(embeddings)
            return w[(encoding_indices,)]
        embeddings = self.param(
            'embeddings',
            lambda rng, shape, dtype: jax.random.uniform(
                rng, shape, dtype, minval=-1.0 / self.n_e, maxval=1.0 / self.n_e
            ),
            [self.n_e, self.e_dim], jnp.float32
        )
        
        if encoding_indices is not None:
            return quantize(encoding_indices)

        z_flattened = z.reshape(-1, z.shape[-1])
        d = jnp.sum(z_flattened ** 2, axis=1, keepdims=True) + \
            jnp.sum(embeddings.T ** 2, axis=0, keepdims=True) - \
            2 * jnp.einsum('bd,nd->bn', z_flattened, embeddings)
        
        min_encoding_indices = jnp.argmin(d, axis=1)
        z_q = quantize(min_encoding_indices)
        z_q = jnp.reshape(z_q, z.shape)
        z_q = z + jax.lax.stop_gradient(z_q - z)

        encodings_one_hot = jax.nn.one_hot(min_encoding_indices, num_classes=self.n_e)
        assert len(encodings_one_hot.shape) == 2
        min_encoding_indices = jnp.reshape(min_encoding_indices, z.shape[:-1])

        return z_q, min_encoding_indices


class DownsamplingBlock(nn.Module):
    config: VQGANConfig
    block_idx: int

    @nn.compact
    def __call__(self, hidden_states):
        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]
        for _ in range(self.config.num_res_blocks):
            hidden_states = ResnetBlock(
                block_out, dropout_prob=self.config.dropout
            )(hidden_states) 
            if hidden_states.shape[1] in self.config.attn_resolutions:
                hidden_states = AttnBlock()(hidden_states)
        if self.block_idx != self.config.num_resolutions - 1:
            hidden_states = Downsample(self.config.resample_with_conv)(hidden_states)
        return hidden_states


class ResnetBlock(nn.Module):
    out_channels: Optional[int] = None
    use_conv_shortcut: bool = False
    dropout_prob: float = 0.0

    @nn.compact
    def __call__(self, hidden_states):
        out_channels = self.out_channels or hidden_states.shape[-1]
        residual = hidden_states
        hidden_states = nn.GroupNorm()(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = nn.Conv(out_channels, [3, 3])(hidden_states)
        hidden_states = nn.GroupNorm()(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = nn.Dropout(self.dropout_prob, deterministic=True)(hidden_states)
        hidden_states = nn.Conv(out_channels, [3, 3])(hidden_states)
        if out_channels != residual.shape[-1]:
            if self.use_conv_shortcut:
                residual = nn.Conv(out_channels, [3, 3])(residual)
            else:
                residual = nn.Conv(out_channels, [1, 1])(residual)
        return hidden_states + residual
        

class AttnBlock(nn.Module):
    @nn.compact
    def __call__(self, hidden_states):
        residual = hidden_states
        hidden_states = nn.GroupNorm()(hidden_states)
        query = nn.Conv(hidden_states.shape[-1], [1, 1])(hidden_states)
        key = nn.Conv(hidden_states.shape[-1], [1, 1])(hidden_states)
        value = nn.Conv(hidden_states.shape[-1], [1, 1])(hidden_states)
        query, key, value = map(
            lambda x: x.reshape(x.shape[0], -1, x.shape[-1]),
            [query, key, value]
        )
        attn_weights = jnp.einsum("bqd,bkd->bqk", query, key)
        attn_weights *= hidden_states.shape[-1] ** -0.5
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        hidden_states = jnp.einsum("bqk,bkd->bqd", attn_weights, value)
        hidden_states = nn.Conv(hidden_states.shape[-1], [1, 1])(hidden_states)
        return hidden_states + residual

        
class Downsample(nn.Module):
    with_conv: bool
    
    @nn.compact
    def __call__(self, hidden_states):
        if self.with_conv:
            hidden_states = jnp.pad(
                hidden_states,
                [(0, 0), (0, 1), (0, 1), (0, 0)]
            )
            hidden_states = nn.Conv(
                hidden_states.shape[-1], [3, 3], 
                strides=[2, 2], 
                padding="VALID"
            )(hidden_states)
        else:
            hidden_states = nn.avg_pool(hidden_states, [2, 2], [2, 2])
        return hidden_states

        
class Upsample(nn.Module):
    with_conv: bool

    @nn.compact
    def __call__(self, hidden_states):
        B, H, W, C = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            (B, H * 2, W * 2, C),
            method="nearest"
        )
        if self.with_conv:
            hidden_states = nn.Conv(hidden_states.shape[-1], [3, 3])(hidden_states)
        return hidden_states


class UpsamplingBlock(nn.Module):
    config: VQGANConfig
    block_idx: int

    @nn.compact
    def __call__(self, hidden_states):
        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]
        for _ in range(self.config.num_res_blocks + 1):
            hidden_states = ResnetBlock(
                block_out, dropout_prob=self.config.dropout
            )(hidden_states)
            if hidden_states.shape[1] in self.config.attn_resolutions:
                hidden_states = AttnBlock()(hidden_states)
        if self.block_idx != 0:
            hidden_states = Upsample(self.config.resample_with_conv)(hidden_states)
        return hidden_states


class MidBlock(nn.Module):
    config: VQGANConfig
    no_attn: bool
    dropout: float

    @nn.compact
    def __call__(self, hidden_states):
        hidden_states = ResnetBlock(dropout_prob=self.dropout)(hidden_states)
        if not self.no_attn:
            hidden_states = AttnBlock()(hidden_states)
        hidden_states = ResnetBlock(dropout_prob=self.dropout)(hidden_states)
        return hidden_states
