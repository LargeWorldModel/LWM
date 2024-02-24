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
    """
    A class representing a Vector Quantized Generative Adversarial Network (VQGAN) model.

    Attributes:
        vqgan_checkpoint (str): Path to the VQGAN model checkpoint.
        replicate (bool): Flag to indicate whether to replicate the model parameters across devices.
        config (VQGANConfig): Configuration object for the VQGAN model.
        params (dict): Loaded model parameters.
        model (VQGANModel): The VQGAN model instance.

    Methods:
        encode(pixel_values): Encodes input pixel values into latent representations.
        decode(encoding): Decodes latent representations back into pixel values.
    """
    def __init__(self, vqgan_checkpoint, replicate=False):
        """
        Initializes the VQGAN model with the given checkpoint and replication settings.

        Parameters:
            vqgan_checkpoint (str): Path to the VQGAN model checkpoint.
            replicate (bool, optional): Whether to replicate the model parameters across devices. Defaults to False.
        """
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
        """
        Wraps a function with JAX's jit or pmap for performance optimization based on the replication setting.

        Parameters:
            fn (Callable): The function to be wrapped.

        Returns:
            Callable: The wrapped function, optimized with jit or pmap.
        """
        if self.replicate:
            return jax.pmap(fn, devices=jax.local_devices())
        else:
            return jax.jit(fn)
    
    @cached_property
    def _encode(self):
        """
        Encodes input pixel values into latent representations using the VQGAN model.

        Parameters:
            pixel_values (jnp.ndarray): The input pixel values to encode.

        Returns:
            jnp.ndarray: The encoded latent representations.
        """
        def fn(pixel_values, params):
            return self.model.apply(
                {'params': params}, 
                pixel_values,
                method=self.model.encode
            )
        return partial(self._wrap_fn(fn), params=self.params)
    
    @cached_property
    def _decode(self):
        """
        Decodes latent representations back into pixel values using the VQGAN model.

        Parameters:
            encoding (jnp.ndarray): The latent representations to decode.

        Returns:
            jnp.ndarray: The decoded pixel values.
        """
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
    """
    Configuration class for the VQGAN model, containing various architectural and training settings.

    Attributes:
        resolution (int): The resolution of input images.
        num_channels (int): The number of channels in the input images.
        hidden_channels (int): The number of hidden channels in the model.
        channel_mult (tuple): Multipliers for channels in different stages of the model.
        num_res_blocks (int): The number of residual blocks in each stage.
        attn_resolutions (tuple): Resolutions at which to apply self-attention.
        no_attn_mid_block (bool): Whether to exclude self-attention in the middle block.
        z_channels (int): The number of channels in the latent space.
        num_embeddings (int): The number of embeddings in the quantizer.
        quantized_embed_dim (int): The dimensionality of quantized embeddings.
        dropout (float): The dropout rate.
        resample_with_conv (bool): Whether to use convolutional layers for resampling.
        commitment_cost (float): The commitment cost for vector quantization.

    Methods:
        get_default_config(updates): Returns the default configuration, optionally updated with provided values.
        load_config(path): Loads the configuration from a specified path.
    """
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
        """
        Initializes the VQGAN configuration with the specified settings.

        Parameters:
            resolution (int, optional): Resolution of input images. Defaults to 256.
            num_channels (int, optional): Number of channels in the input images. Defaults to 3.
            hidden_channels (int, optional): Number of hidden channels in the model. Defaults to 128.
            channel_mult (tuple, optional): Channel multipliers for different stages. Defaults to (1, 2, 2, 4, 6).
            num_res_blocks (int, optional): Number of residual blocks in each stage. Defaults to 2.
            attn_resolutions (tuple, optional): Resolutions for applying self-attention. Defaults to ().
            no_attn_mid_block (bool, optional): Exclude self-attention in the middle block. Defaults to True.
            z_channels (int, optional): Number of channels in the latent space. Defaults to 64.
            num_embeddings (int, optional): Number of embeddings in the quantizer. Defaults to 8192.
            quantized_embed_dim (int, optional): Dimensionality of quantized embeddings. Defaults to 64.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            resample_with_conv (bool, optional): Use convolution for resampling. Defaults to True.
            commitment_cost (float, optional): Commitment cost for vector quantization. Defaults to 0.25.
        """
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
        """
        Returns the default configuration for the VQGAN model, optionally updated with provided values.

        Parameters:
            updates (dict, optional): A dictionary of updates to apply to the default configuration.

        Returns:
            VQGANConfig: The default (or updated) model configuration.
        """
        config = function_args_to_config(cls.__init__)
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        config.num_resolutions = len(config.channel_mult)
        return config
    
    @classmethod
    def load_config(cls, path):
        """
        Loads the VQGAN model configuration from the specified path.

        Parameters:
            path (str): The path to the configuration file.

        Returns:
            VQGANConfig: The loaded model configuration.
        """
        return cls.get_default_config(cls)

        
class VQGANModel(nn.Module):
    """
    The VQGAN model, consisting of an encoder, decoder, and a vector quantizer for latent space discretization.

    Attributes:
        config (VQGANConfig): Configuration object for the VQGAN model.

    Methods:
        encode(pixel_values): Encodes input pixel values into quantized latent representations.
        decode(encoding, is_codebook_indices): Decodes quantized latent representations (or codebook indices) back into pixel values.
    """
    config: VQGANConfig

    def setup(self):
        """
        Sets up the VQGAN model components, including the encoder, decoder, quantizer, and related convolutional layers.
        """
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.quantize = VectorQuantizer(
            self.config.num_embeddings, self.config.quantized_embed_dim
        )
        self.quant_conv = nn.Conv(self.config.quantized_embed_dim, [1, 1])
        self.post_quant_conv = nn.Conv(self.config.z_channels, [1, 1])
    
    def encode(self, pixel_values):
        """
        Encodes input pixel values into quantized latent representations using the encoder and quantizer.

        Parameters:
            pixel_values (jnp.ndarray): The input pixel values to encode.

        Returns:
            tuple: A tuple containing the quantized latent representations and the corresponding codebook indices.
        """
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
        """
        Decodes quantized latent representations (or codebook indices) back into pixel values using the decoder.

        Parameters:
            encoding (jnp.ndarray): The quantized latent representations or codebook indices to decode.
            is_codebook_indices (bool, optional): Flag indicating whether 'encoding' contains codebook indices. Defaults to True.

        Returns:
            jnp.ndarray: The decoded pixel values.
        """
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
        """
        Processes input pixel values through the VQGAN model, encoding and then decoding them.

        Parameters:
            pixel_values (jnp.ndarray): The input pixel values to process.

        Returns:
            jnp.ndarray: The reconstructed pixel values after encoding and decoding.
        """
        encoding = self.encode(pixel_values)[1]
        recon = self.decode(encoding)
        return recon
    

class Encoder(nn.Module):
    """
    Encoder part of the VQGAN model, responsible for converting input images into a latent representation.

    Attributes:
        config (VQGANConfig): Configuration object specifying model parameters.

    Methods:
        __call__(pixel_values): Processes input pixel values through a series of convolutional and downsampling layers.
    """
    config: VQGANConfig
    
    @nn.compact
    def __call__(self, pixel_values):
        """
        Transforms input pixel values into a high-dimensional latent space.

        Parameters:
            pixel_values (jnp.ndarray): Input pixel values with shape [batch_size, height, width, channels].

        Returns:
            jnp.ndarray: The encoded latent representation of the input images.
        """
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
    """
    Decoder part of the VQGAN model, responsible for reconstructing images from latent representations.

    Attributes:
        config (VQGANConfig): Configuration object specifying model parameters.

    Methods:
        __call__(hidden_states): Processes latent representations through a series of upsampling and convolutional layers.
    """
    config: VQGANConfig

    @nn.compact
    def __call__(self, hidden_states):
        """
        Reconstructs images from their latent representations.

        Parameters:
            hidden_states (jnp.ndarray): Latent representations with shape [batch_size, latent_height, latent_width, latent_channels].

        Returns:
            jnp.ndarray: The reconstructed images with shape [batch_size, height, width, channels].
        """
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
    """
    Module for quantizing the continuous latent space of the encoder's output into discrete embeddings.

    Attributes:
        n_e (int): The number of embeddings.
        e_dim (int): The dimension of each embedding vector.

    Methods:
        __call__(z, encoding_indices): Quantizes the input tensor z into a discrete set of embeddings.
    """
    n_e: int
    e_dim: int

    @nn.compact
    def __call__(self, z, encoding_indices=None):
        """
        Quantizes the input tensor into a set of discrete embeddings or retrieves embeddings by indices.

        Parameters:
            z (jnp.ndarray): The input tensor to quantize.
            encoding_indices (jnp.ndarray, optional): Indices of embeddings to retrieve.

        Returns:
            jnp.ndarray: The quantized tensor or the retrieved embeddings based on encoding_indices.
        """
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
    """
    A downsampling block used in the Encoder, applying a series of convolutions and reducing spatial dimensions.

    Attributes:
        config (VQGANConfig): Configuration object specifying model parameters.
        block_idx (int): Index of the current block in the encoder.

    Methods:
        __call__(hidden_states): Applies downsampling to the input tensor.
    """
    config: VQGANConfig
    block_idx: int

    @nn.compact
    def __call__(self, hidden_states):
        """
        Applies convolutions and reduces the spatial dimensions of the input tensor.

        Parameters:
            hidden_states (jnp.ndarray): Input tensor to the downsampling block.

        Returns:
            jnp.ndarray: The downsampled tensor.
        """
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
    """
    A residual block, applying a series of convolutions and adding the input tensor to the output.

    Attributes:
        out_channels (int, optional): The number of output channels.
        use_conv_shortcut (bool): Whether to use a convolutional layer in the shortcut connection.
        dropout_prob (float): Dropout probability.

    Methods:
        __call__(hidden_states): Processes the input tensor through the residual block.
    """
    out_channels: Optional[int] = None
    use_conv_shortcut: bool = False
    dropout_prob: float = 0.0

    @nn.compact
    def __call__(self, hidden_states):
        """
        Applies convolutions and a residual connection to the input tensor.

        Parameters:
            hidden_states (jnp.ndarray): Input tensor to the residual block.

        Returns:
            jnp.ndarray: The output tensor of the residual block.
        """
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
    """
    An attention block, applying self-attention to the input tensor.

    Methods:
        __call__(hidden_states): Applies self-attention to the input tensor.
    """
    @nn.compact
    def __call__(self, hidden_states):
        """
        Applies self-attention to the input tensor.

        Parameters:
            hidden_states (jnp.ndarray): Input tensor to the attention block.

        Returns:
            jnp.ndarray: The output tensor with applied self-attention.
        """
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
    """
    Downsamples the input tensor, optionally using a convolutional layer.

    Attributes:
        with_conv (bool): Whether to use a convolutional layer for downsampling.

    Methods:
        __call__(hidden_states): Applies downsampling to the input tensor.
    """
    with_conv: bool
    
    @nn.compact
    def __call__(self, hidden_states):
        """
        Reduces the spatial dimensions of the input tensor.

        Parameters:
            hidden_states (jnp.ndarray): Input tensor to downsample.

        Returns:
            jnp.ndarray: The downsampled tensor.
        """
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
    """
    Upsamples the input tensor, optionally using a convolutional layer.

    Attributes:
        with_conv (bool): Whether to use a convolutional layer for upsampling.

    Methods:
        __call__(hidden_states): Applies upsampling to the input tensor.
    """
    with_conv: bool

    @nn.compact
    def __call__(self, hidden_states):
        """
        Increases the spatial dimensions of the input tensor.

        Parameters:
            hidden_states (jnp.ndarray): Input tensor to upsample.

        Returns:
            jnp.ndarray: The upsampled tensor.
        """
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
    """
    An upsampling block used in the Decoder, applying a series of convolutions and increasing spatial dimensions.

    Attributes:
        config (VQGANConfig): Configuration object specifying model parameters.
        block_idx (int): Index of the current block in the decoder.

    Methods:
        __call__(hidden_states): Applies upsampling to the input tensor.
    """
    config: VQGANConfig
    block_idx: int

    @nn.compact
    def __call__(self, hidden_states):
        """
        Applies convolutions and increases the spatial dimensions of the input tensor.

        Parameters:
            hidden_states (jnp.ndarray): Input tensor to the upsampling block.

        Returns:
            jnp.ndarray: The upsampled tensor.
        """
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
    """
    A middle block used in both Encoder and Decoder, applying a series of transformations without changing spatial dimensions.

    Attributes:
        config (VQGANConfig): Configuration object specifying model parameters.
        no_attn (bool): Whether to exclude self-attention in this block.
        dropout (float): Dropout probability.

    Methods:
        __call__(hidden_states): Processes the input tensor through the middle block.
    """
    config: VQGANConfig
    no_attn: bool
    dropout: float

    @nn.compact
    def __call__(self, hidden_states):
        """
        Applies convolutions, optional self-attention, and a residual connection to the input tensor.

        Parameters:
            hidden_states (jnp.ndarray): Input tensor to the middle block.

        Returns:
            jnp.ndarray: The output tensor of the middle block.
        """
        hidden_states = ResnetBlock(dropout_prob=self.dropout)(hidden_states)
        if not self.no_attn:
            hidden_states = AttnBlock()(hidden_states)
        hidden_states = ResnetBlock(dropout_prob=self.dropout)(hidden_states)
        return hidden_states
