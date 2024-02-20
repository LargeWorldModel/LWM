from typing import Any, Dict, List, Optional, Tuple, Union
import json
import warnings
import copy

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze
from flax.traverse_util import flatten_dict, unflatten_dict

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from transformers.generation.flax_utils import SampleState, FlaxLogitsProcessorList, FlaxSampleOutput, logger
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers import GenerationConfig

from tux import load_pickle, open_file
from lwm.llama import LLaMAConfig, LLAMA_STANDARD_CONFIGS, FlaxLLaMABlockCollection, RMSNorm


VIDEO_LLAMA_STANDARD_CONFIGS = LLAMA_STANDARD_CONFIGS


class VideoLLaMAConfig(LLaMAConfig):
    model_type = "video_llama"

    def __init__(self, vision_vocab_size=8448, tie_vision_embeddings=False, sample_mode='all', **kwargs):
        super().__init__(**kwargs)
        self.vision_vocab_size = vision_vocab_size # 8192 + 256
        self.tie_vision_embeddings = tie_vision_embeddings
        self.sample_mode = sample_mode

    @staticmethod
    def get_partition_rules(scan_layers=False, scan_axis=0):
        """Parition rules are orderd, so that the beginning rules match first."""
        if scan_layers:
            if scan_axis == 0:
                return (
                    # embeddings
                    ("transformer/wte/embedding", PS("tp", ("fsdp", "sp"))),
                    ("transformer/vte/embedding", PS("tp", ("fsdp", "sp"))),
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
                    ("vision_head/kernel", PS(("fsdp", "sp"), "tp")),
                    ('.*', PS(None)),
                )
            elif scan_axis == 1:
                return (
                    # embeddings
                    ("transformer/wte/embedding", PS("tp", ("fsdp", "sp"))),
                    ("transformer/vte/embedding", PS("tp", ("fsdp", "sp"))),
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
                    ("vision_head/kernel", PS(("fsdp", "sp"), "tp")),
                    ('.*', PS(None)),
                )
            else:
                raise ValueError(f"Invalid scan_axis {scan_axis}")
        else:
            return (
                # embeddings
                ("transformer/wte/embedding", PS("tp", ("fsdp", "sp"))),
                ("transformer/vte/embedding", PS("tp", ("fsdp", "sp"))),
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
                ("vision_head/kernel", PS(("fsdp", "sp"), "tp")),
                ('.*', PS(None)),
            )

    @classmethod
    def load_config(cls, path):
        if path in VIDEO_LLAMA_STANDARD_CONFIGS:
            return cls.from_dict(VIDEO_LLAMA_STANDARD_CONFIGS[path])
        load_type, load_path = path.split('::', 1)
        if load_type == 'pickle':
            return cls.from_dict(load_pickle(load_path)['llama_config'])
        elif load_type == 'json':
            with open_file(load_path, 'r') as fin:
                raw_config = fin.read()
            return cls.from_dict(json.loads(raw_config))
        else:
            raise ValueError(f'Unsupported load config type: {load_type}')


class FlaxVideoLLaMAPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VideoLLaMAConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: VideoLLaMAConfig,
        input_shape: Tuple = (4, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_cache(self, batch_size, max_length):
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        segment_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        vision_masks = jnp.ones((batch_size, max_length), dtype=bool)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, vision_masks, attention_mask, segment_ids, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    def init_weights(self, rng, input_shape, params=None):
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        vision_masks = jnp.ones(input_ids.shape, dtype=bool)
        segment_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, input_ids, vision_masks, attention_mask, segment_ids, position_ids, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward("")
    def __call__(
        self,
        input_ids,
        vision_masks,
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

        if segment_ids is None:
            segment_ids = jnp.zeros((batch_size, sequence_length))

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
            jnp.array(vision_masks, dtype="f4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(segment_ids, dtype="i4"),
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


class FlaxVideoLLaMAModule(nn.Module):
    config: VideoLLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.vte = nn.Embed(
            self.config.vision_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

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
        vision_masks,
        attention_mask,
        segment_ids,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_ids = input_ids.astype("i4")

        if input_ids.shape[1] == 1:
            if self.config.sample_mode == 'text':
                input_embeds = self.wte(input_ids)
            elif self.config.sample_mode == 'vision':
                input_embeds = self.vte(input_ids)
            elif self.config.sample_mode == 'all':
                raise NotImplementedError
            else:
                raise ValueError(f"Invalid sample_mode: {self.config.sample_mode}")
        else:
            input_text_embeds = self.wte(jnp.where(vision_masks, 0, input_ids))
            input_vision_embeds = self.vte(jnp.where(vision_masks, input_ids, 0))
            vision_masks = vision_masks[..., None].astype("f4") # 1 is vision, 0 is text
            input_embeds = input_text_embeds * (1 - vision_masks) + input_vision_embeds * vision_masks

        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        outputs = self.h(
            hidden_states,
            attention_mask,
            segment_ids,
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


class FlaxVideoLLaMAForCausalLMModule(nn.Module):
    config: VideoLLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.transformer = FlaxVideoLLaMAModule(self.config, dtype=self.dtype)
        self.vision_head = nn.Dense(
            self.config.vision_vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )
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
        vision_masks,
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
        if segment_ids is None:
            segment_ids = jnp.zeros_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )


        outputs = self.transformer(
            input_ids,
            vision_masks,
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

        if self.config.tie_vision_embeddings:
            shared_kernel = self.transformer.variables["params"]["vte"]["embedding"].T
            vision_logits = self.vision_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            vision_logits = self.vision_head(hidden_states)

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if self.config.sample_mode == 'all':
            if not return_dict:
                return (vision_logits, lm_logits,) + outputs[1:]

            return FlaxCausalLMOutput(logits=(vision_logits, lm_logits), hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        elif self.config.sample_mode == 'vision':
            if not return_dict:
                return (vision_logits,) + outputs[1:]

            return FlaxCausalLMOutput(logits=vision_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        elif self.config.sample_mode == 'text':
            if not return_dict:
                return (lm_logits,) + outputs[1:]

            return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        else:
            raise ValueError(f"Invalid sample_mode: {self.config.sample_mode}")



@add_start_docstrings("", "")
class FlaxVideoLLaMAForCausalLM(FlaxVideoLLaMAPreTrainedModel):
    module_class = FlaxVideoLLaMAForCausalLMModule

    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: Optional[jax.Array] = None, vision_masks = None
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
            "vision_masks": vision_masks
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        return {
            "past_key_values":  model_outputs.past_key_values,
            "position_ids": model_kwargs["position_ids"][:, -1:] + 1,
            "attention_mask": model_kwargs["attention_mask"],
            "vision_masks": model_kwargs["vision_masks"]
        }

    def _sample_vision(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        prng_key: Optional[jnp.ndarray] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        logits_warper: Optional[FlaxLogitsProcessorList] = None,
        cfg_scales: jnp.ndarray = 1.0,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape
        initial_len = cur_len

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self

        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = SampleState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=prng_key,
            model_kwargs=model_kwargs,
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def sample_search_body_fn(state):
            """state update fn."""
            prng_key, prng_key_next = jax.random.split(state.prng_key)
            model_outputs = model(state.running_token, params=params, **state.model_kwargs)

            logits = model_outputs.logits[:, -1]
            cond_logits, uncond_logits = jnp.split(logits, 2, axis=0)
            logits = uncond_logits + cfg_scales[:, None] * (cond_logits - uncond_logits)

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)
            # apply top_p, top_k, temperature
            logits = logits_warper(logits, logits, state.cur_len)

            next_token = jax.random.categorical(prng_key, logits, axis=-1)
            next_token = jax.lax.cond(
                (state.cur_len - initial_len + 1) % 257 == 0,
                lambda: jnp.full_like(next_token, 8192),
                lambda: next_token
            )
            next_token = jnp.concatenate([next_token, next_token], axis=0)

            #next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
                prng_key=prng_key_next,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = sample_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(sample_search_cond_fn, sample_search_body_fn, state)
        else:
            state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)

        return FlaxSampleOutput(sequences=state.sequences)

    def generate_vision(
        self,
        input_ids: jnp.ndarray,
        cfg_scales: jnp.ndarray,
        generation_config: Optional[GenerationConfig] = None,
        prng_key: Optional[jnp.ndarray] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        **kwargs,
    ):
        # Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # two conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same).
            if self.generation_config._from_model_config and self.generation_config._original_object_hash == hash(
                self.generation_config
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        logits_processor = logits_processor if logits_processor is not None else FlaxLogitsProcessorList()

        # set init values
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask") is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        if generation_config.decoder_start_token_id is None and self.config.is_encoder_decoder:
            raise ValueError("`decoder_start_token_id` has to be defined for encoder-decoder generation.")

        # decoder-only models should use left-padding for generation (can't be checked with `trace=True`)
        if not self.config.is_encoder_decoder and not trace:
            if (
                generation_config.pad_token_id is not None
                and jnp.sum(input_ids[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        batch_size = input_ids.shape[0]

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            if model_kwargs.get("encoder_outputs") is None:
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, params, model_kwargs)
            # prepare decoder_input_ids for generation
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                model_kwargs=model_kwargs,
            )

        # Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) "
                "to control the generation length.  recommend setting `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing`max_new_tokens`."
            )

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            logits_processor=logits_processor,
        )

        if not generation_config.do_sample and generation_config.num_beams == 1:
            raise NotImplementedError
        elif generation_config.do_sample and generation_config.num_beams == 1:
            logits_warper = self._get_logits_warper(generation_config=generation_config)
            return self._sample_vision(
                input_ids,
                generation_config.max_length,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                cfg_scales=cfg_scales,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif not generation_config.do_sample and generation_config.num_beams > 1:
            raise NotImplementedError
        else:
            raise NotImplementedError("`Beam sampling is currently not implemented.")
