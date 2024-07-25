import pprint
import os
from functools import partial

from tqdm import tqdm, trange
import numpy as np
from absl.app import run
import absl.logging as logging
import tux

import jax
import flax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
from transformers import AutoTokenizer

from lwm.data import DatasetFactory
from tux import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_mask,
    make_shard_and_gather_fns, with_sharding_constraint, define_flags_with_default,
    OptimizerFactory, StreamingCheckpointer
)
from lwm.llama import LLaMAConfig, FlaxLLaMAForCausalLMModule
from lwm.vision_llama import VideoLLaMAConfig, FlaxVideoLLaMAForCausalLMModule


FLAGS, FLAGS_DEF = define_flags_with_default(
    modality='text',
    use_data_sharded_loader=True,
    seed=42,
    mesh_dim='1,-1,1,1',
    dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer='LargeWorldModel/LWM-Text-1M',
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=VideoLLaMAConfig.get_default_config(),
    logger=tux.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    autoresume=False,
)


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = tux.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = tux.user_flags_to_config_dict(FLAGS, FLAGS_DEF)

    logger = tux.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    if jax.process_index() == 0:
        output_dir = logger.output_dir
    else:
        output_dir = os.path.join(logger.output_dir, logger.experiment_id)

    if FLAGS.modality == 'text':
        config_cls = LLaMAConfig
        llama_cls = FlaxLLaMAForCausalLMModule
    elif FLAGS.modality == 'vision,text':
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxVideoLLaMAForCausalLMModule
    else:
        raise ValueError(f"Unsupported modality: {FLAGS.modality}")

    mesh = config_cls.get_jax_mesh(FLAGS.mesh_dim)
    node_info = config_cls.get_ranks_and_size(mesh)

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer, node_info=node_info)
    if FLAGS.autoresume and tux.check_exists(output_dir):
        logging.info('Found existing output. Resuming dataset from latest checkpoint...')
        resume_path = f"{output_dir}/dataset.pkl"
        dataset.load_state_dict(tux.load_pickle(resume_path))
    elif FLAGS.load_dataset_state != '':
        dataset.load_state_dict(tux.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length

    if FLAGS.load_llama_config != '':
        llama_config = config_cls.load_config(FLAGS.load_llama_config)
        updates = config_cls(**FLAGS.llama)
        llama_config.update(dict(
            scan_attention=updates.scan_attention,
            scan_mlp=updates.scan_mlp,
            scan_query_chunk_size=updates.scan_query_chunk_size,
            scan_key_chunk_size=updates.scan_key_chunk_size,
            scan_mlp_chunk_size=updates.scan_mlp_chunk_size,
            scan_layers=updates.scan_layers,
            param_scan_axis=updates.param_scan_axis,
        ))
    else:
        llama_config = config_cls(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        bos_token_id=dataset.tokenizer.bos_token_id,
        eos_token_id=dataset.tokenizer.eos_token_id,
    ))
    if llama_config.vocab_size < dataset.vocab_size:
        llama_config.update(dict(vocab_size=dataset.vocab_size))
    llama_config.update(dict(mesh_dim=FLAGS.mesh_dim))

    model = llama_cls(
        llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_mask(config_cls.get_weight_decay_exclusions()),
        None,
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        batch = 512
        if FLAGS.modality == 'text':
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        elif FLAGS.modality == 'vision,text':
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        else:
            raise ValueError(f"Unsupported modality: {FLAGS.modality}")
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
        def loss_and_accuracy(params):
            if FLAGS.modality == 'text':
                logits = model.apply(
                    params,
                    batch['input_tokens'],
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                loss, acc = cross_entropy_loss_and_accuracy(
                    logits,
                    batch['target_tokens'],
                    batch['loss_masks']
                )
                metrics = dict(acc=acc)
                return loss, metrics
            elif FLAGS.modality == 'vision,text':
                vision_logits, text_logits = model.apply(
                    params,
                    batch['input_tokens'],
                    batch['input_vision_masks'],
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits,
                    jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_vision_masks']
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits,
                    jnp.where(batch['target_vision_masks'], 0, batch['target_tokens']),
                    batch['loss_masks'] * (1.0 - batch['target_vision_masks'])
                )
                loss = 0.5 * (vision_loss + text_loss)

                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                )
            else:
                raise ValueError(f"Unsupported modality: {FLAGS.modality}")
            return loss, metrics
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, loss_metrics), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            param_norm=global_norm(train_state.params),
            gradient_norm=global_norm(grads),
            **loss_metrics
        )
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
        if FLAGS.modality == 'text':
            logits = model.apply(
                train_state.params,
                batch['input_tokens'],
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            loss, acc = cross_entropy_loss_and_accuracy(
                logits,
                batch['target_tokens'],
                batch['loss_masks']
            )
            metrics = dict(
                eval_loss=loss,
                eval_acc=acc,
            )
        elif FLAGS.modality == 'vision,text':
            vision_logits, text_logits = model.apply(
                train_state.params,
                batch['input_tokens'],
                batch['input_vision_masks'],
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                vision_logits,
                jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_vision_masks']
            )
            text_loss, text_acc = cross_entropy_loss_and_accuracy(
                text_logits,
                jnp.where(batch['target_vision_masks'], 0, batch['target_tokens']),
                batch['loss_masks'] * (1.0 - batch['target_vision_masks'])
            )
            loss = 0.5 * (vision_loss + text_loss)
            metrics = dict(
                eval_loss=loss,
                eval_vision_accuracy=vision_acc,
                eval_vision_loss=vision_loss,
                eval_text_accuracy=text_acc,
                eval_text_loss=text_loss,
            )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        config_cls.get_partition_rules(llama_config.scan_layers, llama_config.param_scan_axis), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    if FLAGS.use_data_sharded_loader:
        batch_spec = PS(('dp', 'fsdp'), 'sp')
    else:
        batch_spec = PS()
    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), batch_spec),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    with mesh:
        train_state, restored_params = None, None

        if FLAGS.autoresume and tux.check_exists(output_dir):
            logging.info('Found existing output. Resuming model from latest checkpoint...')
            resume_path = f"trainstate::{output_dir}/streaming_train_state"
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                resume_path, train_state_shapes, shard_fns, max_buffer_size=32 * 2 ** 30
            )
        elif FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns, max_buffer_size=32 * 2 ** 30
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(flax.core.unfreeze(restored_params))
            del restored_params

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)
        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch
            )
            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metrics = jax.device_get(eval_metrics)
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    run(main)
