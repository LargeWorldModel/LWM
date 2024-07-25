from absl.app import run
from tqdm import tqdm
import imageio
import numpy as np
from PIL import Image
from transformers import GenerationConfig, AutoTokenizer
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from tux import (
    define_flags_with_default, StreamingCheckpointer, JaxDistributedConfig,
    set_random_seed, get_float_dtype_by_name, JaxRNG,
    match_partition_rules, make_shard_and_gather_fns,
    with_sharding_constraint, tree_apply, next_rng
)
from lwm.vision_llama import VideoLLaMAConfig, FlaxVideoLLaMAForCausalLM
from lwm.vqgan import VQGAN


FLAGS, FLAGS_DEF = define_flags_with_default(
    prompt='Fireworks over the city',
    output_file='',
    temperature_image=1.0,
    temperature_video=1.0,
    top_k_image=8192,
    top_k_video=100,
    cfg_scale_image=1.0,
    cfg_scale_video=1.0,
    vqgan_checkpoint='',
    n_frames=1,
    seed=1234,
    mesh_dim='1,-1,1,1',
    dtype='fp32',
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    tokenizer='LargeWorldModel/LWM-Text-1M',
    llama=VideoLLaMAConfig.get_default_config(),
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


def main(argv):
    assert FLAGS.output_file != ''
    if FLAGS.output_file.endswith('mp4'):
        assert FLAGS.n_frames > 1
    elif FLAGS.output_file.endswith('png') or FLAGS.output_file.endswith('jpg'):
        assert FLAGS.n_frames == 1
    else:
        raise ValueError(f"Unsupported output file extension: {FLAGS.output_file}")

    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    set_random_seed(FLAGS.seed)

    tokens_per_frame = 257
    vqgan = VQGAN(FLAGS.vqgan_checkpoint, replicate=False)
    mesh = VideoLLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
    prefix_tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer, truncation_side='left', padding_side='left')
    if FLAGS.load_llama_config != '':
        llama_config = VideoLLaMAConfig.load_config(FLAGS.load_llama_config)
        updates = VideoLLaMAConfig(**FLAGS.llama)
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
        llama_config = VideoLLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ))
    llama_config.update(dict(mesh_dim=FLAGS.mesh_dim))

    with jax.default_device(jax.devices("cpu")[0]):
        _, params = StreamingCheckpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, disallow_trainstate=True, max_buffer_size=32 * 2 ** 30
        )
        model = FlaxVideoLLaMAForCausalLM(
            llama_config,
            input_shape=(512, 8192),
            seed=FLAGS.seed,
            _do_init=False,
            dtype=get_float_dtype_by_name(FLAGS.dtype),
        )
        model_ps = match_partition_rules(
            VideoLLaMAConfig.get_partition_rules(llama_config.scan_layers, llama_config.param_scan_axis), params
        )
        shard_fns, _ = make_shard_and_gather_fns(
            model_ps, get_float_dtype_by_name(FLAGS.dtype)
        )

        with mesh:
            params = tree_apply(shard_fns, params)

    def _forward_generate(params, rng, batch, n_tokens, cfg_scale, top_k, temperature):
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
        cfg_scales = jnp.ones((batch['input_ids'].shape[0] // 2,), dtype=jnp.float32) * cfg_scale
        cfg_scales = with_sharding_constraint(cfg_scales, PS(('dp', 'fsdp')))
        rng_generator = JaxRNG(rng)
        output = model.generate_vision(
            batch['input_ids'],
            cfg_scales,
            attention_mask=batch['attention_mask'],
            vision_masks=batch['vision_masks'],
            params=params['params'],
            prng_key=rng_generator(),
            generation_config=GenerationConfig(
                max_new_tokens=n_tokens,
                min_new_tokens=n_tokens,
                pad_token_id=tokenizer.pad_token_id,
                temperature=temperature,
                do_sample=True,
                top_k=top_k,
            )
        ).sequences[:, batch['input_ids'].shape[1]:]
        return output, rng_generator()
    _sharded_forward_generate = pjit(
        _forward_generate,
        in_shardings=(model_ps, PS(), PS()),
        out_shardings=(PS(), PS()),
        static_argnums=(3, 4, 5, 6)
    )

    # Generate an image or first frame (for video)
    def generate_first_frame(prompts, max_input_length):
        nonlocal sharded_rng
        uncond_prompts = ["<s><vision>"] * len(prompts)
        prompts = prompts + uncond_prompts
        inputs = prefix_tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=max_input_length,
            return_tensors='np'
        )
        batch = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            vision_masks=np.zeros(inputs.input_ids.shape, dtype=bool),
        )
        with mesh:
            output, sharded_rng = _sharded_forward_generate(
                params, sharded_rng, batch,
                tokens_per_frame, FLAGS.cfg_scale_image,
                FLAGS.top_k_image, FLAGS.temperature_image
            )
            output = jax.device_get(output)
            output = np.split(output, 2, axis=0)[0]
        output = output.reshape(len(prompts) // 2, tokens_per_frame)
        image = vqgan.decode(output[:, :-1].reshape(-1, 16, 16))
        image = ((jax.device_get(image) + 1) * 127.5).astype(np.uint8)
        return output, image

    sharded_rng = next_rng()
    prompts = [FLAGS.prompt]
    entries = []
    for prompt in prompts:
        entries.append({
            'caption': prompt,
            'prompt': f"<s>You are a helpful assistant. USER: Generate an image of {prompt} ASSISTANT: <vision>",
        })

    B = 1
    images, image_encodings = [], []
    for i in tqdm(list(range(0, len(entries), B))):
        entries_i = entries[i:i + B]
        prompts = [entry['prompt'] for entry in entries_i]
        img_enc, img = generate_first_frame(prompts, max_input_length=128)
        image_encodings.extend(img_enc)
        images.extend(img)

    if FLAGS.n_frames == 1:
        image = images[0]
        Image.fromarray(image).save(FLAGS.output_file)
        return

    # Generate the rest of the video
    def generate_video_pred(prompts, images, max_input_length):
        nonlocal sharded_rng
        images = np.concatenate([images, images], axis=0)
        uncond_prompts = ["<s><vision>"] * len(prompts)
        prompts = prompts + uncond_prompts
        inputs = prefix_tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=max_input_length,
            return_tensors='np'
        )
        batch = dict(
            input_ids=np.concatenate([inputs.input_ids, images], axis=1),
            attention_mask=np.concatenate([inputs.attention_mask, np.ones(images.shape, dtype=inputs.attention_mask.dtype)], axis=1),
            vision_masks=np.concatenate([
                np.zeros(inputs.input_ids.shape, dtype=bool),
                np.ones(images.shape, dtype=bool)
            ], axis=1),
        )
        with mesh:
            output, sharded_rng = _sharded_forward_generate(
                params, sharded_rng, batch,
                (FLAGS.n_frames - 1) * tokens_per_frame, FLAGS.cfg_scale_video,
                FLAGS.top_k_video, FLAGS.temperature_video
            )
            output = jax.device_get(output)
            output = np.split(output, 2, axis=0)[0]
        output = output.reshape(len(prompts) // 2, FLAGS.n_frames - 1, tokens_per_frame)
        output = np.concatenate([images[:len(prompts) // 2, None], output], axis=1)
        output = output[:, :, :-1].reshape(-1, FLAGS.n_frames, 16, 16)
        vision = []
        for v in output:
            v = vqgan.decode(v)
            v = ((jax.device_get(v) + 1) * 127.5).astype(np.uint8)
            vision.append(v)
        return vision

    new_entries = []
    for img_enc, entry in zip(image_encodings, entries):
        new_entries.append({
            'caption': entry['caption'],
            'prompt': f"<s>You are a helpful assistant. USER: Generate a video of {entry['caption']} ASSISTANT: <vision>",
            'image': np.array(img_enc, dtype=np.int32),
        })
    entries = new_entries

    B = 1
    videos = []
    for i in tqdm(list(range(0, len(entries), B))):
        entries_i = entries[i:i + B]
        prompts = [entry['prompt'] for entry in entries_i]
        images = np.array([entry['image'] for entry in entries_i], dtype=np.int32)
        videos.extend(generate_video_pred(prompts, images, max_input_length=128))

    video = videos[0]
    writer = imageio.get_writer(FLAGS.output_file, fps=4)
    for frame in video:
        writer.append_data(frame)
    writer.close()

    print('done')

if __name__ == "__main__":
    run(main)
