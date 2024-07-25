from absl.app import run
import math
from tqdm import tqdm
from PIL import Image
import decord
from functools import cached_property
import numpy as np
import jax
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from transformers import GenerationConfig, AutoTokenizer
from tux import (
    define_flags_with_default, StreamingCheckpointer, JaxDistributedConfig,
    set_random_seed, get_float_dtype_by_name, JaxRNG, next_rng,
    match_partition_rules, make_shard_and_gather_fns,
    with_sharding_constraint, tree_apply, open_file
)
from lwm.vision_llama import VideoLLaMAConfig, FlaxVideoLLaMAForCausalLM
from lwm.vqgan import VQGAN


FLAGS, FLAGS_DEF = define_flags_with_default(
    prompt="",
    input_file="",
    vqgan_checkpoint="",
    temperature=0.2,
    max_n_frames=8,
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


class Sampler:
    def __init__(self):
        self.mesh = VideoLLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
        self.vqgan = VQGAN(FLAGS.vqgan_checkpoint, replicate=False)
        self.prefix_tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer, truncation_side='left', padding_side='left')
        self.tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
        self.n_tokens_per_frame = 257
        self.min_buffer_size = 256
        self.sharded_rng = next_rng()
        self._load_model()

    @property
    def block_size(self):
        return max(self.config.scan_query_chunk_size, self.config.scan_key_chunk_size) * self.mesh.shape['sp']

    @property
    def data_dim(self):
        return self.mesh.shape['dp'] * self.mesh.shape['fsdp']

    def _process_frame(self, image, size):
        width, height = image.size
        if width < height:
            new_width = size
            new_height = int(size * height / width)
        else:
            new_height = size
            new_width = int(size * width / height)
        image = image.resize((new_width, new_height))

        left = (new_width - size) / 2
        top = (new_height - size) / 2
        right = (new_width + size) / 2
        bottom = (new_height + size) / 2
        image = image.crop((left, top, right, bottom))
        return np.array(image, dtype=np.float32) / 127.5 - 1

    def _read_process_vision(self, path, max_n_frames):
        f = open_file(path, 'rb')
        if path.endswith('.png') or path.endswith('.jpg'):
            image = Image.open(f).convert('RGB')
            vision = self._process_frame(image, 256)[None]
        else:
            vr = decord.VideoReader(f, ctx=decord.cpu(0))
            duration = len(vr)
            if duration <= max_n_frames:
                frame_id_list = list(range(duration))
            else:
                frame_id_list = np.linspace(0, duration - 1, max_n_frames, dtype=int).tolist()
            video = vr.get_batch(frame_id_list).asnumpy()
            vision = np.stack([self._process_frame(Image.fromarray(frame), 256) for frame in video])

        B = 1
        encodings = []
        for i in range(0, len(vision), 1):
            v = vision[i:i + B]
            if len(v) % B == 0:
                n_pad = 0
            else:
                n_pad = B - len(v) % B
            v = np.pad(v, ((n_pad, 0), (0, 0), (0, 0), (0, 0)))
            enc = jax.device_get(self.vqgan.encode(v))[1].astype(int)
            enc = enc[n_pad:]
            for t in range(len(enc)):
                encodings.extend(enc[t].reshape(-1).tolist())
                if t == len(enc) - 1:
                    encodings.append(8193)
                else:
                    encodings.append(8192)
        return encodings

    def construct_input(self, prompts, max_n_frames):
        max_input_length = max_n_frames * self.n_tokens_per_frame + self.min_buffer_size
        max_input_length = int(math.ceil(max_input_length / self.block_size) * self.block_size)

        vision_start = self.tokenizer.encode('<vision>')
        vision_end = self.tokenizer.encode('</vision>')

        input_ids = np.zeros((len(prompts), max_input_length), dtype=int)
        vision_masks = np.zeros((len(prompts), max_input_length), dtype=bool)
        attention_mask = np.zeros((len(prompts), max_input_length), dtype=int)
        for i, prompt in enumerate(tqdm(prompts)):
            vision = self._read_process_vision(prompt['input_path'], max_n_frames)
            text_1 = self.tokenizer.encode(f"<s>You are a helpful assistant. USER: {prompt['question']}\n")
            tail = self.tokenizer.encode(" ASSISTANT:")

            tokens, vm = [], []
            tokens.extend(text_1)
            vm.extend([False] * len(text_1))
            tokens.extend(vision_start)
            vm.extend([False] * len(vision_start))
            tokens.extend(vision)
            vm.extend([True] * len(vision))
            tokens.extend(vision_end)
            vm.extend([False] * len(vision_end))
            tokens.extend(tail)
            vm.extend([False] * len(tail))
            assert len(tokens) < max_input_length, (len(tokens), max_input_length)
            assert len(tokens) == len(vm)
            input_ids[i, -len(tokens):] = tokens
            vision_masks[i, -len(tokens):] = vm
            attention_mask[i, -len(tokens):] = 1
        return {
            'input_ids': input_ids,
            'vision_masks': vision_masks,
            'attention_mask': attention_mask
        }


    def _load_model(self):
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
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        ))
        llama_config.update(dict(mesh_dim=FLAGS.mesh_dim))
        self.config = llama_config

        self.model = FlaxVideoLLaMAForCausalLM(
            llama_config,
            input_shape=(512, self.block_size),
            seed=FLAGS.seed,
            _do_init=False,
            dtype=get_float_dtype_by_name(FLAGS.dtype),
        )

        with jax.default_device(jax.devices("cpu")[0]):
            _, self.params = StreamingCheckpointer.load_trainstate_checkpoint(
                    FLAGS.load_checkpoint, disallow_trainstate=True, max_buffer_size=32 * 2 ** 30
            )
        self.model_ps = match_partition_rules(
            VideoLLaMAConfig.get_partition_rules(llama_config.scan_layers, llama_config.param_scan_axis), self.params
        )
        shard_fns, _ = make_shard_and_gather_fns(
            self.model_ps, get_float_dtype_by_name(FLAGS.dtype)
        )

        with self.mesh:
            self.params = tree_apply(shard_fns, self.params)

    @cached_property
    def _forward_generate(self):
        def fn(params, rng, batch):
            batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
            rng_generator = JaxRNG(rng)
            output = self.model.generate(
                batch['input_ids'],
                vision_masks=batch['vision_masks'],
                attention_mask=batch['attention_mask'],
                params=params['params'],
                prng_key=rng_generator(),
                generation_config=GenerationConfig(
                    max_new_tokens=self.block_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=FLAGS.temperature,
                    do_sample=True,
                )
            ).sequences[:, batch['input_ids'].shape[1]:]
            return output, rng_generator()
        return pjit(
            fn,
            in_shardings=(self.model_ps, PS(), PS()),
            out_shardings=(PS(), PS())
        )

    def __call__(self, prompts, max_n_frames):
        batch = self.construct_input(prompts, max_n_frames)
        with self.mesh:
            output, self.sharded_rng = self._forward_generate(
                self.params, self.sharded_rng, batch
            )
            output = jax.device_get(output)
        output_text = []
        for text in list(self.tokenizer.batch_decode(output, skip_special_tokens=True)):
            if self.tokenizer.eos_token in text:
                text = text.split(self.tokenizer.eos_token, maxsplit=1)[0]
            output_text.append(text)
        return output_text

def main(argv):
    assert FLAGS.prompt != ''
    assert FLAGS.input_file != ''

    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    set_random_seed(FLAGS.seed)

    prompts = [{'input_path': FLAGS.input_file, 'question': FLAGS.prompt}]
    sampler = Sampler()
    output = sampler(prompts, FLAGS.max_n_frames)[0]
    print(f"Question: {FLAGS.prompt}\nAnswer: {output}")

if __name__ == "__main__":
    run(main)
