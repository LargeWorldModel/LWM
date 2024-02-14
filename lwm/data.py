import time
import random
from functools import partial
import json
from multiprocessing import Pool

from tux import open_file
from ml_collections import ConfigDict
import numpy as np
import jax
from jax.experimental.multihost_utils import host_local_array_to_global_array
from jax.sharding import PartitionSpec as PS
from datasets import load_dataset


class DatasetFactory(object):
    """ Datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = 'huggingface'
        config.text_processor = TextProcessor.get_default_config()
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()

        config.vision_text_processor = VisionTextProcessor.get_default_config()
        config.json_vision_dataset = JsonVisionDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        if config.type == 'huggingface':
            text_processor = TextProcessor(config.text_processor, tokenizer)
            return HuggingfaceDataset(
                config.huggingface_dataset, tokenizer, text_processor, **kwargs
            )
        elif config.type == 'json':
            text_processor = TextProcessor(config.text_processor, tokenizer)
            return JsonDataset(config.json_dataset, tokenizer, text_processor, **kwargs)
        elif config.type == 'json_vision':
            vision_text_processor = VisionTextProcessor(config.vision_text_processor, tokenizer)
            return JsonVisionDataset(config.json_vision_dataset, tokenizer, vision_text_processor, **kwargs)
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError('DatasetFactory is a static class and should not be instantiated.')


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ''
        config.fields = ''
        config.subfield_separator = ' '
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ''
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields != '' or self.config.fields_from_example != '', (
            'Either fields or fields_from_example must be specified.'
        )
        self.tokenizer = tokenizer

    def __call__(self, example, has_aux=False, add_bos_token=True, add_eos_token=True):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        token_buffer = []
        loss_mask_buffer = []

        if add_bos_token and self.config.add_bos_token:
            token_buffer.append(self.tokenizer.bos_token_id)
            loss_mask_buffer.append(0.0)

        if self.config.fields_from_example != '':
            fields = example[self.config.fields_from_example].split(',')
        else:
            fields = self.config.fields.split(',')

        for i, field in enumerate(fields):
            if field.startswith('[') and field.endswith(']'):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field == '<|bos|>':
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(mask)
            elif field == '<|eos|>':
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(mask)
            else:
                subfields = field.split('+')
                text = self.config.subfield_separator.join(
                    [example[subfield] for subfield in subfields]
                )
                if i == 0:
                    text = self.config.prepend_text + text
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if add_eos_token and self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer, *aux


class VisionTextProcessor(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ''
        config.subfield_separator = ' '
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ''
        config.fields_index = -1
        config.eof_token = 8192 # denotes end of each frame for video generation
        config.eov_token = 8193 # denotes end of vision generation
        config.n_tokens_per_frame = 256 # 16 x 16 VQ codes
        config.max_n_frames = -1
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields_from_example != '', (
            'fields_from_example must be specified.'
        )
        self.tokenizer = tokenizer
        self.vision_start = tokenizer.encode('<vision>')
        self.vision_end = tokenizer.encode('</vision>')

    def __call__(self, example, has_aux=False, add_bos_token=True, add_eos_token=True):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        rand_state = random.Random(aux[-1]) # makes augmentations deterministic by line number
        token_buffer = []
        loss_mask_buffer = []
        vision_mask = []

        fields = example[self.config.fields_from_example]
        if isinstance(fields, (tuple, list)):
            if self.config.fields_index >= 0:
                fields = fields[self.config.fields_index]
            else:
                # seed based on line number
                fields = rand_state.choice(fields)
        fields = fields.split(',')

        if add_bos_token and self.config.add_bos_token:
            token_buffer.append(self.tokenizer.bos_token_id)
            loss_mask_buffer.append(0.0)
            vision_mask.append(False)

        for i, field in enumerate(fields):
            if field.startswith('[') and field.endswith(']'):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field == '<|bos|>':
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(mask)
                vision_mask.append(False)
            elif field == '<|eos|>':
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(mask)
                vision_mask.append(False)
            elif 'vision' in field:
                vision_tokens = example[field]
                n_frames = int(len(vision_tokens) / self.config.n_tokens_per_frame)
                if self.config.max_n_frames > 0 and n_frames > self.config.max_n_frames: # uniformly select
                    idxs = np.linspace(0, n_frames - 1, self.config.max_n_frames).astype(int)
                    new_vision_tokens = []
                    for idx in idxs:
                        new_vision_tokens.extend(vision_tokens[idx * self.config.n_tokens_per_frame:(idx + 1) * self.config.n_tokens_per_frame])
                    vision_tokens = new_vision_tokens
                    n_frames = self.config.max_n_frames
                assert int(len(vision_tokens) / self.config.n_tokens_per_frame) == n_frames, (int(len(vision_tokens) / self.config.n_tokens_per_frame), n_frames)

                assert n_frames > 0, len(vision_tokens)
                tokens = list(self.vision_start)
                for j in range(n_frames):
                    tokens.extend(vision_tokens[j*self.config.n_tokens_per_frame:(j+1)*self.config.n_tokens_per_frame])
                    if j == n_frames - 1: # last frame
                        tokens.append(self.config.eov_token)
                    else:
                        tokens.append(self.config.eof_token)
                tokens.extend(self.vision_end) 

                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                vision_mask.extend([False] * len(self.vision_start))
                vision_mask.extend([True] * (self.config.n_tokens_per_frame * n_frames + n_frames)) # include extra eof/eov token at the end of each frame
                vision_mask.extend([False] * len(self.vision_end))
            else:
                subfields = field.split('+')
                text = self.config.subfield_separator.join(
                    [example[subfield] for subfield in subfields]
                )
                if i == 0:
                    text = self.config.prepend_text + text
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                vision_mask.extend([False] * len(tokens))
        
        if add_eos_token and self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)
            vision_mask.append(False)

        assert len(token_buffer) == len(loss_mask_buffer) == len(vision_mask), (len(token_buffer), len(loss_mask_buffer), len(vision_mask))
        keep = True
        return token_buffer, loss_mask_buffer, vision_mask, keep, *aux


class HuggingfaceDataset(object):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._dataset = load_dataset(
            self.config.path, name, split=split, streaming=self.config.streaming
        )

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = 0
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for index, example in enumerate(self._dataset):
                tokens, loss_masks = self.text_processor(example)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                while len(token_buffer) > chunk_size + 1:
                    total_tokens += chunk_size
                    metrics = {
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                    }
                    batch = {
                        'input_tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    if self.config.always_start_with_bos:
                        batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                    yield batch, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(config=self.config)

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200
        config.pad = False
        config.use_data_sharded_loader = True
        config.return_local_batch = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor, node_info):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._node_info = node_info
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self._total_tokens = self.config.tokens_count_at_start

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        index, file_loc = self._index, self._file_loc
        with open_file(self.config.path, 'r') as fin:
            fin.seek(file_loc)
            while True:
                line = fin.readline()
                file_loc = fin.tell()
                if not line:   # Reached EOF
                    index = 0
                    fin.seek(0)
                    continue

                data = self.parse_json(line)
                if data is not None and (not self.config.use_data_sharded_loader or index % self._node_info['dp_node_size'] == self._node_info['dp_node_rank']):
                    # JSON parsing succeeded
                    yield data, file_loc, index
                index += 1

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                self._file_loc = loc
                self._index = index
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(
                self.json_iterator(), self.config.tokenizer_parallel_batch_size
            )
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator),
                    chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator),
                        chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        global_chunk_size = self.config.batch_size * self.config.seq_length
        if self.config.use_data_sharded_loader:
            local_batch_size = self.config.batch_size // self._node_info['dp_node_size']
        else:
            local_batch_size = self.config.batch_size
        chunk_size = local_batch_size * self.config.seq_length

        token_buffer = []
        loss_mask_buffer = []

        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for tokens, loss_masks, loc, index in self.parallel_example_iterator():
            self._file_loc = loc
            self._index = index
            if self.config.pad:
                tokens = tokens[:self.config.seq_length + 1]
                tokens.extend([self._tokenizer.bos_token_id] * (self.config.seq_length + 1 - len(tokens)))
                loss_masks = loss_masks[:self.config.seq_length + 1]
                loss_masks.extend([0.0] * (self.config.seq_length + 1 - len(loss_masks)))
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += global_chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size:]
                average_throughput = global_chunk_size / np.mean(step_times)
                accumulated_throughput = (
                    (self._total_tokens - start_tokens) / (time.time() - start_time)
                )
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': self._total_tokens,
                    'dataset_accumulated_tps': accumulated_throughput,
                    'dataset_average_tps': average_throughput,
                }
                batch = {
                    'input_tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        local_batch_size, -1
                    ),
                    'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                        local_batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                        local_batch_size, -1
                    ),
                }
                batch.update({
                    'input_vision_masks': np.zeros(batch['input_tokens'].shape, dtype=bool),
                    'target_vision_masks': np.zeros(batch['input_tokens'].shape, dtype=bool),
                })
                if self.config.always_start_with_bos:
                    batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id

                if self.config.use_data_sharded_loader and not self.config.return_local_batch:
                    mesh = self._node_info['mesh']
                    sp_nodes_size = max(1, mesh.shape['sp'] // jax.local_device_count())
                    sp_nodes_rank = jax.process_index() % sp_nodes_size
                    assert self.config.seq_length % sp_nodes_size == 0, (self.config.seq_len, sp_nodes_size)
                    seq_chunk_size = self.config.seq_length // sp_nodes_size
                    batch = {k: v[:, sp_nodes_rank*seq_chunk_size:(sp_nodes_rank+1)*seq_chunk_size] for k, v in batch.items()}
                    batch = host_local_array_to_global_array(batch, self._node_info['mesh'], PS(('dp', 'fsdp'), 'sp'))

                yield batch, metrics
                if self.config.pad:
                    token_buffer, loss_mask_buffer = [], []
                else:
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def _make_callback(self, v):
        return lambda index: v[index]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
        )

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))
        self._index = state_dict.get('index', self.config.example_index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)
        self._total_tokens = state_dict.get('total_tokens', self.config.tokens_count_at_start)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)


class JsonVisionDataset(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.seq_length = 384
        config.batch_size = 4
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200
        config.use_data_sharded_loader = True
        config.return_local_batch = False
        config.mode = 'pad'

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor, node_info):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._node_info = node_info
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self._total_tokens = 0

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        index, file_loc = self._index, self._file_loc
        with open_file(self.config.path, 'r', block_size=50 * 2 ** 20) as fin:
            fin.seek(file_loc)
            while True:
                line = fin.readline()
                file_loc = fin.tell()
                if not line:   # Reached EOF
                    index = 0
                    fin.seek(0)
                    continue
                if not self.config.use_data_sharded_loader or index % self._node_info['dp_node_size'] == self._node_info['dp_node_rank']:
                    data = self.parse_json(line)
                    if data is not None:
                        # JSON parsing succeeded
                        yield data, file_loc, index
                index += 1

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                self._file_loc = loc
                self._index = index
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(
                self.json_iterator(), self.config.tokenizer_parallel_batch_size
            )
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator),
                    chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator),
                        chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        if self.config.mode == 'pad':
            fn = self._iter_pad
        elif self.config.mode == 'no_pad':
            fn = self._iter_no_pad
        else:
            raise ValueError(f'Unknown mode: {self.config.mode}')
        return fn()
        
    def _iter_pad(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        if self.config.use_data_sharded_loader:
            local_batch_size = self.config.batch_size // self._node_info['dp_node_size']
        else:
            local_batch_size = self.config.batch_size
        last_time = 0.0
        buffer = []
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for tokens, loss_masks, vision_masks, keep, loc, index in self.parallel_example_iterator():
            if not keep:
                continue
            self._file_loc = loc
            self._index = index
            buffer.append((tokens, loss_masks, vision_masks))
            while len(buffer) >= local_batch_size:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size:]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (
                    (self._total_tokens - start_tokens) / (time.time() - start_time)
                )
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': self._total_tokens,
                    'dataset_accumulated_tps': accumulated_throughput,
                    'dataset_average_tps': average_throughput,
                }

                batch = {
                    'input_tokens': np.full(
                        (local_batch_size, self.config.seq_length),
                        self._tokenizer.bos_token_id,
                        dtype=np.int32
                    ),
                    'target_tokens': np.full(
                        (local_batch_size, self.config.seq_length),
                        self._tokenizer.bos_token_id,
                        dtype=np.int32
                    ),
                    'loss_masks': np.zeros(
                        (local_batch_size, self.config.seq_length),
                        dtype=np.float32
                    ),
                    'input_vision_masks': np.zeros(
                        (local_batch_size, self.config.seq_length),
                        dtype=bool
                    ),
                    'target_vision_masks': np.zeros(
                        (local_batch_size, self.config.seq_length),
                        dtype=bool
                    )
                }
                for i in range(local_batch_size):
                    tokens, loss_masks, vision_masks = buffer[i]
                    if len(tokens) > self.config.seq_length:
                        tokens = tokens[:self.config.seq_length + 1]
                        loss_masks = loss_masks[1:self.config.seq_length + 1]
                        vision_masks = vision_masks[:self.config.seq_length + 1]
                    input_tokens, target_tokens = tokens[:-1], tokens[1:]
                    input_vision_masks, target_vision_masks = vision_masks[:-1], vision_masks[1:]
                    loss_masks = loss_masks[1:]
                    batch['input_tokens'][i, :len(input_tokens)] = input_tokens
                    batch['target_tokens'][i, :len(target_tokens)] = target_tokens
                    batch['input_vision_masks'][i, :len(input_vision_masks)] = input_vision_masks
                    batch['target_vision_masks'][i, :len(target_vision_masks)] = target_vision_masks
                    batch['loss_masks'][i, :len(loss_masks)] = loss_masks

                if self.config.use_data_sharded_loader and not self.config.return_local_batch:
                    mesh = self._node_info['mesh']
                    sp_nodes_size = max(1, mesh.shape['sp'] // jax.local_device_count())
                    sp_nodes_rank = jax.process_index() % sp_nodes_size
                    assert self.config.seq_length % sp_nodes_size == 0, (self.config.seq_len, sp_nodes_size)
                    seq_chunk_size = self.config.seq_length // sp_nodes_size
                    batch = {k: v[:, sp_nodes_rank*seq_chunk_size:(sp_nodes_rank+1)*seq_chunk_size] for k, v in batch.items()}
                    batch = host_local_array_to_global_array(batch, self._node_info['mesh'], PS(('dp', 'fsdp'), 'sp'))
                yield batch, metrics
                buffer = buffer[local_batch_size:]

    def _iter_no_pad(self):
        global_chunk_size = self.config.batch_size * self.config.seq_length
        if self.config.use_data_sharded_loader:
            local_batch_size = self.config.batch_size // self._node_info['dp_node_size']
        else:
            local_batch_size = self.config.batch_size
        chunk_size = local_batch_size * self.config.seq_length

        token_buffer = []
        loss_mask_buffer = []
        vision_mask_buffer = []

        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for tokens, loss_masks, vision_masks, keep, loc, index in self.parallel_example_iterator():
            if not keep:
                continue
            self._file_loc = loc
            self._index = index
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            vision_mask_buffer.extend(vision_masks)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += global_chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size:]
                average_throughput = global_chunk_size / np.mean(step_times)
                accumulated_throughput = (
                    (self._total_tokens - start_tokens) / (time.time() - start_time)
                )
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': self._total_tokens,
                    'dataset_accumulated_tps': accumulated_throughput,
                    'dataset_average_tps': average_throughput,
                }
                batch = {
                    'input_tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        local_batch_size, -1
                    ),
                    'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                        local_batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                        local_batch_size, -1
                    ),
                    'input_vision_masks': np.array(vision_mask_buffer[:chunk_size], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    'target_vision_masks': np.array(vision_mask_buffer[1:chunk_size + 1], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                }

                if self.config.use_data_sharded_loader and not self.config.return_local_batch:
                    mesh = self._node_info['mesh']
                    sp_nodes_size = max(1, mesh.shape['sp'] // jax.local_device_count())
                    sp_nodes_rank = jax.process_index() % sp_nodes_size
                    assert self.config.seq_length % sp_nodes_size == 0, (self.config.seq_len, sp_nodes_size)
                    seq_chunk_size = self.config.seq_length // sp_nodes_size
                    batch = {k: v[:, sp_nodes_rank*seq_chunk_size:(sp_nodes_rank+1)*seq_chunk_size] for k, v in batch.items()}
                    batch = host_local_array_to_global_array(batch, self._node_info['mesh'], PS(('dp', 'fsdp'), 'sp'))

                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]
                vision_mask_buffer = vision_mask_buffer[chunk_size:]
    

    def _make_callback(self, v):
        return lambda index: v[index]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
        )

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))
        self._index = state_dict.get('index', self.config.example_index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)
        self._total_tokens = state_dict.get('total_tokens', self.config.tokens_count_at_start)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self._tokenizer)
