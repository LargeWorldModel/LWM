# Data

We support two types of datasets: Huggingface dataset and JSON dataset. The dataset modules are implemented in the [data.py](/lwm/data.py) file.

Configuration requires dataset type, text processor, and dataset specific configurations.
The following is an example of using the Huggingface dataset to train a model:
```bash
python -m lwm.train \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.type='huggingface' \
    --train_dataset.huggingface_dataset.path='openwebtext'
```

In this example, we select the Huggingface dataset by specifying the `type` of
`train_dataset` to be `huggingface`. We then specify the path to the dataset,
which is `c4` in this case. The examples loaded from the dataset will be processed
by a TextProcessor, which is configured by the `text_processor` field.

The following options are supported for the dataset module:
* `type`: The type of the dataset. Supported values are `huggingface` and `json`.
* `text_processor`: The configuration of the TextProcessor used to process the
  loaded examples.
* `huggingface_dataset`: The configuration of the Huggingface dataset.
* `json_dataset`: The configuration of the JSON dataset.

For huggingface dataset, we expect loading examples from a Huggingface dataset.
* `path`: The path to the dataset. Same as the `path` argument in
  `datasets.load_dataset`.
* `name`: Name of the dataset within the path. Same as the `name` argument in
  `datasets.load_dataset`.
* `split`: The split of the dataset. Same as the `split` argument in
  `datasets.load_dataset`.
*  `streaming`: Whether to stream the dataset. Same as the `streaming` argument
   in `datasets.load_dataset`.
* `seq_length`: The length of the tokenized sequence.
* `batch_size`: Batch size of tokenized examples.

For JSON dataset, we expect loading examples from a text file, where each line where each line represents a
JSON encoded dictionary. Here are the configurable options for JSON dataset:
* `path`: Path to the text file. The file can be located on the local file system
  or on Google Cloud Storage bucket.
* `seq_length`: The length of the tokenized sequence.
* `batch_size`: Batch size of tokenized examples.
* `start_seek_loc`: The starting seek location in the file. This is useful when
  you want to resume training from a particular location in the file.
* `index_at_start`: The counting index at the beginning. This is useful to
  keep the index count when resuming from a particular location in the file.
  Note that this is only for logging purpose, and does not affect the actual
  examples starting from. To start from a different example in the dataset,
  you should use the `start_seek_loc` option.
* `tokenizer_processes`: The number of processes to use for tokenization.
  Tokenization is done in parallel to speed up the loading process.

A JSON dataset can be generated as follows:
```python
from datasets import load_dataset
import json
from multiprocessing import Pool, cpu_count

dataset = load_dataset("openwebtext")

split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

def save_split(split):
    with open(f"openwebtext_{split}.jsonl", "w") as f:
        for example in split_dataset[split]:
            json.dump({"text": example["text"]}, f)
            f.write("\n")

with Pool(cpu_count()) as p:
    p.map(save_split, ["train", "val"])
```

This generates two files, `openwebtext_train.jsonl` and `openwebtext_val.jsonl`, which can be used as the dataset for training. Both files contain a single field, `text`, which is the text to be processed by the model.
For example, to train a model using the `openwebtext_train.jsonl` file, you can use the following command:
```bash
python -m lwm.train \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.type='json' \
    --train_dataset.json_dataset.path='openwebtext_train.jsonl' \
```

For vision-langauge training, we recommend using the JSON dataset, as it allows you to pre-tokenize vision (images and videos), and load the tokenized vision along with the text.

Each loaded example is a dictionary, which will be processed by a TextProcessor


## Text Processor
We use the `TextProcessor` class to process the loaded examples from a dataset. This allows us to flexibly process various formats.
Each input example is a dictionary of multiple text fields. The TextProcessor will
process text fields according to its configurations, and return the final tokens.

Here are the configurable options for TextProcessor:
* `fields`: A comma separated list of text fields to process.
* `fields_from_example`: Whether to use the keys of the input example as the
  text fields to process. If this option is set, the `fields` argument will
  be ignored.
* `subfield_separator`: The text separator to use when concatenating subfields
  of a texts.
* `add_eos_token`: Whether to add an EOS token to the end of the text.
* `prepend_text`: The text to prepended to the beginning.

The most important configuration for TextProcessor is the `fields` argument. It
is a comma separated list of text fields to process. Each field consists of one
or more subfields, which are separated by a `+`. Each subfield represent a key
used to extract the text from the input example dictionary. The TextProcessor
joins the extracted subfields of texts with the `subfield_separator` in the text
level and then tokenize the joined text. Finally, the TextProcessor will concatenate
the tokenized text fields at the token level, and add the EOS token if specified.

Other than the keys in the input example, you can also use the following special
keys to indicate a special token for a text field:
* `<|bos|>`: Beginning of sentence token.
* `<|eos|>`: End of sentence token.

For each text field, you can encapulate the subfields with `[]` to specify that
the loss should not be computed for this field. Doing so will make the loss
masks to be 0 for this field. This is useful when you want to use the text field
as a prompt for the model.


To give a concrete example, if the input example looks like this:
```python
{
    'vision': 'VQ tokens of a picture of a cat',
    'question': 'what is the color of the cat',
    'answer': 'the color of the cat is yellow',
}
```
To use the `vision` and `question` as the input text, and `answer` as the target,
we can specify the following configuration for the `fields` argument:
```
[vision+question],answer
```

The `vision+question` indicates that the `vision` and `question` should be joined
togather with the `subfield_separator`, which is a space by default. The `[]`
indicates that the loss should not be computed for this field. The `answer` field
is then concatenated at the token level, where the loss will be computed.

