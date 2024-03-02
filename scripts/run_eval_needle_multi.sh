#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

export llama_tokenizer_path=""
export lwm_text_checkpoint=""
# jsonl file containing text for haystack. Each line should be a json
# with a single key "text" containing the text.
export haystack_file=""
export output_file=""

python3 -u scripts/eval_needle_multi.py \
    --mesh_dim='!1,1,-1,1' \
    --dtype='fp32' \
    --load_llama_config='7b' \
    --update_llama_config="dict(theta=10000000,max_sequence_length=131072,scan_attention=True,scan_query_chunk_size=1024,scan_key_chunk_size=1024,scan_mlp=True,scan_mlp_chunk_size=1024,scan_layers=True)" \
    --load_checkpoint="params::$lwm_text_checkpoint" \
    --tokenizer.vocab_file="$llama_tokenizer_path" \
    --max_tokens_per_batch=5000 \
    --output_file="$output_file" \
    --haystack_file="$haystack_file" \
    --context_lengths_min=1000 \
    --context_lengths_max=10000 \
    --n_context_length_intervals=10 \
    --n_document_depth_intervals=10 \
    --n_needles_total=4 \
    --n_needles_retrieve=2 \
    --n_rounds=10
read
