#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

export llama_tokenizer_path=""
export lwm_text_checkpoint=""
# jsonl file containing text for haystack. Each line should be a json
# with a single key "text" containing the text.
export haystack_file=""
export output_file=""

python3 -u scripts/eval_needle.py \
    --mesh_dim='!1,-1,4,1' \
    --dtype='fp32' \
    --load_llama_config='7b' \
    --update_llama_config="dict(theta=10000000,max_sequence_length=131072,use_flash_attention=False,scan_attention=True,scan_query_chunk_size=1024,scan_key_chunk_size=1024,scan_mlp=True,scan_mlp_chunk_size=1024,scan_layers=True)" \
    --load_checkpoint="params::$lwm_text_checkpoint" \
    --tokenizer.vocab_file="$llama_tokenizer_path" \
    --max_tokens_per_batch=5000 \
    --output_file="$output_file" \
    --haystack_file="$haystack_file" \
    --context_lengths_min=1000 \
    --context_lengths_max=10000 \
    --n_context_length_intervals=20 \
    --n_document_depth_intervals=20 \
    --n_rounds=3
read
