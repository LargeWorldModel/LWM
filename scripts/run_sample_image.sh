#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

export llama_tokenizer_path=""
export vqgan_checkpoint=""
export lwm_checkpoint=""

python3 -u -m lwm.vision_generation \
    --prompt='Fireworks over the city' \
    --output_file='fireworks.png' \
    --temperature_image=1.0 \
    --top_k_image=8192 \
    --cfg_scale_image=5.0 \
    --vqgan_checkpoint="$vqgan_checkpoint" \
    --n_frames=1 \
    --mesh_dim='!1,1,-1,1' \
    --dtype='fp32' \
    --load_llama_config='7b' \
    --update_llama_config="dict(sample_mode='vision',theta=50000000,max_sequence_length=32768,use_flash_attention=True,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)" \
    --load_checkpoint="params::$lwm_checkpoint" \
    --tokenizer.vocab_file="$llama_tokenizer_path"
read
