#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

export llama_tokenizer_path=""
export vqgan_checkpoint=""
export lwm_checkpoint=""
export input_file=""

python3 -u -m lwm.vision_chat \
    --prompt="What is the video about?" \
    --input_file="$input_file" \
    --vqgan_checkpoint="$vqgan_checkpoint" \
    --mesh_dim='!1,-1,32,1' \
    --dtype='fp32' \
    --load_llama_config='7b' \
    --max_n_frames=8 \
    --update_llama_config="dict(sample_mode='text',theta=50000000,max_sequence_length=131072,use_flash_attention=False,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,remat_attention='',scan_mlp=False,scan_mlp_chunk_size=2048,remat_mlp='',remat_block='',scan_layers=True)" \
    --load_checkpoint="params::$lwm_checkpoint" \
    --tokenizer.vocab_file="$llama_tokenizer_path" \
2>&1 | tee ~/output.log
read
