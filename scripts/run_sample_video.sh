#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

export llama_tokenizer_path=""
export vqgan_checkpoint=""
export lwm_checkpoint=""

# Relevant params
# --temperature_*: Temperature that is applied to each of the logits
# --top_k_*: Only sample from the tokens with the top k logits
# --cfg_scale_*: Classifier-free guidance scale for each modality
# --n_frames: Number of frames to generate

python3 -u -m lwm.vision_generation \
    --prompt='Fireworks over the city' \
    --output_file='fireworks.mp4' \
    --temperature_image=1.0 \
    --temperature_video=1.0 \
    --top_k_image=8192 \
    --top_k_video=1000 \
    --cfg_scale_image=5.0 \
    --cfg_scale_video=1.0 \
    --vqgan_checkpoint="$vqgan_checkpoint" \
    --n_frames=8 \
    --mesh_dim='!1,1,-1,1' \
    --dtype='fp32' \
    --load_llama_config='7b' \
    --update_llama_config="dict(sample_mode='vision',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)" \
    --load_checkpoint="params::$lwm_checkpoint" \
    --tokenizer.vocab_file="$llama_tokenizer_path"
read
