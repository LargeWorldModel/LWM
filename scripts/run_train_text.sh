#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

export llama_tokenizer_path="LargeWorldModel/LWM-Text-1M"
export dataset_path=""
export output_dir=""

export project_id='lwm'
export experiment_note=''
export experiment_id='example-text-train'

# mesh_dim: dp, fsdp, tp, sp
python3 -u -m lwm.train \
    --modality='text' \
    --mesh_dim='!1,-1,2,2' \
    --dtype='fp32' \
    --total_steps=200 \
    --log_freq=1 \
    --save_model_freq=0 \
    --save_milestone_freq=10 \
    --load_llama_config='debug' \
    --update_llama_config="dict(theta=10000,max_sequence_length=2048,scan_attention=True,scan_query_chunk_size=256,scan_key_chunk_size=512,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=1024,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)" \
    --tokenizer="$llama_tokenizer_path" \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=8e-5 \
    --optimizer.adamw_optimizer.end_lr=8e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=5 \
    --optimizer.adamw_optimizer.lr_decay_steps=200 \
    --use_data_sharded_loader=True \
    --train_dataset.type='json' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.json_dataset.path="$dataset_path" \
    --train_dataset.json_dataset.seq_length=2048 \
    --train_dataset.json_dataset.batch_size=1024 \
    --train_dataset.json_dataset.tokenizer_processes=16 \
    --train_dataset.json_dataset.use_data_sharded_loader=True \
    --checkpointer.save_optimizer_state=True \
    --autoresume=False \
    --logger.append_uuid=False \
    --logger.online=False \
    --logger.project_id="$project_id" \
    --logger.experiment_id="$experiment_id" \
    --logger.experiment_note="$experiment_note" \
    --logger.output_dir="$output_dir" \
    --logger.wandb_dir="$HOME/experiment_output/$project_id"
read
