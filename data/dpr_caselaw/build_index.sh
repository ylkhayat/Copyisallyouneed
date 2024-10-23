#!/bin/bash
export NCCL_IB_DISABLE=1

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Please set it before running the script."
    exit 1
fi

mode=$1
variant=${2:-""}

dataset_path=/srv/elkhyo/data/iterations/$mode
if [ -n "$variant" ]; then
  dataset_path="$dataset_path/$variant"
fi

gpu_ids=(${CUDA_VISIBLE_DEVICES//,/ })
CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 28205 encode_doc.py\
    --dataset_path $dataset_path \
    --batch_size 512 \
    --chunk_length 128 \
    --cut_size 500000 

CUDA_VISIBLE_DEVICES=$gpu_ids python build_index.py --dataset_path $dataset_path