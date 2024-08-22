#!/bin/bash
export NCCL_IB_DISABLE=1
cuda=$1
gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 28205 encode_doc.py\
    --data_path ../lawmt_1024/base_data_128.txt \
    --batch_size 512 \
    --cut_size 500000

first_cuda_device=$(echo $cuda | cut -d',' -f1)
CUDA_VISIBLE_DEVICES=$first_cuda_device python build_index.py