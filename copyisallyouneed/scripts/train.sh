#!/bin/bash

# dude, what the fuck !
export NCCL_IB_DISABLE=1

# ========== metadata ========== #
dataset=$1
model=$2
port=${3:-28444} 
# cuda=$3 
# ========== metadata ========== #

root_dir=$(cat config/base.yaml | shyaml get-value root_dir)
version=$(cat config/base.yaml | shyaml get-value version)
data_root_dir=$(cat config/base.yaml | shyaml get-value data_root_dir)

# backup
recoder_file=$root_dir/rest/$dataset/$model/recoder_$version.txt

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Please set it before running the script."
    exit 1
fi

gpu_ids=(${CUDA_VISIBLE_DEVICES//,/ })


echo "find root_dir: $root_dir"
echo "find version: $version"
echo "find data_root_dir: $data_root_dir"
echo "write running log into recoder file: $recoder_file"
mv $root_dir/ckpt/$dataset/$model/*_$version.pt $root_dir/bak/$dataset/$model
# delete the previous tensorboard file
rm $root_dir/rest/$dataset/$model/$version/* 
rm -rf $root_dir/rest/$dataset/$model/$version 


CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=${#gpu_ids[@]} --max-restarts=3 --master_addr 127.0.0.1 --master_port $port train.py \
    --dataset $dataset \
    --dataset_path $data_root_dir \
    --model $model \
    --multi_gpu $gpu_ids \
    --total_workers ${#gpu_ids[@]}