#!/bin/bash

# dude, what the fuck !
export NCCL_IB_DISABLE=1

cd "/srv/elkhyo/Copyisallyouneed/copyisallyouneed" || exit

CONFIG_FILE="config/variants.yaml"
TRAIN_FILE="train.py"
# ========== metadata ========== #
version=$1

root_dir=$(shyaml get-value root_dir < "$CONFIG_FILE")
description=$(shyaml get-value variants."$version".description < "$CONFIG_FILE")
dataset_name=$(shyaml get-value variants."$version".dataset_name < "$CONFIG_FILE")
model_name=$(shyaml get-value variants."$version".model_name < "$CONFIG_FILE")
data_root_dir=$(shyaml get-value variants."$version".data_root_dir < "$CONFIG_FILE")
port=$(shyaml get-value variants."$version".port < "$CONFIG_FILE")
# ========== metadata ========== #

# Check if any variable is undefined
if [ -z "$root_dir" ] || [ -z "$description" ] || [ -z "$dataset_name" ] || [ -z "$model_name" ] || [ -z "$data_root_dir" ] || [ -z "$port" ]; then
  echo "Error: One or more required variables are not defined in the YAML configuration."
  exit 1
fi

echo "[!] Running training for model version '$version'"
echo "[!] Dataset: '$dataset_name', Model_name: '$model_name', Port: '#$port'"
echo "[!] Description: '$description', Data Root Dir: '$data_root_dir'"

# backup
recoder_file=$root_dir/rest/$dataset_name/$model_name/recoder_$version.txt

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Please set it before running the script."
    exit 1
fi

gpu_ids=(${CUDA_VISIBLE_DEVICES//,/ })
echo "[!] GPU IDs: ${gpu_ids[*]}"

echo "[!] Write running log into recoder file: $recoder_file"
mv "$root_dir"/ckpt/"$dataset_name"/"$model_name"/*_"$version".pt "$root_dir"/bak/"$dataset_name"/"$model_name"
# delete the previous tensorboard file
rm "$root_dir"/rest/"$dataset_name"/"$model_name"/"$version"/* 
rm -rf "$root_dir"/rest/"$dataset_name"/"$model_name"/"$version" 

CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=${#gpu_ids[@]} --max-restarts=3 --master_addr 127.0.0.1 --master_port "$port" "$TRAIN_FILE" \
    --dataset "$dataset_name" \
    --dataset_path "$data_root_dir" \
    --model "$model_name" \
    --multi_gpu "$gpu_ids" \
    --total_workers ${#gpu_ids[@]} \
    --model_version "$version"