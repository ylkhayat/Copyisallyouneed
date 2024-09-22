#!/bin/bash

cd "/srv/elkhyo/Copyisallyouneed/copyisallyouneed" || exit

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Please set it before running the script."
    exit 1
fi

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model_path> [mode] [variant] [decoding_method]"
    echo "  model_path: /srv/elkhyo/ell.pt"
    echo "  mode: veterans"
    echo "  variant: cited, uncited, cited-0_1"
    exit 1
fi
model_path=$1
mode=$2
variant=${3-""}
prefix_length=${4-128}

dataset_path="/srv/elkhyo/data/iterations/$mode/$variant"
model_basename=$(basename "$model_path")
model_version=${model_basename#best_}
model_version=${model_version%.pt}
evaluations_dir="/srv/elkhyo/data/iterations/$mode/$variant/$model_version"

gpu_ids=(${CUDA_VISIBLE_DEVICES//,/ })

echo "[!] Start evaluation with (model version: $model_version) (GPUs: $gpu_ids) for both decoding methods."
echo "[!] Dataset path: $dataset_path"
echo "[!] Model path: $model_path"
echo "[!] Evaluations dir: $evaluations_dir"
echo "[!] Prefix length: $prefix_length"

if [ ! -f "$evaluations_dir/greedy.json" ]; then
    CUDA_VISIBLE_DEVICES=$gpu_ids python copyisallyouneed_test.py --dataset_path "$dataset_path" --model_path "$model_path" --decoding_method greedy --prefix_length "$prefix_length"
else
    echo "Skipping execution for greedy.json because it exists."
fi

if [ ! -f "$evaluations_dir/nucleus_sampling.json" ]; then
    CUDA_VISIBLE_DEVICES=$gpu_ids python copyisallyouneed_test.py --dataset_path "$dataset_path" --model_path "$model_path" --decoding_method nucleus_sampling --prefix_length "$prefix_length"
else
    echo "Skipping execution for nucleus_sampling.json because it exists."
fi

echo "[!] Evaluation finished. Append commands to the command file."
command_type="test"
model_version_first=$(echo "$model_version" | cut -d'_' -f1)
new_command="/srv/elkhyo/Copyisallyouneed/evaluation/run.sh $evaluations_dir"

nucleus_file="$evaluations_dir/nucleus_sampling.json"
greedy_file="$evaluations_dir/greedy.json"

if [[ -f "$nucleus_file" && -f "$greedy_file" ]]; then
    python /home/elkhyo/commands/enqueue_command.py "$command_type" "$model_version_first" "$new_command"
else
    echo "[!] Error: Required files nucleus_sampling.json and/or greedy.json do not exist in $evaluations_dir."
fi