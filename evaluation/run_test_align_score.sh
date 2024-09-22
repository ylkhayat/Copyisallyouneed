#!/bin/bash

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Please set it before running the script."
    exit 1
fi

base_dir=$1
device=cuda:$CUDA_VISIBLE_DEVICES

for dir in "$base_dir"/7_*; do
  if [ -d "$dir" ]; then
    echo "Running tests for $dir"
    python align_score/test.py --test_dir $dir --test_type greedy --device $device
    python align_score/test.py --test_dir $dir --test_type nucleus_sampling --device $device
  fi
done
