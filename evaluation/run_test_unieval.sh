#!/bin/bash

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Please set it before running the script."
    exit 1
fi

base_dir=$1

for dir in "$base_dir"/7_*; do
  if [ -d "$dir" ]; then
    if [ ! -f "$dir/results.json" ]; then
      echo "Running tests for $dir"
      python unieval/test.py --test_dir $dir --test_type greedy
      python unieval/test.py --test_dir $dir --test_type nucleus_sampling
    else
      echo "Skipping $dir, results.json already exists"
    fi
  fi
done
