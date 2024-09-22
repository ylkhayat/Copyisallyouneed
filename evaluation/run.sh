#!/bin/bash


cd "/srv/elkhyo/Copyisallyouneed/evaluation" || exit

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Please set it before running the script."
    exit 1
fi

test_dir=$1
device=cuda:$CUDA_VISIBLE_DEVICES

basename_test_dir=$(basename "$test_dir")

cd "/srv/elkhyo/Copyisallyouneed/evaluation" || exit

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Please set it before running the script."
    exit 1
fi

test_dir=$1
device=cuda:$CUDA_VISIBLE_DEVICES

basename_test_dir=$(basename "$test_dir")
parent_dir=$(dirname "$test_dir")

# unieval
python unieval/test.py --test_dir "$test_dir" --test_type "greedy" 
python unieval/test.py --test_dir "$test_dir" --test_type "nucleus_sampling" 

# align_score
python align_score/test.py --test_dir "$test_dir" --test_type "greedy" --device "$device"
python align_score/test.py --test_dir "$test_dir" --test_type "nucleus_sampling" --device "$device"

# rouge
python rouge/test.py --test_dir "$test_dir" --test_type "greedy" 
python rouge/test.py --test_dir "$test_dir" --test_type "nucleus_sampling" 

# bert_score
python bert_score/test.py --test_dir "$test_dir" --test_type "greedy" --device "$device"
python bert_score/test.py --test_dir "$test_dir" --test_type "nucleus_sampling" --device "$device"