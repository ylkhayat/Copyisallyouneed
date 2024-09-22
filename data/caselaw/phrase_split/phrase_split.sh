#!/bin/bash

mode=$1
variant=$2
chunk_size=$3

dataset_path=/srv/elkhyo/data/iterations/$mode/$variant

worker=(0)
for i in ${worker[@]}
do
    echo "worker $i begin"
    python phrase_split.py --worker_id $i --dataset caselaw --recall_method dpr --chunk_size $chunk_size --dataset_path $dataset_path
    python stat.py --recall_method dpr --chunk_length $chunk_size --dataset_path $dataset_path
done
