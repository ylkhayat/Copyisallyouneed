#!/bin/bash

mode=$1
variant=$2

dataset_path=/srv/elkhyo/data/iterations/$mode/$variant

worker=(0)
for i in ${worker[@]}
do
    echo "worker $i begin"
    python phrase_split.py --worker_id $i --dataset caselaw --recall_method dpr --chunk_size 128 --dataset_path $dataset_path
    python stat.py --recall_method dpr --chunk_length 128 --dataset_path $dataset_path
done
