#!/bin/bash

recall_method=$1
worker_id=$2
chunk_length=$3
dataset_path=$4
echo $chunk_length
# python stat.py --worker_id $worker_id --recall_method $recall_method --chunk_length $chunk_length 

# test statistic
python stat.py --recall_method $recall_method --chunk_length $chunk_length --dataset_path --dataset_path $dataset_path
