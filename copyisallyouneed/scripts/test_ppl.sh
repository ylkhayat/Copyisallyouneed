#!/bin/bash

model=$1
CUDA_VISIBLE_DEVICES=0 python test_ppl.py --dataset wikitext103 --model $model
