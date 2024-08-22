#!/bin/bash

python test_phrase_split.py --dataset caselaw --recall_method dpr --chunk_size 128 --dataset_path /srv/elkhyo/data/iterations/sample_0_1/uncited
python stat_test.py --recall_method dpr --chunk_length 128 --dataset_path /srv/elkhyo/data/iterations/sample_0_1/uncited
