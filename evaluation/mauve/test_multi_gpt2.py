from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
import ipdb
import mauve
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='1.0')
    parser.add_argument("--device", type=int)
    return parser.parse_args()

def load_result(path):
    with open(path) as f:
        test_set = json.load(f)
        dataset = []
        for item in tqdm(test_set):
            prefix = item['prefix']
            reference = item['reference'].strip()
            result = item['text']

            reference_ids = vocab.encode(reference, add_special_tokens=False)
            result_ids = vocab.encode(result, add_special_tokens=False)

            # min_length = min(len(reference_ids), len(result_ids))
            # reference_ids, result_ids = reference_ids[:min_length], result_ids[:min_length]
            # reference = vocab.decode(reference_ids)
            # result = vocab.decode(result_ids)

            reference = prefix + ' ' + reference
            result = prefix + ' ' + result
            if len(reference_ids) > 0:
                dataset.append((reference, result))
    print(f'[!] collect {len(dataset)} samples')
    return dataset

if __name__ == "__main__":
    args = vars(parse_config())
    vocab = AutoTokenizer.from_pretrained('gpt2-large')
    random_seeds = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0]
    scores = []
    for seed in tqdm(random_seeds):
        # path = f'raw_files/random_runs/{args["dataset"]}_gpt2_result_nucleus_sampling_on_wikitext103_testset_seed_{seed}.json'
        path = f'raw_files/random_runs_retro_wikitext103/{args["dataset"]}_retro_result_sampling_random_seed_{seed}.json'
        dataset = load_result(path)
        out = mauve.compute_mauve(
            p_text=[i[0] for i in dataset], 
            q_text=[i[1] for i in dataset], 
            device_id=args['device'], 
            max_text_length=512, 
            verbose=False, 
            mauve_scaling_factor=2, 
            featurize_model_name='gpt2-large',
        )
        scores.append(out.mauve)
    print(path + ':', scores)
    scores = round(np.mean(scores), 4)
    print(f'Results for gpt2 {args["dataset"]}', 'MAUVE:', scores, file=open(f'{path}_result.txt', 'w'))
    print(f'Results for gpt2 {args["dataset"]}', 'MAUVE:', scores)
