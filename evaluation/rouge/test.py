import os
import numpy as np
from tqdm import tqdm
import json
import torch
import argparse
import evaluate


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default='copyisallyouneed_result.json')
    parser.add_argument("--test_type", type=str, default='greedy')
    return parser.parse_args()

def load_result(path):
    with open(path) as f:
        test_set = json.load(f)
        dataset = []
        for item in tqdm(test_set):
            prefix = item['prefix']
            reference = item['reference']
            result = item['text']
            dataset.append((prefix, reference, result))
    print(f'[!] collect {len(dataset)} samples')
    return dataset


if __name__ == "__main__":
    args = vars(parse_config())
    parent_dir = os.path.dirname(args['test_dir'])
    model_version = os.path.basename(args['test_dir'])
    result_path = f"{parent_dir}/result.json"
    test_file = f"{args['test_dir']}/{args['test_type']}.json"
    method_name = args['test_type']
    rouge = evaluate.load('rouge')
    dataset = load_result(test_file)
    
    with torch.no_grad():
        claims_list = []
        reference_list = []
        predictions_list = []

        for prefix, reference, result in tqdm(dataset):
            reference_list.append(reference)
            predictions_list.append(result)
        scores = rouge.compute(predictions=predictions_list,
                            references=reference_list,
                            use_aggregator=True)
    
    results = {}
    if os.path.exists(result_path):
        with open(result_path, 'r') as file:
            results = json.load(file)
            
    if model_version not in results:
        results[model_version] = {}
    if method_name not in results[model_version]:
        results[model_version][method_name] = {}
    results[model_version][method_name]['rouge'] = scores
    with open(result_path, 'w') as file:
        json.dump(results, file, indent=4)
