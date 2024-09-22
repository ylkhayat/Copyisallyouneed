import os
import numpy as np
from tqdm import tqdm
import json
import torch
import argparse

from alignscore import AlignScore



# config_path = "/srv/elkhyo/Copyisallyouneed/copyisallyouneed/config/copyisallyouneed.yaml"
# with open(config_path, 'r') as file:
#     config = yaml.safe_load(file)

# prefix_encoder_tokenizer_en = config['prefix_encoder_tokenizer']['en']

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default='copyisallyouneed_result.json')
    parser.add_argument("--test_type", type=str, default='greedy')
    parser.add_argument("--device", type=str, default='cuda:0')
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
    
    dataset = load_result(test_file)
    
    with torch.no_grad():
        claims_list = []
        context_list = []

        for prefix, reference, result in tqdm(dataset):
            context = prefix + ' ' + reference
            context_list.append(context)
            claims_list.append(result)
            
        scorer = AlignScore(model='roberta-base', batch_size=32, device=args['device'], ckpt_path='/srv/elkhyo/models/', evaluation_mode='nli_sp')
        score = scorer.score(contexts=context_list, claims=claims_list)
    
    results = {}
    if os.path.exists(result_path):
        with open(result_path, 'r') as file:
            results = json.load(file)
            
    if model_version not in results:
        results[model_version] = {}
    if method_name not in results[model_version]:
        results[model_version][method_name] = {}
    results[model_version][method_name]['align_score'] = score
    with open(result_path, 'w') as file:
        json.dump(results, file, indent=4)
