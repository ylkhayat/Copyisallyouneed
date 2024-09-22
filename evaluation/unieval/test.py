import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils import convert_to_json
from metric.evaluator import get_evaluator
import numpy as np
from tqdm import tqdm
import json
import torch
import argparse


# config_path = "/srv/elkhyo/Copyisallyouneed/copyisallyouneed/config/copyisallyouneed.yaml"
# with open(config_path, 'r') as file:
#     config = yaml.safe_load(file)

# prefix_encoder_tokenizer_en = config['prefix_encoder_tokenizer']['en']

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
    
    task = 'summarization'
    evaluator = get_evaluator(task)
    dataset = load_result(test_file)
    
    with torch.no_grad():
        scores = []
        src_list = []
        output_list = []
        ref_list = []

        for prefix, reference, result in tqdm(dataset):
            src_list.append(prefix)
            ref_list.append(reference)
            output_list.append(result)

        data = convert_to_json(output_list=output_list, src_list=src_list, ref_list=ref_list)
        eval_scores = evaluator.evaluate(data)
    
    results = {}
    if os.path.exists(result_path):
        with open(result_path, 'r') as file:
            results = json.load(file)
            
    sum_scores = {key: 0 for key in eval_scores[0].keys()}
    for score in eval_scores:
        for key, value in score.items():
            sum_scores[key] += value
    mean_scores = {key: round(sum_scores[key] / len(eval_scores), 4) for key in sum_scores}

    if model_version not in results:
        results[model_version] = {}
    if method_name not in results[model_version]:
        results[model_version][method_name] = {}
    results[model_version][method_name]['unieval_sum'] = mean_scores
    with open(result_path, 'w') as file:
        json.dump(results, file, indent=4)
    # print('Results for', args['test_dir'], 'UniEval:', round(np.mean(scores), 4), 'Dataset size', len(dataset), file=open(f'{args["test_dir"]}_unieval_sum_result.txt', 'w'))
