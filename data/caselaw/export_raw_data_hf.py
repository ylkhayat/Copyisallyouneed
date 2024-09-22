

from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
import sys
import os
import argparse

sys.path.append('..')

cache_dir = "/srv/elkhyo/data"
case_counter = 0
train_split_threshold = 0.8


base_path = '/srv/elkhyo/data/iterations'

MODES = {
    'all': f'{base_path}/all',
    'sample_5': f'{base_path}/sample_5',
    'sample_0_1': f'{base_path}/sample_0_1',
    'most_cited_3': f'{base_path}/most_cited_3',
    'supreme_court': f'{base_path}/supreme_court',
    'veterans': f'{base_path}/veterans',
    'veterans_hf': f'{base_path}/veterans_hf'
}

parser = argparse.ArgumentParser(description='Process raw data.')
parser.add_argument('mode', choices=MODES.keys(), help='Mode of the data processing')
parser.add_argument('variant', nargs='?', help='Variant of the data processing',  default='')
args = parser.parse_args()

mode = args.mode
variant = args.variant
main_dir = f'{MODES[mode]}/{variant}'

train_processed_path = os.path.join(main_dir, 'base_data.txt')
test_processed_path = os.path.join(main_dir, 'test.txt')

os.makedirs(main_dir, exist_ok=True)

train_file = open(train_processed_path, 'w')
test_file = open(test_processed_path, 'w')

dataset = load_dataset("TeraflopAI/Caselaw_Access_Project", split='train', cache_dir="/srv/elkhyo/.cache/huggingface")
dataset = dataset.filter(lambda x: 'Veterans' in x['reporter'])
dataset = dataset.train_test_split(test_size=0.1)

def convert_date(example):
    date_str = example['decision_date']
    try:
        sortable_date = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        try:
            sortable_date = datetime.strptime(date_str, '%Y').strftime('%Y-%m-%d')
        except ValueError:
            sortable_date = '0000-00-00'
    example['sortable_date'] = sortable_date
    return example

dataset = dataset.map(convert_date)
dataset = dataset.sort("sortable_date")

case_counter = 0
texts = []
for mode in ['train', 'test']:
    file_write = train_file if mode == 'train' else test_file
    pbar = tqdm(dataset[mode], total=len(dataset[mode]))
    file_write = train_file if mode == 'train' else test_file
    for item in pbar:
        text = item['text'].strip()
        text = text.replace('\n', ' ')
        text = item['decision_date'].strip() + '\t' + text
        if text:
            label = f'{case_counter}'
            file_write.write(text + '\t' + label + '\n')
        case_counter += 1
        pbar.set_description(f'[!] document id: {case_counter}')


train_file.close()
test_file.close()