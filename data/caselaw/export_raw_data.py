import subprocess
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
    'veterans': f'{base_path}/veterans'
}

parser = argparse.ArgumentParser(description='Process raw data.')
parser.add_argument('mode', choices=MODES.keys(), help='Mode of the data processing')
parser.add_argument('variant', nargs='?', choices=['cited', 'uncited', ''], help='Variant of the data processing',  default='')
args = parser.parse_args()

mode = args.mode
variant = args.variant
main_dir = f'{MODES[mode]}/{variant}'

raw_txt_path = os.path.join(main_dir, 'text.txt')
base_data_processed_path = os.path.join(main_dir, 'original_base_data.txt')
train_processed_path = os.path.join(main_dir, 'base_data.txt')
test_processed_path = os.path.join(main_dir, 'test.txt')

result = subprocess.run(['wc', '-l', raw_txt_path], stdout=subprocess.PIPE, text=True)
total_lines = int(result.stdout.split()[0])

train_threshold = int(total_lines * train_split_threshold)

train_test_file = open(base_data_processed_path, 'w')
train_file = open(train_processed_path, 'w')
test_file = open(test_processed_path, 'w')


pbar = tqdm(open(raw_txt_path, 'r'))
line_counter = 0
for line in pbar:
    if line:
        if line.startswith('|='):
            case_counter = 0
        else:
            text_parts = line.strip().split('\t')
            volume = text_parts[-1]
            text = '\t'.join(text_parts[:-1])
            label = f'{volume},{case_counter}'
            train_test_file.write(text + '\t' + label + '\n')
            if line_counter < train_threshold:
                train_file.write(text + '\t' + label + '\n')
            else:
                test_file.write(text + '\t' + label + '\n')
            case_counter += 1
            line_counter += 1
    pbar.set_description(f'[!] case id: {case_counter}')

print(f"Processed {line_counter} lines!")
train_file.close()
test_file.close()