import argparse
import chunk
import re
import os
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import threading
from utils import clean_line_train

def sentence_token_nltk(str):
    sent_tokenize_list = sent_tokenize(str)
    return sent_tokenize_list

def get_number_of_lines(file_path):
    result = os.popen(f'wc -l < {file_path}').read().strip()
    return int(result)

def update_progress_bar(file_path, pbar):
    number_of_lines = get_number_of_lines(file_path)
    pbar.total = number_of_lines
    pbar.refresh()


base_path = '/srv/elkhyo/data/iterations'
# Define modes
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
parser.add_argument('--mode', choices=MODES.keys(), help='Mode of the data processing')
parser.add_argument('--chunk_size', nargs='?', type=int, help='Chunk size for the data processing', default=128)
parser.add_argument('--variant', nargs='?', help='Variant of the data processing',  default='')
args = parser.parse_args()

mode = args.mode
variant = args.variant
main_dir = f'{MODES[mode]}/{variant}'

base_data_processed_path = os.path.join(main_dir, f'base_data.txt')
base_data_128_processed_path = os.path.join(main_dir, f'base_data_{args.chunk_size}.txt')

pbar = tqdm(open(base_data_processed_path, 'r'), total=0)

thread = threading.Thread(target=update_progress_bar, args=(base_data_processed_path, pbar))
thread.start()

chunks = 0
output_file = open(base_data_128_processed_path, 'w')


chunk_size = args.chunk_size

def write_line_to_file(writing):
    global chunks
    global counter
    text_to_write = ' '.join(writing)
    if len(writing) > chunk_size:
        print(f"[!] length of cache {len(writing)}")
    string = f"{text_to_write}\t'{line_counter},{counter}'\n"
    output_file.write(string)
    chunks += 1
    counter += 1
    return 1

from_hf = 'hf' in mode
line_counter = 0
for line in pbar:
    if from_hf:
        text = '\t'.join(line.strip().split('\t')[1:-1])
    else:
        text = '\t'.join(line.strip().split('\t')[:-1])
    cleaned_text = clean_line_train(text)
    sentences = sentence_token_nltk(cleaned_text)
    cache, counter = [], 0
    for sent in sentences:
        words = sent.split(' ')
        for word in words:
            cummulative_words_length = sum(len(word) for word in cache) + len(cache) + len(word) + 1
            if cummulative_words_length > chunk_size and len(cache) > 0:
                write_line_to_file(cache)
                cache = [word]
            else:
                cache.append(word)
    if len(cache) > 0:
        if len(cache) > chunk_size:
            print(f"[!] length of cache {len(cache)}")
            for i in range(0, len(cache), chunk_size):
                current_tokens_chunk = cache[i:i + chunk_size]
                write_line_to_file(current_tokens_chunk)
    line_counter += 1

print(f'[!] collect {chunks} chunks from the base_data.txt')

output_file.close()