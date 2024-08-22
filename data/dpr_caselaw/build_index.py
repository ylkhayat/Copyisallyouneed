from utils import *
import ipdb
import torch
from tqdm import tqdm
import joblib
import numpy as np
import glob
import re
import argparse


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset_path', default='ecommerce', type=str)
    return parser.parse_args()


def build_index(index_type, **args):
    embds, texts = [], []
    current_num = 0
    
    files = glob.glob(f"{args['dataset_path']}/dpr_chunk_*_*.pt")
    numbers = [(int(re.search('dpr_chunk_(\d+)_(\d+).pt', f).group(1)), int(re.search('dpr_chunk_(\d+)_(\d+).pt', f).group(2))) for f in files]
    max_gpu_number = max(numbers, key=lambda x: x[0])[0]
    chunks_per_gpu = {gpu: max([chunk for gpu_num, chunk in numbers if gpu_num == gpu]) + 1 for gpu in range(max_gpu_number + 1)}
    
    for i in tqdm(range(max_gpu_number + 1)):
        for idx in range(chunks_per_gpu.get(i, 0)):
            try:
                text, embed = torch.load(f"{args['dataset_path']}/dpr_chunk_{i}_{idx}.pt")
                print(f'[!] load dpr_chunk_{i}_{idx}.pt')
                current_num += len(embed)
            except Exception as error:
                print(error)
                break
            embds.append(embed.numpy())
            texts.extend(text)
            print(f'[!] collect embeddings: {current_num}')
    
    embds = np.concatenate(embds)
    searcher = Searcher(index_type, dimension=768)
    searcher._build(embds, texts, speedup=True)
    print(f'[!] train the searcher over')
    searcher.move_to_cpu()

    searcher.save(f"{args['dataset_path']}/dpr_faiss.ckpt", f"{args['dataset_path']}/dpr_corpus.ckpt")
    print(f'[!] save faiss index over')

if __name__ == "__main__":
    args = vars(parser_args())
    build_index('Flat', **args)