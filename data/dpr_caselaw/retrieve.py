import os
import torch
import numpy as np
import random
import pickle
import ipdb
from torch.utils.data import dataset, dataloader
import argparse
from tqdm import tqdm
import torch.distributed as dist
from utils import *

local_rank = int(os.environ["LOCAL_RANK"])
def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pool_size', default=256, type=int)
    parser.add_argument('--chunk_size', default=256, type=int)
    parser.add_argument('--chunk_length', default=128, type=int)
    return parser.parse_args()

def load_datasets(path):
    datasets = {}
    datasets_counter = {}
    with open(path) as f:
        for line in tqdm(f.readlines()):
            items = line.strip().split('\t')
            document = '\t'.join(items[:-1])
            label = items[-1].strip()
            datasets[label] = document
            doc_label = label.split(',')[0]
            if doc_label in datasets_counter:
                datasets_counter[doc_label] += 1
            else:
                datasets_counter[doc_label] = 1
    print(f'[!] load {len(datasets)} samples') 
    return datasets, datasets_counter

def search_one_job(worker_id):
    label, embed = torch.load(f"{args['dataset_path']}/dpr_chunk_{worker_id}_0.pt")
    print(f'[!] load {len(label)} samples from dpr_chunk_{worker_id}_0.pt')
    
    searcher = Searcher('Flat', dimension=768, nprobe=1)
    searcher.load(f"{args['dataset_path']}/dpr_faiss.ckpt", f"{args['dataset_path']}/dpr_corpus.ckpt")
    searcher.move_to_gpu(device=local_rank)

    # search
    collection = []

    pbar = tqdm(total=len(embed))
    
    chunk_prefix_path = f'../{args["dataset"]}/dpr_search_chunk_{args["chunk_length"]}_{worker_id}.pkl'
    if args['dataset_path']:
        chunk_prefix_path = f"{args['dataset_path']}/dpr_search_chunk_{args['chunk_length']}_{worker_id}.pkl"
    counter = 0

    recall, acc, similarity = [], [], []
    for i in range(0, len(embed), args['batch_size']):
        sublabel = label[i:i+args['batch_size']]
        subembed = embed[i:i+args['batch_size']]
        result, distance = searcher._search(subembed.numpy(), topk=args['pool_size']+1)

        for l, rest, dist in zip(sublabel, result, distance):
            doc_label = l.split(',')[0]
            num = len(set([item for item in rest if doc_label == item.split(',')[0]]))
            # rest = [item for item in rest if l != item]
            recall.append(num/datasets_counter[doc_label])
            acc.append(num/args['pool_size'])
            collection.append((l, rest[1:]))
            similarity.append(dist[-1])
        pbar.update(len(sublabel))
        # if len(similarity) > 10000:
        #     break
    pickle.dump(collection, open(f'{chunk_prefix_path}', 'wb'))
    print(f'[!] recall: {round(np.mean(recall), 4)}; acc: {round(np.mean(acc), 4)}')
    print(f'[!] similarity: {round(np.mean(similarity), 4)}')
    print(f'[!] save data into {chunk_prefix_path}')

if __name__ == '__main__':
    args = vars(parser_args())
    if "dataset_path" in args and args["dataset_path"]:
        dataset_path = f"{args['dataset_path']}/base_data_{args['chunk_length']}.txt"
    else:
        dataset_path = f'../{args["dataset"]}/base_data_{args["chunk_length"]}.txt'
    datasets, datasets_counter = load_datasets(dataset_path)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    search_one_job(local_rank)

