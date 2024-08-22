from tqdm import tqdm
from torch.cuda.amp import autocast
import ipdb
import mauve
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import argparse

def get_features(model, ids, ids_mask):
    length = ids_mask.sum(dim=-1) - 1
    # features = model(input_ids=ids, output_hidden_states=True).hidden_states[-1][range(len(ids)), length, :]    # [B, E]
    features = model(input_ids=ids, output_hidden_states=True).hidden_states[-1][range(len(ids)), 0, :]    # [B, E]
    return features.cpu()

def get_vocab_and_model(path):
    model= ElectraForPreTraining.from_pretrained("google/electra-large-discriminator")
    vocab= ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
    model.eval()
    model.cuda()
    return vocab, model

def convert_to_batch(vocab, lists):
    ids = []
    for item in lists:
        tokens = vocab.encode(item, add_special_tokens=False)
        tokens = torch.LongTensor(tokens[-512:])
        ids.append(tokens)
    ids = pad_sequence(ids, batch_first=True, padding_value=vocab.pad_token_id)
    mask = generate_mask(ids, pad_token_idx=vocab.eos_token_id)
    ids, mask = to_cuda(ids, mask)
    return ids, mask

def to_cuda(*args):
    '''map the tensor on cuda device'''
    if not torch.cuda.is_available():
        return args
    tensor = []
    for i in args:
        i = i.cuda()
        tensor.append(i)
    return tensor

def generate_mask(ids, pad_token_idx=0):
    '''generate the mask matrix of the ids, default padding token idx is 0'''
    mask = torch.ones_like(ids)
    mask[ids == pad_token_idx] = 0.
    return mask

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default='gpt2_result.json')
    parser.add_argument("--device", type=int)
    return parser.parse_args()

def load_result(path):
    with open(path) as f:
        test_set = json.load(f)
        dataset = []
        for item in tqdm(test_set):
            prefix = item['prefix']
            reference = item['reference']
            result = item['text']
            prefix = prefix.replace('<unk>', '[UNK]')
            reference = reference.replace('<unk>', '[UNK]')
            result = result.replace('<unk>', '[UNK]')
            
            reference_ids = vocab.encode(reference, add_special_tokens=False)

            # remove the \n and unk
            # try:
            #     reference_ids.remove(50118)
            #     reference_ids.remove(3)
            # except:
            #     pass

            if len(reference_ids) <= 130: 
                # reference_ids = reference_ids[:120]
                # reference = vocab.decode(reference_ids)
                dataset.append((reference, result))
    print(f'[!] collect {len(dataset)} samples')
    return dataset

if __name__ == "__main__":
    args = vars(parse_config())
    vocab, model = get_vocab_and_model('roberta-base')
    batch_size = 32
    dataset = load_result(args["test_path"])
    with torch.no_grad():
        gt_f, pre_f = [], []
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            ids, mask = convert_to_batch(vocab, [i[0] for i in batch])
            gt_f.append(get_features(model, ids, mask))
            ids, mask = convert_to_batch(vocab, [i[1] for i in batch])
            pre_f.append(get_features(model, ids, mask))
        gt_f = torch.cat(gt_f).numpy()
        pre_f = torch.cat(pre_f).numpy()
        out = mauve.compute_mauve(
            p_features=gt_f,
            q_features=pre_f,
            device_id=args['device'],
            mauve_scaling_factor=1, 
        )
    print('Results for', args['test_path'], 'MAUVE:', out.mauve, 'Dataset size', len(dataset), file=open(f'{args["test_path"]}_result.txt', 'w'))
