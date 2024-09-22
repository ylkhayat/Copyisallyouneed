from header import *
from dataloader import *
from models import *
from config import *
import sys
sys.path.append('../data/')
from dpr_caselaw import Retriever

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset_path', default='', type=str)
    parser.add_argument('--model_path', default='', type=str)
    parser.add_argument('--decoding_method', type=str)
    parser.add_argument('--prefix_length', type=int, default=1024)
    parser.add_argument('--recall_topk', type=int, default=20)
    parser.add_argument('--model', type=str, default='copyisallyouneed')
    parser.add_argument('--dataset', default='caselaw', type=str)
    return parser.parse_args()

def main_generation(**args):
    dataset_path = args['dataset_path']
    prefix_length = args['prefix_length']
    retriever = Retriever(f"{dataset_path}/base_data_{prefix_length}.txt", 200, dataset_path)
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(args['model_path'])
    print(f'[!] init model over')

    collection = []
    with open(f'{dataset_path}/test.txt') as f:
        # collect the valid prefixes
        texts = []
        for line in tqdm(f.readlines()):
            ids = agent.model.tokenizer.encode(line, add_special_tokens=False)
            prefix, reference = ids[:prefix_length], ids[prefix_length:]
            if len(prefix) == prefix_length:
                prefix = agent.model.tokenizer.decode(prefix)
                reference = agent.model.tokenizer.decode(reference)
                texts.append((prefix, reference))
        print(f'[!] collect {len(texts)} valid samples which have at least {prefix_length} tokens in prefix')

        for prefix, reference in tqdm(texts):
            text, candidates, time_cost = agent.generate_one_sample(prefix, retriever, decoding_method=args["decoding_method"], top_k=0, top_p=0.95, temp=1., get_time_cost=True)
            collection.append({
                'prefix': prefix, 
                'reference': reference, 
                'text': text, 
                'phrases': candidates,
                'time_cost': time_cost
            })
    return collection

if __name__ == "__main__":
    args = vars(parser_args())
    result = main_generation(**args)
    dataset_path = args['dataset_path']
    model_file_name = os.path.basename(args['model_path']).replace('.pt','').replace('best_','')
    output_dir = f"{dataset_path}/{model_file_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/{args["decoding_method"]}.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
