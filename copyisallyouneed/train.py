from header import *
from dataloader import *
from models import *
from config import *
import subprocess

local_rank = int(os.environ['LOCAL_RANK'])
print(f'[!] local rank: {local_rank}')

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--total_workers', type=int)
    parser.add_argument('--model_version', type=int, default=1)
    return parser.parse_args()

def test_model(save_path, **args):
    dataset_basename = os.path.basename(args["dataset_path"])

    try:
        command_type = "prepare_test"
        new_command = f"/srv/elkhyo/Copyisallyouneed/copyisallyouneed/scripts/copyisallyouneed_test.sh {save_path} veterans {dataset_basename}"
        enqueue_command = [
            "python", "/home/elkhyo/commands/enqueue_command.py",
            command_type, str(args['model_version']), new_command, str(args['chunk_length'])
        ]
        subprocess.run(enqueue_command, capture_output=True, text=True)
    except Exception as e:
        print(f'[!] Failed to append commands to the file: {e}')
    

def main(**args):
    torch.cuda.empty_cache()
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    args['global_rank'] = dist.get_rank()
    print(f'[!] global rank: {args["global_rank"]}')

    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    train_data, train_iter, sampler = load_dataset(args)
    if local_rank == 0:
        sum_writer = SummaryWriter(
            log_dir=f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/{args["model_version"]}',
        )
    else:
        sum_writer = None
        
    args['warmup_step'] = int(args['warmup_ratio'] * args['total_step'])
    agent = load_model(args)
    current_step, over_train_flag = 0, False
    pbar = tqdm(total=args['total_step'], initial=current_step)
    sampler.set_epoch(0)    # shuffle for DDP
    if agent.load_last_step:
        current_step = agent.load_last_step + 1
        print(f'[!] load latest step: {current_step}')
        pbar.update(current_step)
    for _ in range(1000):
        for batch in train_iter:
            agent.train_model(
                batch, 
                recoder=sum_writer, 
                current_step=current_step, 
                pbar=pbar
            )
            if args['global_rank'] == 0 and current_step % args['save_every'] == 0 and current_step > 0:
                save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{args["model_version"]}_{current_step}.pt'
                agent.save_model_long(save_path, current_step)
                test_model(save_path, **args)
            current_step += 1
            if current_step > args['total_step']:
                over_train_flag = True
                break
        if over_train_flag:
            break
    if sum_writer:
        sum_writer.close()

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
