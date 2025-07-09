import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)

print(f"[GPU {args.local_rank}] CUDA: {torch.cuda.get_device_name(args.local_rank)} available = {torch.cuda.is_available()}")
