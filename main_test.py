import argparse
import datetime
import glob
import json
import math
import os
import random
import time
import pickle
from pathlib import Path
import tempfile

import yaml
import torch
import torch.distributed as dist
import numpy as np
from util.misc import torch_distributed_zero_first
from util.distributed_util import Custom_DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torch.multiprocessing

import util.misc as utils
from datasets.dataset import Diffusion_multi_env
from models.diffusion import *
from models.contrastive_model import *
from train_eval.train_eval_diffusion import *
from util.diffusion import *
from util.Temp_to_Freq import *


def get_args_parser():
    parser = argparse.ArgumentParser('Set arguments for evaluation the spectrum prediction model')
    parser.add_argument("--device", default='cuda', type=str, help="device to use for training / testing")
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    
    # Model parameters
    parser.add_argument('--hpy', type=str, default='cfg/cfg.yaml', help="hyper parameters path")
    parser.add_argument('--Temporal_model', type=str, default='weights/T2F/model_449.pth', help="Temporal model")
    parser.add_argument('--diffusion_model', type=str, default='weights/diffusion/model_449.pth', help="Diffusion model")
    
    # dataset parameters
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--val-path', default='path/val/', help='val dataset path')
    parser.add_argument('--cache-data', default=True, type=bool, help='cache data for faster training')
    parser.add_argument('--train-split', default=0.8, type=float, help='train split')
    parser.add_argument('--output-dir', default='weights/contrast', help='path where to save, empty for no saving')
    
    return parser

def main(args):
    print(args)
    
    device = torch.device(args.device)
    if "cuda" in args.device:
        torch.cuda.set_device(0)
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # load hyper parameters
    with open(args.hpy) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    # dataset generate
    print("Testing dataset generating...")
    dataset_test = Diffusion_multi_env(data_folder_path=args.val_path,
                                       cache=args.cache_data,
                                       past_steps=cfg["contrast_sequence_length"],
                                       future_steps=args.time_step,
                                       train=False,
                                       temp_dim=cfg["Temporal_dim"])
    
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, 
                                                       args.batch_size, drop_last=True)
    
    # dataloader
    print("Diffusion model dataloader generating...")
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print(f"Using {nw} dataloader workers every process")
    
        
        
        