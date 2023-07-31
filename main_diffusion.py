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
from train_eval.train_eval_contrast import *
from util.diffusion import *


torch.multiprocessing.set_sharing_strategy('file_system')


def get_args_parser():
    parser = argparse.ArgumentParser('Set Denoising Diffusion Learning', add_help=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='only evaluate model on validation set')

    # Model parameters
    parser.add_argument('--resume', type=str, default='', help="initial weights path")  # weights/model_940.pth
    parser.add_argument('--time-step', type=int, default=64, help="number of time steps to predict")
    parser.add_argument('--hpy', type=str, default='cfg/cfg.yaml', help="hyper parameters path")
    parser.add_argument('--positional-embedding', default='learned', choices=('sine', 'learned'),
                        help="type of positional embedding to use on top of the image features")
    parser.add_argument('--sync-bn', action='store_true', help='enabling apex sync BN.')
    parser.add_argument('--freeze-encoder', action='store_true', help="freeze the encoder")
    parser.add_argument('--save-best', action='store_true', help="save best model")

    # Optimization parameters
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--lrf', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--epochs', default=30000, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # dataset parameters
    parser.add_argument('--train-path', default='path/Data_files_with_label/train/', help='train dataset path')
    parser.add_argument('--val-path', default='path/Data_files_with_label/val/', help='val dataset path')
    parser.add_argument('--cache-data', default=True, type=bool, help='cache data for faster training')
    parser.add_argument('--train-split', default=0.8, type=float, help='train split')
    parser.add_argument('--output-dir', default='weights/diffusion', help='path where to save, empty for no saving')

    # distributed training parameters
    parser.add_argument('--world_size', default=8, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_false', help='use mixed precision')
    
    return parser


def main(args):
    utils.init_distributed_mode(args)
    if args.amp:
        assert torch.backends.cudnn.enabled, \
            "NVIDIA Apex extension is not available. Please check environment and/or dependencies."
        assert torch.backends.cudnn.version() >= 7603, \
            "NVIDIA Apex extension is outdated. Please update Apex extension."
    if args.rank in [-1, 0]:
        print(args)

        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(comment=args.name)

    device = torch.device(args.device)
    if "cuda" not in args.device:
        raise EnvironmentError("CUDA is not available, please check your environment settings.")
    
    wdir = args.output_dir + os.sep  # weights dir
    best = wdir + "best.pt"
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # Remove previous results
    if args.rank in [-1, 0]:
        for f in glob.glob(results_file) + glob.glob("tmp.pk"):
            os.remove(f)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load hyper parameters
    with open(args.hpy) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # dataset generate
    print("Contrastive dataset generating...")
    dataset_train = Diffusion_multi_env(data_folder_path=args.train_path, 
                                        cache=args.cache_data,
                                        past_steps=cfg["contrast_sequence_length"],
                                        future_steps=args.time_step,
                                        train=True,
                                        temp_dim=cfg["Temporal_dim"])
    dataset_val = Diffusion_multi_env(data_folder_path=args.val_path,
                                      cache=args.cache_data,
                                      past_steps=cfg["contrast_sequence_length"],
                                      future_steps=args.time_step,
                                      train=False,
                                      temp_dim=cfg["Temporal_dim"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set Denoising Diffusion Learning', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    