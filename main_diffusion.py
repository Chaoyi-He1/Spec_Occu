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
from datasets.dataset import Contrastive_data, Contrastive_data_multi_env
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
    parser.add_argument('--time-step', type=int, default=12, help="number of time steps to predict")
    parser.add_argument('--hpy', type=str, default='cfg/cfg.yaml', help="hyper parameters path")
    parser.add_argument('--positional-embedding', default='sine', choices=('sine', 'learned'),
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
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # dataset parameters
    parser.add_argument('--train-path', default='path/train/', help='train dataset path')
    parser.add_argument('--val-path', default='path/val/', help='val dataset path')
    parser.add_argument('--cache-data', default=True, type=bool, help='cache data for faster training')
    parser.add_argument('--train-split', default=0.8, type=float, help='train split')
    parser.add_argument('--output-dir', default='weights', help='path where to save, empty for no saving')

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_false', help='use mixed precision')
    
    return parser


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set Denoising Diffusion Learning', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    