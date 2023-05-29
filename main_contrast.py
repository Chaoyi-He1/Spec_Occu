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
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torch.multiprocessing

import util.misc as utils


torch.multiprocessing.set_sharing_strategy('file_system')


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='only evaluate model on validation set')

    # Model parameters
    parser.add_argument('--resume', type=str, default='', help="initial weights path")
    parser.add_argument('--time-step', type=int, default=12, help="number of time steps to predict")
    parser.add_argument('--hpy', type=str, default='cfgs/hyper_params.yaml', help="hyper parameters path")
    parser.add_argument('--positional-embedding', default='sine', choices=('sine', 'learned'), help="type of positional embedding to use on top of the image features")
    parser.add_argument('--freeze-encoder', action='store_true', help="freeze the encoder")
    parser.add_argument('--savebest', action='store_true', help="save best model")

    # Optimization parameters
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lf', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # dataset parameters
    parser.add_argument('--data_path', default='path/to/train.mat', help='dataset path')
    parser.add_argument('--output_dir', default='weights/output', help='path where to save, empty for no saving')

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_false', help='use mixed precision')
    
    return parser