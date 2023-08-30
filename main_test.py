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
from models.Temp_to_Freq_model import *
from train_eval.train_eval_diffusion import *
from util.diffusion import *
from util.Temp_to_Freq import *


def get_args_parser():
    parser = argparse.ArgumentParser('Set arguments for evaluation the spectrum prediction model')
    parser.add_argument("--device", default='cuda', type=str, help="device to use for training / testing")
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    
    # Model parameters
    parser.add_argument('--hpy', type=str, default='cfg/cfg.yaml', help="hyper parameters path")
    parser.add_argument('--encoder-model', type=str, default='weights/contrast/model_449.pth', help="encoder path")
    parser.add_argument('--Temporal-model', type=str, default='weights/T2F/model_449.pth', help="Temporal model")
    parser.add_argument('--diffusion-model', type=str, default='weights/diffusion/model_449.pth', help="Diffusion model")
    parser.add_argument('--positional-embedding', default='sine', choices=('sine', 'learned'),
                        help="type of positional embedding to use on top of the image features")
    
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
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_sampler=batch_sampler_test,
                                                   collate_fn=dataset_test.collate_fn, num_workers=nw)
    
    # model
    print("model generating...")
    variance = VarianceSchedule(num_steps=cfg["diffusion_num_steps"],
                                mode=cfg["diffusion_schedule_mode"])
    variance.to(device)

    diffusion_model = build_diffusion_model(diffnet="TransformerConcatLinear", cfg=cfg)
    diffusion_model.to(device)

    diffusion_util = Diffusion_utils(var_sched=variance)
    diffusion_util.to(device)

    encoder = build_feature_extractor(cfg=cfg)
    encoder.to(device)
    
    Temp_2_Freq = build_T2F(cfg=cfg, pos_type=args.positional_embedding)
    Temp_2_Freq.to(device)
    
    # load model weights
    print("Loading Contrastive encoder weights from {}".
          format(args.encoder_model))
    ckpt = torch.load(args.encoder_model, map_location='cpu')
    try:
        ckpt["model"] = {k: ckpt["model"][k] 
                            for k, v in encoder.state_dict().items()
                            if ckpt["model"][k].numel() == v.numel()}
        encoder.load_state_dict(ckpt["model"], strict=False)
    except KeyError as e:
        s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
            % (args.weights, args.hyp, args.weights)
        raise KeyError(s) from e
    del ckpt
    print("Loading encoder from: ", args.encoder_path, "finished.")
    
    print("Loading Temporal model weights from {}".
          format(args.Temporal_model))
    ckpt = torch.load(args.Temporal_model, map_location='cpu')
    try:
        ckpt["model"] = {k: ckpt["model"][k] 
                            for k, v in Temp_2_Freq.state_dict().items()
                            if ckpt["model"][k].numel() == v.numel()}
        Temp_2_Freq.load_state_dict(ckpt["model"], strict=False)
    except KeyError as e:
        s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
            % (args.weights, args.hyp, args.weights)
        raise KeyError(s) from e
    del ckpt
    print("Loading Temporal model from: ", args.Temporal_model, "finished.")
    
    print("Loading Diffusion model weights from {}".
            format(args.diffusion_model))
    ckpt = torch.load(args.diffusion_model, map_location='cpu') 
    
        
        