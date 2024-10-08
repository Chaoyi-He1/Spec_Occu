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
from datasets.dataset import Diffusion_multi_env, Special_case_multi_env
from models.diffusion import *
from models.contrastive_model import *
from models.Temp_to_Freq_model import *
from train_eval.evaluation import *
from util.diffusion import *
from util.Temp_to_Freq import Temporal_Freq_Loss


torch.multiprocessing.set_sharing_strategy('file_system')


def get_args_parser():
    parser = argparse.ArgumentParser('Set Denoising Diffusion Learning', add_help=False)
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='only evaluate model on validation set')

    # Model parameters
    parser.add_argument('--resume', type=str, default='weights/fine_tune/cache/model_452.pth', help="initial weights path")  # weights/model_940.pth
    parser.add_argument('--encoder-path', type=str, default='', help="encoder path")
    parser.add_argument('--T2F-path', type=str, default='weights/T2F/conv/', help="T2F path")
    parser.add_argument('--diffusion-path', type=str, default='weights/diffusioni/', help="diffusion path")
    parser.add_argument('--time-step', type=int, default=32, help="number of time steps to predict")
    parser.add_argument('--hpy', type=str, default='cfg/cfg.yaml', help="hyper parameters path")
    parser.add_argument('--positional-embedding', default='learned', choices=('sine', 'learned'),
                        help="type of positional embedding to use on top of the image features")
    parser.add_argument('--sync-bn', action='store_true', help='enabling apex sync BN.')
    parser.add_argument('--freeze-encoder', type=bool, default=True, help="freeze the encoder")
    parser.add_argument('--freeze-diffusion', action='store_true', help="freeze the diffusion model")
    parser.add_argument('--save-best', action='store_true', help="save best model")

    # Optimization parameters
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lrf', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # dataset parameters
    parser.add_argument('--train-path', default='path/Data_files_with_label/train/', help='train dataset path')
    parser.add_argument('--val-path', default='path/Data_files_with_label/val/', help='val dataset path')
    parser.add_argument('--cache-data', default=True, type=bool, help='cache data for faster training')
    parser.add_argument('--train-split', default=0.8, type=float, help='train split')
    parser.add_argument('--output-dir', default='weights/fine_tune', help='path where to save, empty for no saving')

    # distributed training parameters
    parser.add_argument('--world_size', default=8, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_false', help='use mixed precision')
    
    return parser


def main(args):
    print(args)
    
    device = torch.device(args.device)
    if "cuda" not in args.device:
        raise EnvironmentError("CUDA is not available, please check your environment settings.")
    
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load hyper parameters
    with open(args.hpy) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # check if the sequence length from args is consistent with the cfg
    assert args.time_step == cfg["T2F_encoder_sequence_length"], \
        "The sequence length from args is not consistent with the cfg T2F_encoder_sequence_length."

    # dataset generate
    print("Diffusion dataset generating...")
    dataset_train = Special_case_multi_env(data_folder_path=args.train_path, 
                                           cache=args.cache_data,
                                           past_steps=cfg["contrast_sequence_length"],
                                           future_steps=args.time_step,
                                           train=False,
                                           temp_dim=cfg["Temporal_dim"])
    dataset_val = Special_case_multi_env(data_folder_path=args.val_path,
                                         cache=args.cache_data,
                                         past_steps=cfg["contrast_sequence_length"],
                                         future_steps=args.time_step,
                                         train=False,
                                         temp_dim=cfg["Temporal_dim"])
    
    sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, args.batch_size, drop_last=True)

    # dataloader
    print("Diffusion model dataloader generating...")
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print(f"Using {nw} dataloader workers every process")
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_train, 
                                                    collate_fn=dataset_train.collate_fn , num_workers=nw)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_sampler=batch_sampler_val,
                                                  collate_fn=dataset_val.collate_fn, num_workers=nw)
    
    # model
    print("Diffusion model generating...")
    variance = VarianceSchedule(num_steps=cfg["diffusion_num_steps"],
                                mode=cfg["diffusion_schedule_mode"])
    variance.to(device)

    diffusion_model = build_diffusion_model(diffnet="TransformerConcatLinear", cfg=cfg)
    diffusion_model.to(device)

    diffusion_util = Diffusion_utils(var_sched=variance)
    diffusion_util.to(device)

    encoder = build_feature_extractor(cfg=cfg)
    encoder.to(device)
    
    T2F_model = build_T2F(cfg=cfg)
    T2F_model.to(device)
    
    # loss function
    T2F_criterion = Temporal_Freq_Loss(time_step_weights=cfg["time_step_weights"])
    T2F_criterion.to(device)
    
    # load previous model if resume training
    start_epoch = 0
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.resume.endswith(".pth"):
        print("Resuming training from %s" % args.resume)
        ckpt = torch.load(args.resume, map_location='cpu')

        try:
            ckpt["diffusion_model"] = {k: v for k, v in ckpt["diffusion_model"].items() 
                                       if diffusion_model.state_dict()[k].numel() == v.numel()}
            diffusion_model.load_state_dict(ckpt["diffusion_model"], strict=False)

            ckpt["encoder"] = {k: v for k, v in ckpt["encoder"].items()
                               if encoder.state_dict()[k].numel() == v.numel()}
            encoder.load_state_dict(ckpt["encoder"], strict=False)
            
            ckpt["diffusion_util"] = {k: v for k, v in ckpt["diffusion_util"].items()
                                      if diffusion_util.state_dict()[k].numel() == v.numel()}
            diffusion_util.load_state_dict(ckpt["diffusion_util"], strict=False)
            
            ckpt["T2F_model"] = {k: v for k, v in ckpt["T2F_model"].items()
                                 if T2F_model.state_dict()[k].numel() == v.numel()}
            T2F_model.load_state_dict(ckpt["T2F_model"], strict=False)

        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                % (args.weights, args.hyp, args.weights)
            raise KeyError(s) from e

        print("Loading model from: ", args.resume, "finished.")
    
    if args.encoder_path.endswith(".pth") and not args.diffusion_path.endswith(".pth"):
        print("Loading encoder from: ", args.encoder_path)
        ckpt = torch.load(args.encoder_path, map_location='cpu')
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
    
    if args.diffusion_path.endswith(".pth"):
        print("Loading diffusion model from: ", args.diffusion_path)
        ckpt = torch.load(args.diffusion_path, map_location='cpu')
        try:
            ckpt["diffusion_model"] = {k: ckpt["diffusion_model"][k]
                                       for k, v in diffusion_model.state_dict().items()
                                       if ckpt["diffusion_model"][k].numel() == v.numel()}
            diffusion_model.load_state_dict(ckpt["diffusion_model"], strict=False)
            
            ckpt["diffusion_util"] = {k: ckpt["diffusion_util"][k]
                                      for k, v in diffusion_util.state_dict().items()
                                      if ckpt["diffusion_util"][k].numel() == v.numel()}
            diffusion_util.load_state_dict(ckpt["diffusion_util"], strict=False)
            
            ckpt["encoder"] = {k: ckpt["encoder"][k]
                               for k, v in encoder.state_dict().items()
                               if ckpt["encoder"][k].numel() == v.numel()}
            encoder.load_state_dict(ckpt["encoder"], strict=False)
            
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                % (args.weights, args.hyp, args.weights)
            raise KeyError(s) from e
        del ckpt
        print("Loading diffusion model from: ", args.diffusion_path, "finished.")
    
    if args.T2F_path.endswith(".pth"):
        print("Loading T2F model from: ", args.T2F_path)
        ckpt = torch.load(args.T2F_path, map_location='cpu')
        try:
            ckpt["model"] = {k: ckpt["model"][k]
                                 for k, v in T2F_model.state_dict().items()
                                 if ckpt["model"][k].numel() == v.numel()}
            T2F_model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                % (args.weights, args.hyp, args.weights)
            raise KeyError(s) from e
        del ckpt
        print("Loading T2F model from: ", args.T2F_path, "finished.")

    # model info
    n_parameters, layers = 0, 0
    for p in diffusion_model.parameters():
        n_parameters += p.numel()
        layers += 1
    print('number of Diffusion params:', n_parameters)
    print('Diffusion Model Summary: %g layers, %g parameters' % 
          (layers, n_parameters))
    
    n_parameters, layers = 0, 0
    for p in encoder.parameters():
        n_parameters += p.numel()
        layers += 1
    print('number of Encoder params:', n_parameters)
    print('Encoder Model Summary: %g layers, %g parameters' %
          (layers, n_parameters))

    # start training
    print("Start testing...")
    output_dir = Path(results_file)
    best_ADE_loss, best_FDE_loss = float('inf'), float('inf')
    start_time = time.time()
    
    # Training set evaluation
    (_, _, _) = evaluate(encoder=encoder, diff_model=diffusion_model, 
                         T2F_model=T2F_model, diff_criterion=diffusion_util, 
                         T2F_criterion=T2F_criterion, data_loader=data_loader_train, 
                         device=device, scaler=scaler, is_train=True)
    # Visualize the results for training set

    # Testing set evaluation
    (_, _, _) = evaluate(encoder=encoder, diff_model=diffusion_model, 
                         T2F_model=T2F_model, diff_criterion=diffusion_util, 
                         T2F_criterion=T2F_criterion, data_loader=data_loader_val, 
                         device=device, scaler=scaler, is_train=False)
    # Visualize the results for testing set
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set Denoising Diffusion Learning', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    