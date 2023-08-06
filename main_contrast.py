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
from models.contrastive_model import *
from train_eval.train_eval_contrast import *
from util.Contrastive import ContrastiveLoss


torch.multiprocessing.set_sharing_strategy('file_system')


def get_args_parser():
    parser = argparse.ArgumentParser('Set CPC Contrastive Learning', add_help=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='only evaluate model on validation set')

    # Model parameters
    parser.add_argument('--resume', type=str, default='weights/model_105.pth', help="initial weights path")  # weights/model_940.pth
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
    parser.add_argument('--output-dir', default='weights/contrast', help='path where to save, empty for no saving')

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
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
    
    wdir = "weights" + os.sep  # weights dir
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
    dataset_train = Contrastive_data_multi_env(data_folder_path=args.train_path, 
                                               cache=args.cache_data,
                                               past_steps=cfg["contrast_sequence_length"],
                                               future_steps=args.time_step,
                                               train=True,
                                               num_frames_per_clip=cfg["num_frames_per_clip"],
                                               temp_dim=cfg["Temporal_dim"])
    dataset_val = Contrastive_data_multi_env(data_folder_path=args.val_path, 
                                             cache=args.cache_data,
                                             past_steps=cfg["contrast_sequence_length"],
                                             future_steps=args.time_step,
                                             train=False,
                                             num_frames_per_clip=cfg["num_frames_per_clip"],
                                             temp_dim=cfg["Temporal_dim"])
    if args.distributed:
        # sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        # sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
        sampler_train = Custom_DistributedSampler(dataset_train, shuffle=True)
        sampler_val = Custom_DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, args.batch_size, drop_last=True)
    
    # dataloader
    print("Contrastive dataloader generating...")
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of wor-kers
    if args.rank in [-1, 0]:
        print('Using %g dataloader workers' % nw)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_train, 
                                                    collate_fn=dataset_train.collate_fn, num_workers=nw)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_sampler=batch_sampler_val,
                                                  collate_fn=dataset_val.collate_fn, num_workers=nw)
    
    # model
    print("Model generating...")
    model = build_contrastive_model(cfg=cfg, timestep=args.time_step, pos_type=args.positional_embedding)
    model.to(device)
    if args.rank in [-1, 0]:
        tb_writer.add_graph(model, torch.randn((args.batch_size, 
                                                cfg["contrast_sequence_length"], 
                                                cfg["in_channels"], 
                                                cfg["num_frames_per_clip"],
                                                cfg["Temporal_dim"]), 
                                              device=device, dtype=torch.float), use_strict_trace=False)
    
    # load previous model if resume training
    start_epoch = 0
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.resume.endswith(".pth"):
        print("Resuming training from %s" % args.resume)
        ckpt = torch.load(args.resume, map_location='cpu')

        try:
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() 
                             if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                % (args.weights, args.hyp, args.weights)
            raise KeyError(s) from e
        
        if args.rank in [-1, 0]:
            # load results
            if ckpt.get("training_results") is not None:
                with open(results_file, "w") as file:
                    file.write(ckpt["training_results"])  # write results.txt
        
        # epochs
        start_epoch = ckpt["epoch"] + 1
        if args.epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (args.resume, ckpt['epoch'], args.epochs))
        if args.amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        del ckpt
        print("Loading model from: ", args.resume, "finished.")
    
    # freeze encoder if args.freeze_encoder is true
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
        print("Encoder frozen.")
    
    # synchronize batch norm layers if args.sync_bn is true
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print("Sync BatchNorm layers.")

    # distributed model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # model info
    params_to_optimize = []
    n_parameters, layers = 0, 0
    for p in model.parameters():
        n_parameters += p.numel()
        layers += 1
        if p.requires_grad:
            params_to_optimize.append(p)
    print('number of params:', n_parameters)
    print('Model Summary: %g layers, %g parameters' % (layers, n_parameters))

    # learning rate scheduler setting
    # After using DDP, the gradients on each device will be averaged, so the learning rate needs to be enlarged
    args.lr *= max(1., args.world_size * args.batch_size / 64)
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  # do not move
    
    # loss function
    criterion = ContrastiveLoss(time_step_weights=cfg["time_step_weights"])

    # start training
    print("Start training...")
    output_dir = Path(args.output_dir)
    best_loss, best_acc = float('inf'), 0.0
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + start_epoch):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # train
        train_loss_dict, step_acc = train_one_epoch(model=model, data_loader=data_loader_train, 
                                                    criterion=criterion, optimizer=optimizer, 
                                                    device=device, epoch=epoch, scaler=scaler,
                                                    steps=args.time_step)
        print(str(step_acc))
        scheduler.step()

        # validation
        test_loss_dict, val_step_acc = evaluate(model=model, data_loader=data_loader_val, 
                                                criterion=criterion, device=device, 
                                                steps=args.time_step)
        print(str(val_step_acc))
        
        # write results
        log_stats = {**{f'train_{k}': v for k, v in train_loss_dict.items()},
                     **{f'test_{k}': v for k, v in test_loss_dict.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                    }
        if args.output_dir and utils.is_main_process():
            with (results_file).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # write tensorboard
        if utils.is_main_process():
            if tb_writer:
                items = {
                    **{f'train_{k}': v for k, v in train_loss_dict.items()},
                    **{f'test_{k}': v for k, v in test_loss_dict.items()},
                }
                for k, v in items.items():
                    tb_writer.add_scalar(k, v, epoch)
        
        # save model
        if args.save_best:
            # save best model
            if test_loss_dict["loss"] < best_loss and \
               test_loss_dict["acc"] > best_acc:
                best_loss = test_loss_dict["loss"]
                best_acc = test_loss_dict["acc"]

                utils.save_on_master({
                    "epoch": epoch,
                    "model": model_without_ddp.state_dict() if args.distributed else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "lr_scheduler": scheduler.state_dict(),
                }, best)
        else:
            # save latest model
            digits = len(str(args.epochs))
            utils.save_on_master({
                "epoch": epoch,
                "model": model_without_ddp.state_dict() if args.distributed else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "lr_scheduler": scheduler.state_dict(),
            }, os.path.join(args.output_dir, 'model_{}.pth'.format(str(epoch).zfill(digits))))
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
