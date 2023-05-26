import torch
import torch.nn as nn
from util.misc import *
from util.Contrastive import *
from typing import Iterable


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    scaler=None):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for i, (past, future) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        past = past.to(device)
        future = future.to(device)
        
        # Compute the output
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred_embed = model(past)    #pred_embed: [time_step, B, feature_dim]
            infoNCELoss, cls_pred = criterion(pred_embed, future)
            
        # reduce losses over all GPUs for logging purposes
        infoNCELoss_reduced = reduce_loss(infoNCELoss)
