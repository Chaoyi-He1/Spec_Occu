import torch
import torch.nn as nn
from util.misc import *
from util.Contrastive import *
from typing import Iterable


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 1.0,
                    scaler=None):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for i, (past, future) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        past = past.to(device)
        future = future.to(device)
        
        # Compute the output
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred_embed_future = model(past)    #pred_embed_future: [time_step, B, feature_dim]
            infoNCELoss, cls_pred = criterion(pred_embed_future, future, model)
            
        # reduce losses over all GPUs for logging purposes
        infoNCELoss_reduced = reduce_loss(infoNCELoss)
        cls_pred_reduced = reduce_loss(cls_pred)

        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(infoNCELoss_reduced).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            infoNCELoss_reduced.backward()
            optimizer.step()
        
        # Update metric
        metric_logger.update(loss=infoNCELoss_reduced.item())
        metric_logger.update(class_error=cls_pred_reduced.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # return summary for logging
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, 
             data_loader: Iterable, device: torch.device):
    model.eval()
    criterion.eval()

    #Metrics
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Test:'

    for i, (past, future) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        past = past.to(device)
        future = future.to(device)

        # When using the CPU, skip GPU-related instructions
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        
        # Compute the output
        pred_embed = model(past)    #pred_embed: [time_step, B, feature_dim]
        infoNCELoss, cls_pred = criterion(pred_embed, future)
        
        # reduce losses over all GPUs for logging purposes
        infoNCELoss_reduced = reduce_loss(infoNCELoss)
        cls_pred_reduced = reduce_loss(cls_pred)

        # Update metric
        metric_logger.update(loss=infoNCELoss_reduced.item())
        metric_logger.update(class_error=cls_pred_reduced.item())
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
