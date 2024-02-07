import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import *
from typing import Iterable


def cls_criterion(pred: torch.Tensor, index: torch.Tensor):
    """
    Calculate the classification loss and accuracy.
    :param pred: Tensor [B, C], C is the number of classes
    :param index: Tensor [B], the index of the true class, 
        index is the label of differen wireless environment
        
    :return: cls_loss: Tensor, the classification loss
    :return: cls_acc: Tensor, the classification accuracy
    """
    cls_loss = F.cross_entropy(pred, index)
    cls_pred = pred.argmax(dim=1)
    cls_acc = (cls_pred == index).float().mean()
    return cls_loss, cls_acc
    

def train_one_epoch(model: torch.nn.Module, criterion: cls_criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0.1,
                    scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_acc', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for i, (past, future, index) in enumerate(metric_logger.log_every(data_loader, 4, header)):
        past = past.to(device)
        # future = future.to(device)
        index = index.to(device)
        
        # Compute the output
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred_embed_future = model(past)    #pred_embed_future: [time_step, B, feature_dim]
            cls_loss, cls_pred = criterion(pred_embed_future, index)

        if torch.isnan(cls_loss):
            raise ValueError('NaN loss detected')
            
        # reduce losses over all GPUs for logging purposes
        cls_loss_reduced = reduce_loss(cls_loss)
        cls_pred_reduced = reduce_loss(cls_pred)

        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(cls_loss).backward()
        else:
            cls_loss.backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Update metric
        metric_logger.update(loss=cls_loss_reduced.item())
        metric_logger.update(class_acc=cls_pred_reduced.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # return summary for logging
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: cls_criterion, 
             data_loader: Iterable, device: torch.device):
    model.eval()

    #Metrics
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('class_accuracy', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Test:'

    for i, (past, future, index) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        past = past.to(device)
        # future = future.to(device)
        index = index.to(device)

        # When using the CPU, skip GPU-related instructions
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        
        # Compute the output
        pred_embed = model(past)    #pred_embed: [time_step, B, feature_dim]
        cls_loss, cls_pred = criterion(pred_embed, index)

        if torch.isnan(cls_loss):
            raise ValueError('NaN loss detected')
        
        # reduce losses over all GPUs for logging purposes
        cls_loss_reduced = reduce_loss(cls_loss)
        cls_pred_reduced = reduce_loss(cls_pred)

        # Update metric
        metric_logger.update(loss=cls_loss_reduced.item())
        metric_logger.update(class_accuracy=cls_pred_reduced.item())
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
