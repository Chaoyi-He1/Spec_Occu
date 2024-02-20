import torch
import torch.nn as nn
from util.misc import *
from util.Contrastive import *
from typing import Iterable


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0.05,
                    scaler=None, steps: int = 12):
    model.train()
    criterion.train()
    Step_Predict = Step_Prediction(num_steps=steps)
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_acc', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for i, (past, future, index) in enumerate(metric_logger.log_every(data_loader, 4, header)):
        past = past.to(device)
        future = future.to(device)
        index = index.to(device)
        
        # Compute the output
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred_embed_future = model(past)    #pred_embed_future: [time_step, B, feature_dim]
            infoNCELoss, cls_pred, steps_cls_pred = criterion(pred_embed_future, future, index, model)

        if torch.isnan(infoNCELoss):
            raise ValueError('NaN loss detected')
            
        # reduce losses over all GPUs for logging purposes
        infoNCELoss_reduced = reduce_loss(infoNCELoss)
        cls_pred_reduced = reduce_loss(cls_pred)
        steps_cls_pred_reduced = reduce_loss(steps_cls_pred)

        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(infoNCELoss).backward()
        else:
            infoNCELoss.backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Update metric
        metric_logger.update(loss=infoNCELoss_reduced.item())
        metric_logger.update(class_acc=cls_pred_reduced.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        Step_Predict.update(steps_cls_pred_reduced)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # return summary for logging
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, Step_Predict


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, 
             data_loader: Iterable, device: torch.device, steps: int = 12):
    model.eval()
    criterion.eval()

    #Metrics
    Step_Predict = Step_Prediction(num_steps=steps)
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('class_accuracy', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Test:'

    for i, (past, future, index) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        past = past.to(device)
        future = future.to(device)

        # When using the CPU, skip GPU-related instructions
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        
        # Compute the output
        pred_embed = model(past)    #pred_embed: [time_step, B, feature_dim]
        infoNCELoss, cls_pred, steps_cls_pred = criterion(pred_embed, future, index, model)

        if torch.isnan(infoNCELoss):
            raise ValueError('NaN loss detected')
        
        # reduce losses over all GPUs for logging purposes
        infoNCELoss_reduced = reduce_loss(infoNCELoss)
        cls_pred_reduced = reduce_loss(cls_pred)
        steps_cls_pred_reduced = reduce_loss(steps_cls_pred)

        # Update metric
        Step_Predict.update(steps_cls_pred_reduced)
        metric_logger.update(loss=infoNCELoss_reduced.item())
        metric_logger.update(class_accuracy=cls_pred_reduced.item())
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, Step_Predict
