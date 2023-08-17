import torch
import torch.nn as nn
from util.misc import *
from util.diffusion import *
from typing import Iterable


def train_one_epoch(encoder: torch.nn.Module, model: torch.nn.Module, 
                    criterion: Diffusion_utils, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int, max_norm: float = 0, scaler=None):
    
    encoder.train()
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for _, (history, hist_labels,
            future, future_labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        torch.autograd.set_detect_anomaly(True)
        history = history.to(device)
        future = future.to(device)

        # Compute the output
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            features = encoder(history)
            BCELoss = criterion.get_loss(x_0=future, context=features, model=model)

        if torch.isnan(BCELoss):
            raise ValueError('NaN loss detected')
            
        # reduce losses over all GPUs for logging purposes
        BCELoss_reduced = reduce_loss(BCELoss)

        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(BCELoss).backward()
        else:
            BCELoss.backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Update metric
        metric_logger.update(loss=BCELoss_reduced.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(encoder: torch.nn.Module, model: torch.nn.Module, 
             criterion: Diffusion_utils, data_loader: Iterable, 
             device: torch.device, scaler=None, repeat=20):
    
    encoder.eval()
    model.eval()
    criterion.eval()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('ADE_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('FDE_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ADE_percentage', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('FDE_percentage', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Test:'

    for _, (history, hist_labels,
            future, future_labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        history = history.to(device)
        future = future.to(device)
        b, l, c, d = future.shape

        # Compute the output
        with torch.cuda.amp.autocast(enabled=scaler is not None), torch.no_grad():
            features = encoder(history)
            predict = criterion.sample(num_points=l, context=features, sample=repeat, bestof=False, step=10,
                                       model=model, point_dim=d, flexibility=0.0, ret_traj=False, sampling="ddpm")
        
        ADE, FDE, ADE_percents, FDE_percents = compute_batch_statistics(predict, future)
        # reduce losses over all GPUs for logging purposes
        ADE_reduced = reduce_loss(ADE)
        FDE_reduced = reduce_loss(FDE)
        ADE_percents_reduced = reduce_loss(ADE_percents)
        FDE_percents_reduced = reduce_loss(FDE_percents)

        # Update metric
        metric_logger.update(ADE_loss=ADE_reduced.mean().item())
        metric_logger.update(FDE_loss=FDE_reduced.mean().item())
        metric_logger.update(ADE_percentage=ADE_percents_reduced.mean().item())
        metric_logger.update(FDE_percentage=FDE_percents_reduced.mean().item())
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
