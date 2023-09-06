import torch
import torch.nn as nn
from util.misc import *
from util.diffusion import *
from typing import Iterable
from itertools import chain
from util.Temp_to_Freq import F1_score
import torch.nn.functional as F
from util.diffusion import Diffusion_utils
from util.Temp_to_Freq import Temporal_Freq_Loss, F1_score


def train_one_epoch(encoder: torch.nn.Module, diff_model: torch.nn.Module, 
                    T2F_model: torch.nn.Module, diff_criterion: Diffusion_utils, 
                    T2F_criterion: Temporal_Freq_Loss, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int, max_norm: float = 0.1, scaler=None, 
                    freeze_encoder: bool =False):
    encoder.train()
    diff_model.train()
    T2F_model.eval()
    diff_criterion.train()
    T2F_criterion.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('F1_score', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for _, (history, hist_labels,
            future, future_labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        torch.autograd.set_detect_anomaly(True)
        history = history.to(device)
        future = future.to(device)
        future_labels = future_labels.to(device)
        
        # Compute the output
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
            features = encoder(history)
            predict = diff_criterion.sample_fine_tune(num_points=future.shape[1], context=features,
                                                      sample=1, bestof=False, step=10, flexibility=0.0, 
                                                      model=diff_model, point_dim=future.shape[-1], 
                                                      ret_traj=False, sampling="ddpm")
            predict_label = T2F_model(predict).detach()
            BCELoss, acc_steps = T2F_criterion(predict_label, future_labels)
            acc = acc_steps.mean()
            F1score = F1_score(predict_label, future_labels)
            
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            features = encoder(history)
            BCELoss = diff_criterion.get_loss_fine_tune(x_0=future, context=features, model=diff_model)
        
        loss = (BCELoss * (2 - torch.stack(F1score))).mean()
            # F1score = F1_score(predict_label, future_labels)

        if torch.isnan(loss):
            raise ValueError('NaN loss detected')
            
        # reduce losses over all GPUs for logging purposes
        loss_reduced = reduce_loss(loss)
        acc_reduced = reduce_loss(acc)

        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if max_norm > 0:
            params = diff_model.parameters() if freeze_encoder \
                else chain(encoder.parameters(), diff_model.parameters())
            torch.nn.utils.clip_grad_norm_(params, max_norm)
        
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            
        # params_updated = {name: param.clone() for name, param in model.named_parameters()}
        # for name, initial_param in initial_params.items():
        #     updated_param = params_updated[name]
    
        #     if not torch.all(torch.eq(initial_param, updated_param)):
        #         print(f"Parameter {name} has been updated.")
            
        # torch.autograd.set_detect_anomaly(False)
        
        # Update metric
        metric_logger.update(loss=loss_reduced.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc=acc_reduced.item())
        metric_logger.update(F1_score=torch.stack(F1score).mean().item())

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
