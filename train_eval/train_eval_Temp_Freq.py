import torch
import torch.nn as nn
import math
import sys
from util.misc import *
from util.Temp_to_Freq import *
from typing import Iterable


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    scaler=None):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc_steps', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for i, (temporal, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        temporal = temporal.to(device)
        targets = targets.to(device)

        # Compute output
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            _, l, _ = targets.shape
            output = model(temporal)
            loss, acc_steps = criterion(output, targets)
            acc = acc_steps.mean()

        # reduce loss over all GPUs for logging purposes
        assert acc_steps.len() == l, \
            "acc_steps shape should be [time_step, ]"
        loss_reduced = reduce_loss(loss)
        acc_reduced = reduce_loss(acc)
        acc_steps_reduced = [reduce_loss(acc_step).item() for acc_step in acc_steps]

        if not math.isfinite(loss_reduced):
            print("Loss is {}, stopping training".format(loss_reduced))
            print(loss_reduced)
            sys.exit(1)
        
        # Compute gradient
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss_reduced).backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_reduced.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        
        # Update meters
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss_reduced.item())
        metric_logger.update(acc=acc_reduced.item())
        metric_logger.update(acc_steps=acc_steps_reduced)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # return summary for logging
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module,
             data_loader: Iterable, device: torch.device):
    model.eval()
    criterion.eval()

    # MetricLogger
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc_steps', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Test:'

    for i, (temporal, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        temporal = temporal.to(device)
        targets = targets.to(device)

        # Compute output
        _, l, _ = targets.shape
        output = model(temporal)
        loss, acc_steps = criterion(output, targets)
        acc = acc_steps.mean()

        # reduce loss over all GPUs for logging purposes
        assert acc_steps.len() == l, \
            "acc_steps shape should be [time_step, ]"
        loss_reduced = reduce_loss(loss)
        acc_reduced = reduce_loss(acc)
        acc_steps_reduced = [reduce_loss(acc_step).item() for acc_step in acc_steps]

        # Update meters
        metric_logger.update(loss=loss_reduced.item())
        metric_logger.update(acc=acc_reduced.item())
        metric_logger.update(acc_steps=acc_steps_reduced)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # return summary for logging
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
