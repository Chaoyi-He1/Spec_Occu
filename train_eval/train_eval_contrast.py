import torch
import torch.nn as nn
from util import *


def train_one_epoch(dataloader, model, criterion, optimizer, device, epoch, print_freq=10):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for samples, targets in metric_logger.log_every(dataloader, print_freq, header):
        samples = samples.to(device)
        targets = targets.to(device)
        outputs = model(samples)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    return metric_logger