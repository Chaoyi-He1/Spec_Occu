import torch
import torch.nn as nn
from util.misc import *
from util.Contrastive import *
from typing import Iterable


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = .01,
                    scaler=None, steps: int = 12):
    model.train()
    criterion.train()
    Step_Predict = Step_Prediction(num_steps=steps)
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    


def evaluate(model: torch.nn.Module, criterion: torch.nn.Module,
             data_loader: Iterable, device: torch.device, epoch: int,
             steps: int = 12):
    pass