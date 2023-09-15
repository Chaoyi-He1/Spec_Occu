import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import List


class Steps_BCELoss(nn.Module):
    def __init__(self):
        super(Steps_BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
    
    def forward(self, pred: Tensor, targets: Tensor):
        _, l, _ = targets.shape
        device = targets.device
        targets = 2 * targets - 1
        # loss_steps = []
        # for i in range(l):
        #     loss_steps.append(self.bce(pred[:, i, :], targets[:, i, :]))
        # loss_steps = torch.as_tensor(loss_steps).to(device)
        loss = self.bce(pred, targets)
        return loss


class Temporal_Freq_Loss(nn.Module):
    def __init__(self, time_step_weights: list = None) -> None:
        super(Temporal_Freq_Loss, self).__init__()
        assert time_step_weights is not None, \
            "time_step_weights must be provided as a list"
        self.weights = torch.tensor(time_step_weights)
        self.loss = Steps_BCELoss()
    
    def forward(self, pred: Tensor, targets: Tensor):
        """
        Calculate the temporal to frequency labels loss.
        :param pred: Tensor [B, time_step, num_freq]
        :param targets: Tensor [B, time_step, num_freq], 
            targets is the ground truth of the frequency occupancy labels

        :return: Temporal_Freq_Loss: Tensor, the temporal to frequency loss
        """
        b, l, n = targets.shape
        device = targets.device

        # assert pred.shape == (b, l, n), \
        #     "pred shape should be [B, time_step, num_freq]"
        
        # assert len(self.weights) == l, \
        #     "time_step_weights length should be the same as time_step"

        # Calculate the loss
        loss = self.loss(pred, targets)

        acc_steps = Steps_Accuracy(pred, targets)
        return loss, acc_steps
    

@torch.no_grad()
def Steps_Accuracy(pred: Tensor, targets: Tensor):
    """
    Calculate the accuracy of the temporal to frequency labels.
    :param pred: Tensor [B, time_step, num_freq]
    :param targets: Tensor [B, time_step, num_freq], 
        targets is the ground truth of the frequency occupancy labels

    :return: Steps_Accuracy: Tensor, the temporal to frequency accuracy
    """
    b, l, n = targets.shape

    assert pred.shape == (b, l, n), \
        "pred shape should be [B, time_step, num_freq]"

    acc_steps = []
    for i in range(l):
        acc_steps.append(torch.eq(F.sigmoid(pred[:, i, :]).round(), 
                                  targets[:, i, :]).sum().float() / (b * n))
    return torch.stack(acc_steps)


@torch.no_grad()
def F1_score(pred: Tensor, targets: Tensor) -> List:
    """
    Calculate the F1 score of the temporal to frequency labels.
    :param pred: Tensor [B, time_step, num_freq]
    :param targets: Tensor [B, time_step, num_freq], 
        targets is the ground truth of the frequency occupancy labels

    :return: F1_score: Tensor, the temporal to frequency F1 score
    """
    b, l, n = targets.shape

    assert pred.shape == (b, l, n), \
        "pred shape should be [B, time_step, num_freq]"

    F1_scores = []
    for i in range(l):
        tp = (F.sigmoid(pred[:, i, :]).round() * targets[:, i, :]).sum().float()
        fp = (F.sigmoid(pred[:, i, :]).round() * (1 - targets[:, i, :])).sum().float()
        fn = ((1 - F.sigmoid(pred[:, i, :]).round()) * targets[:, i, :]).sum().float()
        F1_scores.append(tp / (tp + 0.5 * (fp + fn)))
    return F1_scores
