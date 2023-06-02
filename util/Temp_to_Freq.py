import torch
import torch.nn as nn
from torch import Tensor


class Steps_BCELoss(nn.Module):
    def __init__(self):
        super(Steps_BCELoss, self).__init__()
        self.bce = nn.BCELoss(reduction="sum")
    
    def forward(self, pred: Tensor, targets: Tensor):
        _, l, _ = targets.shape
        device = targets.device
        loss_steps = []
        for i in range(l):
            loss_steps.append(self.bce(pred[:, i, :], targets[:, i, :]))
        loss = torch.as_tensor(loss_steps).to(device)
        return loss


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
        acc_steps.append(torch.eq(pred[:, i, :].round(), 
                                  targets[:, i, :]).sum().float() / (b * n))
    return acc_steps


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

        assert pred.shape == (b, l, n), \
            "pred shape should be [B, time_step, num_freq]"
        
        assert len(self.weights) == l, \
            "time_step_weights length should be the same as time_step"

        # Calculate the loss
        loss_steps = self.loss(pred, targets, reduction="none").to(device)
        loss = torch.sum(loss_steps * self.weights)
        acc_steps = Steps_Accuracy(pred, targets)
        return loss, acc_steps
    