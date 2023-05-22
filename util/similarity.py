import torch
import torch.nn as nn
import math
from torch import Tensor


def Freq_Similarity(Targets: torch.Tensor):
    """
    Calculate the frequency similarity between two tensors.
    :param Targets: Tensor [B, L, N],
    B is the batch size, L is the sequence length, N is the number of frequency blocks
    :return: Tensor
    """
    b, l, n = Targets.shape
    device = Targets.device

    similarity_mtx = torch.zeros((b, b), device=device)

    for i in range(b):
        similarity_mtx[i, i+1:] = torch.sum(Targets[i] == Targets[i+1:], dim=(1, 2))

    similarity_mtx += similarity_mtx.t()  # Add the transpose to get the full similarity matrix

    return similarity_mtx / (l * n)
