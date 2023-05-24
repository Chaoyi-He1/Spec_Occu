import torch
import torch.nn as nn
from torch import Tensor


def Freq_Similarity(Targets: torch.Tensor):
    """
    Calculate the frequency similarity between two tensors.
    :param Targets: Tensor [B, L, N],
    B is the batch size, L is the sequence length, N is the number of frequency blocks
    :return: Tensor
    """
    b, l, n = Targets.shape
    
    # Reshape the Targets tensor for efficient comparison
    reshaped_targets = Targets.view(b, -1)
    
    # Calculate the frequency similarity using matrix multiplication
    similarity_mtx = reshaped_targets @ reshaped_targets.t()
    
    # Set the diagonal elements to zero
    similarity_mtx.diagonal().zero_()

    return similarity_mtx / (l * n)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: Representation Learning with Contrastive Predictive Coding
    (CPC) https://arxiv.org/pdf/1807.03748.pdf
    """
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, similarity_mtx: Tensor):
        """
        Calculate the contrastive loss.
        :param similarity_mtx: Tensor [B, B],
        B is the batch size
        :return: Tensor
        """
        b = similarity_mtx.shape[0]
        
        # Calculate the similarity between each pair of samples
        similarity_mtx = torch.exp(similarity_mtx / self.temperature)
        
        # Calculate the loss
        loss = (-torch.log(similarity_mtx / similarity_mtx.sum(dim=1, keepdim=True))).sum() / b
        
        return loss