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
    def __init__(self, time_step_weights: list = None):
        # time_step_weights: list, the weights for each time step loss
        super(ContrastiveLoss, self).__init__()
        assert time_step_weights is not None, "time_step_weights should be a list"
        self.weights = torch.tensor(time_step_weights)
        self.softmax = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, pred: Tensor, targets: Tensor, model: nn.Module):
        """
        Calculate the infoNCE loss.
        :param pred: Tensor [time_step, B, embed_dim]
        :param targets: Tensor [B, time_step, in_dim], in_dim is the temporal dimension
        :param model: nn.Module, the encoder and regressor model

        :return: NCELoss: Tensor, the infoNCE loss
                 cls_prd: Tensor, the classification accuracy
        """
        b, l, t_d = targets.shape
        device = targets.device

        assert pred.shape == (l, b, model.embed_dim), \
            "pred shape should be [time_step, B, embed_dim]"
        assert t_d == model.AutoEncoder_cfg["in_dim"], \
            "Input temporal dimension should be the same as the in_dim in AutoEncoder"
        assert len(self.weights) == l, "time_step_weights length should be the same as time_step"
        
        # Calculate the true futures (targets) encoded features (embed vectors)
        # encoded_targets: [B, time_step, embed_dim]
        # TODO: use torch.no_grad() or not?
        with torch.no_grad():
            encoded_targets = torch.as_tensor([self.encoder(targets[i, :, :].unsqueeze(1) 
                                                            for i in range(b))]).to(device)
            encoded_targets = encoded_targets.permute(1, 0, 2).contiguous()

        # Calculate the loss
        # pred: [time_step, B, embed_dim]; encoded_targets: [time_step, B, embed_dim]
        # mutural_info: [time_step, B, B]
        mutural_info = torch.matmul(encoded_targets, torch.transpose(pred, 1, 2))
        NCELoss = -torch.sum(torch.diagonal(self.lsoftmax(mutural_info), dim1=-2, dim2=-1) * self.weights)
        NCELoss /= l * b
        cls_prd = torch.sum(torch.eq(torch.argmax(self.softmax(mutural_info), dim=1),
                           torch.arange(b).unsqueeze(0).to(device)))
        cls_prd /= l * b

        return NCELoss, cls_prd