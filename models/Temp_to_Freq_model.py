"""
Temporal to Frequency label model
Based on the Transforemr Encoder-Decoder model with the following modifications:
1. The encoder is formed by a series of 1D conv-Norm-Transformer blocks
2. The decoder is formed by a series of Tramsformer decoder blocks with num of query = time steps
"""

from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .transformer import Transformer_Encoder, Transformer_Decoder, DropPath


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "sigmoid":
        return F.sigmoid
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransEncoder_Conv1d_Act_block(nn.Module):
    def __init__(self, num_layers=4, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1,
                 drop_path=0.4, activation="relu", normalize_before=True,
                 kernel=7, sequence_length=64):
        super(TransEncoder_Conv1d_Act_block, self).__init__()

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = Transformer_Encoder(num_layers=num_layers, norm=encoder_norm, d_model=d_model,
                                           nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                           drop_path=drop_path, activation=activation, 
                                           normalize_before=normalize_before)
        self.conv1d = nn.Conv1d(in_channels=sequence_length, out_channels=sequence_length, 
                                kernel_size=kernel, stride=2, padding=kernel // 2 - 1)
        self.activation = _get_activation_fn(activation)
    
    def forward(self, src: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None) -> Tensor:
        # src: [B, L, Embedding] 
        # L is the sequence length; Embedding is the embedding dimension; B is the batch size

        x = self.encoder(src, src_key_padding_mask, pos_embed)
        x = self.conv1d(x)
        return self.activation(x)


class Encoder(nn.Module):
    def __init__(self, num_blocks=3, num_layers=4, d_model=512, nhead=8, 
                 dim_feedforward=2048, dropout=0.1,
                 drop_path=0.4, activation="relu", normalize_before=True,
                 kernel=7, sequence_length=32) -> None:
        super(Encoder, self).__init__()

        block_params = {
            "num_layers": num_layers,
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "drop_path": drop_path,
            "activation": activation,
            "normalize_before": normalize_before,
            "kernel": kernel,
            "sequence_length": sequence_length
        }

        block_list = []
        for _ in range(num_blocks):
            block_list.append(TransEncoder_Conv1d_Act_block(**block_params))
            block_params["d_model"] /= 2
        
        self.blocks = nn.ModuleList(block_list)
        self.embed_dim = block_params["d_model"]
