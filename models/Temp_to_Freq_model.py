"""
Temporal to Frequency label model
Based on the Transforemr Encoder-Decoder model with the following modifications:
1. The encoder is formed by a series of 1D conv-Norm-Transformer blocks
2. The decoder is formed by a series of Tramsformer decoder blocks with num of query = time steps
"""

from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math

from .transformer import Transformer_Encoder, Transformer_Decoder, DropPath
from .positional_embedding import build_position_encoding


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
    if activation == "tanh":
        return F.tanh
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def calculate_conv1d_padding(stride, kernel_size, d_in, d_out, dilation=1):
    """
    Calculate the padding value for a 1D convolutional layer.

    Args:
        stride (int): Stride value for the convolutional layer.
        kernel_size (int): Kernel size for the convolutional layer.
        d_in (int): Input dimension of the feature map.
        d_out (int): Output dimension of the feature map.
        dilation (int, optional): Dilation value for the convolutional layer.
                                  Default is 1.

    Returns:
        int: Padding value for the convolutional layer.

    """
    padding = math.ceil(((d_out - 1) * stride + kernel_size - 
                         d_in + (kernel_size - 1) * (dilation - 1)) / 2)
    assert padding >= 0, "Padding value must be greater than or equal to 0."

    return padding


def calculate_conv2d_padding(stride, kernel_size, d_in, d_out, dilation=1):
    """
    Calculate the padding value for a 2D convolutional layer.
    
    Arguments:
    - stride (int or tuple): The stride value(s) for the convolution.
    - kernel_size (int or tuple): The size of the convolutional kernel.
    - d_in (tuple): The input dimensions (height, width) of the feature map.
    - d_out (tuple): The output dimensions (height, width) of the feature map.
    - dilation (int or tuple): The dilation value(s) for the convolution. Default is 1.
    
    Returns:
    - padding (tuple): The padding value(s) (padding_h, padding_w) for the convolution.
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    h_in, w_in = d_in
    h_out, w_out = d_out
    h_k, w_k = kernel_size
    h_s, w_s = stride
    h_d, w_d = dilation

    padding_h = math.ceil(((h_out - 1) * h_s + h_k - h_in + (h_k - 1) * (h_d - 1)) / 2)
    padding_w = math.ceil(((w_out - 1) * w_s + w_k - w_in + (w_k - 1) * (w_d - 1)) / 2)
    assert padding_h >= 0 and padding_w >= 0, "Padding value(s) cannot be negative."

    padding = (padding_h, padding_w)
    return padding


class Conv1d_BN_Relu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv1d_BN_Relu, self).__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return super(Conv1d_BN_Relu, self).forward(x)


class Conv2d_BN_Relu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv2d_BN_Relu, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return super(Conv2d_BN_Relu, self).forward(x)


class ResBlock_2d(nn.Module):
    def __init__(self, in_channel: int, kernel_size: int = 3, stride: int = 1, 
                 in_dim: Tuple[int, int] = (256, 1024), dilation: int = 1, 
                 drop_path_ratio: float = 0.4) -> None:
        super(ResBlock_2d, self).__init__()
        pad = calculate_conv2d_padding(stride, kernel_size, in_dim, in_dim, dilation)
        self.conv1 = Conv2d_BN_Relu(in_channel, in_channel // 2, kernel_size=1)
        self.conv2 = Conv2d_BN_Relu(in_channel // 2, in_channel, kernel_size=kernel_size, 
                                    padding=pad, stride=stride, dilation=dilation)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop_path(self.conv2(self.conv1(x)))


class ResBlock_1d(nn.Module):
    def __init__(self, in_channel: int, kernel_size: int = 3, stride: int = 1, 
                 in_dim: int = 1024, dilation: int = 1, 
                 drop_path_ratio: float = 0.4) -> None:
        super(ResBlock_1d, self).__init__()
        pad = calculate_conv1d_padding(stride, kernel_size, in_dim, in_dim, dilation)
        self.conv1 = Conv1d_BN_Relu(in_channel, in_channel // 2, kernel_size=1)
        self.conv2 = Conv1d_BN_Relu(in_channel // 2, in_channel, kernel_size=kernel_size, 
                                    padding=pad, stride=stride, dilation=dilation)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop_path(self.conv2(self.conv1(x)))


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
        padding = calculate_conv1d_padding(stride=3, kernel_size=kernel, 
                                           d_in=d_model, d_out=d_model)
        self.conv1d = nn.Conv1d(in_channels=sequence_length, out_channels=sequence_length, 
                                kernel_size=kernel, stride=3, padding=padding)
        self.activation = _get_activation_fn(activation)
    
    def forward(self, src: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None) -> Tensor:
        # src: [B, L, Embedding] 
        # L is the sequence length; Embedding is the embedding dimension; B is the batch size

        x = self.encoder(src=src, src_key_padding_mask=src_key_padding_mask, 
                         pos=pos_embed)
        x = self.conv1d(x)
        return self.activation(x)


class Encoder(nn.Module):
    def __init__(self, num_blocks=3, num_layers=4, d_model=512, nhead=8, 
                 dim_feedforward=512, dropout=0.1,
                 drop_path=0.4, activation="gelu", normalize_before=True,
                 kernel=5, sequence_length=32) -> None:
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
        
        self.blocks = nn.ModuleList(block_list)
        self.embed_dim = block_params["d_model"]

        self.norm = _get_activation_fn("tanh")

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            src: [B, L, Embedding] 
            L is the sequence length; Embedding is the embedding dimension; B is the batch size
        """
        x = src

        for block in self.blocks:
            x = block(x, src_key_padding_mask, pos_embed)
        return self.norm(x)


class TransDecoder_Conv1d_Act_block(nn.Module):
    def __init__(self, num_layers=4, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1,
                 drop_path=0.4, activation="relu", normalize_before=True,
                 kernel=7, sequence_length=64, kdim=None, vdim=None) -> None:
        super(TransDecoder_Conv1d_Act_block, self).__init__()

        deoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = Transformer_Decoder(num_layers=num_layers, norm=deoder_norm, d_model=d_model,
                                           nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                           drop_path=drop_path, activation=activation,
                                           normalize_before=normalize_before, kdim=kdim, vdim=vdim)
        padding = calculate_conv1d_padding(stride=1, kernel_size=kernel,
                                           d_in=d_model, d_out=d_model)
        self.conv1d = nn.Conv1d(in_channels=sequence_length, out_channels=sequence_length,
                                kernel_size=kernel, stride=1, padding=padding)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None,
                query_pos_embed: Optional[Tensor] = None) -> Tensor:
        # tgt: [B, L, Embedding] 
        # L is the sequence length; Embedding is the embedding dimension; B is the batch size

        x = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                         pos_embed, query_pos_embed)
        x = self.conv1d(x)
        return self.activation(x)


class Decoder(nn.Module):
    def __init__(self, num_blocks=3, num_layers=4, d_model=512, nhead=8, 
                 dim_feedforward=2048, dropout=0.1,
                 drop_path=0.4, activation="gelu", normalize_before=True,
                 kernel=5, sequence_length=32, kdim=None, vdim=None) -> None:
        super(Decoder, self).__init__()

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
            "sequence_length": sequence_length,
            "kdim": kdim,
            "vdim": vdim,
        }

        block_list = [TransDecoder_Conv1d_Act_block(**block_params) 
                      for _ in range(num_blocks)]

        self.blocks = nn.ModuleList(block_list)
        self.embed_dim = block_params["d_model"]

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None,
                query_pos_embed: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            tgt: [B, L, Embedding] 
            L is the sequence length; Embedding is the embedding dimension; B is the batch size
        """
        x = tgt

        for block in self.blocks:
            x = block(x, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                      pos_embed, query_pos_embed)
        return x


class Transformer_Temp_2_Freq(nn.Module):
    def __init__(self, cfg: dict = None, pos_type: str = "sine") -> None:
        super(Transformer_Temp_2_Freq, self).__init__()
        assert cfg is not None, "cfg is None"
        self.encoder_cfg = {
            "num_blocks": cfg["num_T2F_encoder_blocks"],
            "num_layers": cfg["num_T2F_encoder_layers"],
            "d_model": cfg["T2F_encoder_embed_dim"],
            "nhead": cfg["T2F_encoder_nhead"],
            "dropout": cfg["dropout"],
            "drop_path": cfg["drop_path"],
            "sequence_length": cfg["T2F_encoder_sequence_length"],
        }

        self.decoder_cfg = {
            "num_blocks": cfg["num_T2F_decoder_blocks"],
            "num_layers": cfg["num_T2F_decoder_layers"],
            "d_model": cfg["T2F_decoder_embed_dim"],
            "nhead": cfg["T2F_decoder_nhead"],
            "dropout": cfg["dropout"],
            "drop_path": cfg["drop_path"],
            "sequence_length": cfg["T2F_num_queries"],
            "kdim": cfg["T2F_encoder_embed_dim"],
            "vdim": cfg["T2F_encoder_embed_dim"],
        }

        self.reduce_dim_conv = nn.Conv2d(in_channels=cfg["T2F_encoder_sequence_length"],
                                         out_channels=cfg["T2F_encoder_sequence_length"],
                                         kernel_size=(2, 1), stride=(1, 1), padding=(0, 0))

        self.encoder = Encoder(**self.encoder_cfg)
        self.decoder = Decoder(**self.decoder_cfg)

        self.query_embed = nn.Embedding(cfg["T2F_num_queries"], self.decoder_cfg["d_model"])
        
        self.encoder_pos = build_position_encoding(type=pos_type, 
                                                   embed_dim=self.encoder_cfg["d_model"])
        self.decoder_pos = build_position_encoding(type=pos_type,
                                                   embed_dim=self.encoder_cfg["d_model"])
        
        self.classify_head = nn.parameter.Parameter(torch.randn(cfg["T2F_num_queries"], 
                                                      cfg["T2F_decoder_embed_dim"], 
                                                      cfg["T2F_num_classes"]))
        
    def forward(self, src: Tensor,
                src_mask: Optional[Tensor] = None) -> Tensor:
        src = self.reduce_dim_conv(src).squeeze(-2)
        src_pos_embed = self.encoder_pos(src)
        memory = self.encoder(src, src_mask, src_pos_embed)

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(src.shape[0], 1, 1)

        tgt = torch.zeros_like(query_embed)
        decoder_pos_embed = self.decoder_pos(memory)
        # hs: [B, num_queries, decoder_embed_model]
        hs = self.decoder(tgt, memory, pos_embed=decoder_pos_embed, query_pos_embed=query_embed)

        pred_cls = torch.einsum("bnd,ndc->bnc", hs, self.classify_head)
        return pred_cls
    

class Conv1d_Temp_2_Freq(nn.Module):
    def __init__(self, cfg: dict = None) -> None:
        super(Conv1d_Temp_2_Freq, self).__init__()
        assert cfg is not None, "cfg is None"

        self.reduce_dim_conv = nn.Conv2d(in_channels=cfg["T2F_encoder_sequence_length"],
                                         out_channels=cfg["T2F_encoder_sequence_length"],
                                         kernel_size=(2, 1), stride=(1, 1), padding=(0, 0))
        self.channel = cfg["T2F_encoder_sequence_length"]
        self.temp_dim = cfg["T2F_encoder_embed_dim"]

        self.ResNet = nn.ModuleList()
        res_params = list(zip([4, 4, 6, 6, 4], [7, 7, 9, 9, 11],   # num_blocks, kernel_size
                              [3, 3, 3, 3, 3], [1, 5, 5, 3, 3]))   # stride, dilation
        for i, (num_blocks, kernel_size, stride, dilation) in enumerate(res_params):
            self.ResNet.extend([ResBlock_1d(self.channel, kernel_size, stride, self.temp_dim, dilation)
                                for _ in range(num_blocks)])
            if i != len(res_params) - 1:
                pad = calculate_conv1d_padding(stride, kernel_size, self.temp_dim, self.temp_dim // 2, dilation)
                self.ResNet.append(Conv1d_BN_Relu(self.channel, self.channel * 2,
                                                  kernel_size, stride, pad, dilation))
                self.channel *= 2
                self.temp_dim //= 2
        self.ResNet.append(Conv1d_BN_Relu(in_channels=self.channel, 
                                          out_channels=cfg["T2F_encoder_sequence_length"],
                                          kernel_size=1, stride=1, padding=0, dilation=1))
        
        self.classify_head = nn.Linear(self.temp_dim, cfg["T2F_num_classes"])
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: [B, L, 2, T]
                B: batch size
                L: T2F_encoder_sequence_length, length of time frames
                T: T2F_encoder_embed_dim, length of time dimension
                2: real and imag channels
        """
        x = self.reduce_dim_conv(inputs).squeeze(-2)
        for layer in self.ResNet:
            x = layer(x)
        x = self.classify_head(x)
        return x
        

def build_T2F(cfg: dict = None, pos_type: str = "sine", 
              model_type: str = "Conv") -> Transformer_Temp_2_Freq:
    model = Transformer_Temp_2_Freq(cfg, pos_type) if model_type == "Transformer" \
            else Conv1d_Temp_2_Freq(cfg)
    return model