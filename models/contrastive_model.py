"""
Contrastive Model.

Autoencoder model and Autoregressive model for contrastive learning.
Based on: Representation Learning with Contrastive Predictive Coding
(CPC) https://arxiv.org/pdf/1807.03748.pdf

Autoencoder model is used to extract features from the input data.
Autoencoder model is composed of ResNet style 1d convolutional neural network.

Autoregressive model is used to summarizes all z ≤ t
in the latent space and produces a context latent representation c_t = g(z ≤ t).
Autoregressive model is composed of Transformer encoder and conv1d layer for downsampling.
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
    padding = math.ceil((stride * (d_out - 1) - 
                         d_in + (dilation * 
                                 (kernel_size - 1)) + 1) / 2)
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


class ResBlock_2d_with_Attention(nn.Module):
    def __init__(self, in_channel: int, kernel_size: int = 3, stride: int = 1, 
                 in_dim: Tuple[int, int] = (256, 1024), dilation: int = 1, 
                 drop_path_ratio: float = 0.4) -> None:
        super(ResBlock_2d_with_Attention, self).__init__()
        pad = calculate_conv2d_padding(stride, kernel_size, in_dim, in_dim, dilation)
        self.conv1 = Conv2d_BN_Relu(in_channel, in_channel // 2, kernel_size=1)
        self.conv2 = Conv2d_BN_Relu(in_channel // 2, in_channel, kernel_size=kernel_size, 
                                    padding=pad, stride=stride, dilation=dilation)
        self.atten = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        conv_out = self.conv2(self.conv1(x))
        atten_out = F.sigmoid(self.atten(x))
        return x * atten_out + self.drop_path(conv_out)


class ResBlock_1d_with_Attention(nn.Module):
    def __init__(self, in_channel: int, kernel_size: int = 3, stride: int = 1, 
                 in_dim: int = 1024, dilation: int = 1, 
                 drop_path_ratio: float = 0.4) -> None:
        super(ResBlock_1d_with_Attention, self).__init__()
        pad = calculate_conv1d_padding(stride, kernel_size, in_dim, in_dim, dilation)
        self.conv1 = Conv1d_BN_Relu(in_channel, in_channel // 2, kernel_size=1)
        self.conv2 = Conv1d_BN_Relu(in_channel // 2, in_channel, kernel_size=kernel_size, 
                                    padding=pad, stride=stride, dilation=dilation)
        self.atten = nn.Conv1d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        conv_out = self.conv2(self.conv1(x))
        atten_out = F.sigmoid(self.atten(x))
        return x * atten_out + self.drop_path(conv_out)


class Conv2d_AutoEncoder(nn.Module):
    def __init__(self, in_dim: Tuple[int, int] = (256, 1024), 
                 in_channel: int = 2, drop_path: float = 0.4, with_atten: bool = True) -> None:
        super(Conv2d_AutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.temp_dim = in_dim
        self.channel = 16
        pad = calculate_conv2d_padding(1, 11, self.temp_dim, self.temp_dim)
        self.conv1 = Conv2d_BN_Relu(self.in_channel, self.channel, kernel_size=11, padding=pad)
        pad = calculate_conv2d_padding(2, 13, self.temp_dim, tuple(element // 2 for element in self.temp_dim))
        self.conv2 = Conv2d_BN_Relu(self.channel, self.channel * 2, kernel_size=13, stride=2, padding=pad)
        self.channel *= 2
        self.temp_dim = tuple(element // 2 for element in self.temp_dim)

        self.ResNet = nn.ModuleList()
        res_params = list(zip([4, 4, 6, 6, 4], [7, 7, 9, 9, 11],   # num_blocks, kernel_size
                              [3, 3, 3, 3, 3], [5, 5, 5, 3, 3]))   # stride, dilation
        # final channels = 512; final temp_dim = in_dim // (2^5) = in_dim // 32
        for i, (num_blocks, kernel_size, stride, dilation) in enumerate(res_params):
            self.ResNet.extend([ResBlock_2d(self.channel, kernel_size, stride, self.temp_dim, dilation,
                                            drop_path) if not with_atten else \
                                ResBlock_2d_with_Attention(self.channel, kernel_size, stride, self.temp_dim,
                                                           dilation, drop_path)
                                for _ in range(num_blocks)])
            if i != len(res_params) - 1:
                pad = calculate_conv2d_padding(stride, kernel_size, self.temp_dim, 
                                               tuple(element // 2 for element in self.temp_dim), dilation)
                self.ResNet.append(Conv2d_BN_Relu(self.channel, self.channel * 2,
                                                  kernel_size, stride, pad, dilation))
                self.channel *= 2
                self.temp_dim = tuple(element // 2 for element in self.temp_dim)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: [L, 2, in_dim], 
        # L is the sequence_length in the following Transformer based AutoRegressive model
        # output: [L, Embedding], Embedding = 512

        # assert inputs.shape[1] == 2 and len(inputs.shape) == 3, "Input shape should be [B, 2, Embedding]"

        x = self.conv1(inputs)
        x = self.conv2(x)
        for block in self.ResNet:
            x = block(x)
        x = self.avgpool(x)
        return x.squeeze(-1).squeeze(-1)


class Conv1d_AutoEncoder(nn.Module):
    def __init__(self, in_dim: int = 1024, in_channel: int = 2, 
                 drop_path: float = 0.4, with_atten: bool = True) -> None:
        super(Conv1d_AutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.temp_dim = in_dim
        self.channel = 16
        pad = calculate_conv1d_padding(1, 11, self.temp_dim, self.temp_dim)
        self.conv1 = Conv1d_BN_Relu(self.in_channel, self.channel, kernel_size=11, padding=pad)
        pad = calculate_conv1d_padding(2, 13, self.temp_dim, self.temp_dim // 2)
        self.conv2 = Conv1d_BN_Relu(self.channel, self.channel * 2, kernel_size=13, stride=2, padding=pad)
        self.channel *= 2
        self.temp_dim //= 2

        self.ResNet = nn.ModuleList()
        res_params = list(zip([2, 2, 4, 4, 4], [513, 385, 257, 129, 65],   # num_blocks, kernel_size  513, 385, 257, 129, 65    7, 7, 9, 9, 11
                              [3, 3, 3, 3, 3], [5, 5, 5, 3, 3]))   # stride, dilation
        # final channels = 512; final temp_dim = in_dim // (2^5) = in_dim // 32
        for i, (num_blocks, kernel_size, stride, dilation) in enumerate(res_params):
            self.ResNet.extend([ResBlock_1d(self.channel, kernel_size, stride, self.temp_dim, dilation,
                                            drop_path) if not with_atten else \
                                ResBlock_1d_with_Attention(self.channel, kernel_size, stride, self.temp_dim,
                                                           dilation, drop_path)
                                for _ in range(num_blocks)])
            if i != len(res_params) - 1:
                pad = calculate_conv1d_padding(stride, kernel_size, self.temp_dim, self.temp_dim // 2, dilation)
                self.ResNet.append(Conv1d_BN_Relu(self.channel, self.channel * 2,
                                                  kernel_size, stride, pad, dilation))
                self.channel *= 2
                self.temp_dim //= 2
        
        self.ResNet.append(nn.Conv1d(in_channels=self.channel, out_channels=1, kernel_size=1))
        
        self.reduce_temp_dim = nn.Sequential(
            nn.Linear(self.temp_dim, 256),
            nn.Linear(256, 128),
            nn.Linear(128, self.channel),
        )
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: [L, 4, in_dim], 
        # L is the sequence_length in the following Transformer based AutoRegressive model
        # output: [L, Embedding], Embedding = 512

        # assert inputs.shape[1] == 2 and len(inputs.shape) == 3, "Input shape should be [B, 2, Embedding]"

        x = self.conv1(inputs)
        x = self.conv2(x)
        for block in self.ResNet:
            x = block(x)
        x = x.squeeze(-2)
        x = self.reduce_temp_dim(x)
        return x

class TransEncoder_Conv1d_Act_block(nn.Module):
    def __init__(self, num_layers=4, d_model=512, nhead=8, dim_feedforward=512, dropout=0.1,
                 drop_path=0.4, activation="relu", normalize_before=True,
                 kernel=7, sequence_length=64):
        super(TransEncoder_Conv1d_Act_block, self).__init__()

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = Transformer_Encoder(num_layers=num_layers, norm=encoder_norm, d_model=d_model,
                                           nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                           drop_path=drop_path, activation=activation,
                                           normalize_before=normalize_before)
        self.conv1d = nn.Conv1d(in_channels=sequence_length, out_channels=sequence_length,
                                kernel_size=kernel, stride=2, padding=kernel // 2)
        self.activation = _get_activation_fn(activation)

    def forward(self, src: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None) -> Tensor:
        # src: [B, L, Embedding] 
        # L is the sequence length; Embedding is the embedding dimension; B is the batch size

        x = self.encoder(src=src, src_key_padding_mask=src_key_padding_mask, pos=pos_embed)
        x = self.conv1d(x)
        return self.activation(x)


class Autoregressive(nn.Module):
    def __init__(self, num_blocks=3, feature_dim=256, pos_type="sine",
                 num_layers=4, d_model=512, nhead=8, dim_feedforward=512, dropout=0.1,
                 drop_path=0.4, activation="gelu", normalize_before=True,
                 kernel=7, sequence_length=32) -> None:
        super(Autoregressive, self).__init__()

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
        self.pos_embeds = []
        for _ in range(num_blocks):
            block_list.append(TransEncoder_Conv1d_Act_block(**block_params))
            self.pos_embeds.append(build_position_encoding(type=pos_type,
                                                           embed_dim=block_params["d_model"]))
            block_params["d_model"] //= 2

        self.blocks = nn.ModuleList(block_list)
        self.embed_dim = block_params["d_model"]
        self.sequence_length = block_params["sequence_length"]

        self.feature_layer = nn.Linear(self.embed_dim * self.sequence_length, feature_dim)
        self.norm = _get_activation_fn("tanh")

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_normal_(p)

    def forward(self, src: Tensor,
                src_key_padding_mask: Tensor = None) -> Tensor:
        """
        Parameters:
            src: [B, L, Embedding]
            src_key_padding_mask: [B, L], to mask out the padding part
            pos_embed: [B, L, Embedding]
        """
        for i, block in enumerate(self.blocks):
            pos_embed = self.pos_embeds[i](src)
            src = block(src, src_key_padding_mask, pos_embed)
        src = src.flatten(1)
        src = self.feature_layer(src)
        # src = self.norm(src)
        return src


class Autoregressive_Attention(nn.Module):
    def __init__(self, feature_dim=256, pos_type="sine", num_layers=4, 
                 d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1,
                 drop_path=0.4, activation="gelu", normalize_before=True,
                 sequence_length=32) -> None:
        super(Autoregressive_Attention, self).__init__()

        encoder_params = {
            "num_layers": num_layers,
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "drop_path": drop_path,
            "activation": activation,
            "normalize_before": normalize_before,
        }
        
        self.atten_encoder = Transformer_Encoder(**encoder_params)
        self.pos_embed = build_position_encoding(type=pos_type, embed_dim=d_model)
        self.embed_dim = d_model
        self.sequence_length = sequence_length

        self.feature_layer = nn.Linear(self.embed_dim * self.sequence_length, feature_dim)
    
    def forward(self, src: Tensor,
                src_key_padding_mask: Tensor = None) -> Tensor:
        """
        Parameters:
            src: [B, L, Embedding]
            src_key_padding_mask: [B, L], to mask out the padding part
            pos_embed: [B, L, Embedding]
        """
        pos_embed = self.pos_embed(src)
        src = self.atten_encoder(src=src, src_key_padding_mask=src_key_padding_mask, 
                                 pos=pos_embed)
        src = src.flatten(1)
        src = self.feature_layer(src)
        return src


class Encoder_Regressor(nn.Module):
    def __init__(self, cfg: dict = None, timestep: int = 12, 
                 pos_type: str = "sine", in_type: str = "1d") -> None:
        super(Encoder_Regressor, self).__init__()
        assert cfg is not None, "cfg should be a dict"
        padding_input_proj = calculate_conv1d_padding(stride=31, kernel_size=127,
                                                      d_in=cfg["Temporal_dim"], 
                                                      d_out=cfg["feature_dim"])
        self.input_residue_proj = nn.Sequential(
            nn.Conv2d(in_channels=cfg["contrast_sequence_length"],
                      out_channels=cfg["contrast_sequence_length"],
                      kernel_size=(2, 1), stride=(1, 1), padding=(0, 0)),
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Conv1d(in_channels=cfg["contrast_sequence_length"],
                      out_channels=cfg["contrast_sequence_length"],
                      kernel_size=127, stride=31, padding=padding_input_proj),
            nn.Conv1d(in_channels=cfg["contrast_sequence_length"],
                      out_channels=1,
                      kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=-2, end_dim=-1)) if in_type == "1d" else \
            nn.Sequential(
                nn.Conv2d(in_channels=cfg["in_channels"],
                          out_channels=1,
                          kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                nn.Flatten(start_dim=1, end_dim=2),
                nn.Conv1d(in_channels=cfg["contrast_sequence_length"],
                          out_channels=cfg["contrast_sequence_length"],
                          kernel_size=127, stride=31, padding=padding_input_proj),
                nn.Conv1d(in_channels=cfg["contrast_sequence_length"],
                          out_channels=1,
                          kernel_size=1, stride=1, padding=0),
                nn.Flatten(start_dim=-2, end_dim=-1)) 
        
        self.AutoEncoder_cfg = {
            "in_dim": (cfg["num_frames_per_clip"], cfg["Temporal_dim"]),
            "in_channel": cfg["in_channels"],
            "drop_path": cfg["drop_path"],
            "with_atten": cfg["Encoder_attention"],
        } if in_type == "2d" else {
            "in_dim": cfg["Temporal_dim"],
            "in_channel": cfg["in_channels"],
            "drop_path": cfg["drop_path"],
            "with_atten": cfg["Encoder_attention"],
        }
        self.encoder = Conv2d_AutoEncoder(**self.AutoEncoder_cfg) \
            if in_type == "2d" else \
                Conv1d_AutoEncoder(**self.AutoEncoder_cfg)
        self.embed_dim = self.encoder.channel
        self.frames_per_clip = cfg["num_frames_per_clip"]
        self.in_dim = cfg["Temporal_dim"]
        assert self.embed_dim == cfg["contrast_embed_dim"], \
            "embed_dim should be the same as the output of Conv1d_AutoEncoder"
        
        if not cfg["contrast_attention"]:
            self.AutoRegressive_cfg = {
                "num_blocks": cfg["num_contrast_blocks"],
                "feature_dim": cfg["feature_dim"],
                "num_layers": cfg["num_contrast_layers"],
                "d_model": self.embed_dim,
                "nhead": cfg["nhead"],
                "dropout": cfg["dropout"],
                "drop_path": cfg["drop_path"],
                "sequence_length": cfg["contrast_sequence_length"],
                "dim_feedforward": cfg["dim_feedforward"],
                "normalize_before": cfg["contrast_normalize_before"],
                "pos_type": pos_type,
            }
        else:
            self.AutoRegressive_cfg = {
                "feature_dim": cfg["feature_dim"],
                "num_layers": cfg["num_contrast_layers"],
                "d_model": self.embed_dim,
                "nhead": cfg["nhead"],
                "dropout": cfg["dropout"],
                "drop_path": cfg["drop_path"],
                "sequence_length": cfg["contrast_sequence_length"],
                "dim_feedforward": cfg["dim_feedforward"],
                "normalize_before": cfg["contrast_normalize_before"],
                "pos_type": pos_type,
            }

        self.regressor = Autoregressive(**self.AutoRegressive_cfg) if not cfg["contrast_attention"] else \
                         Autoregressive_Attention(**self.AutoRegressive_cfg)

        self.timestep = timestep
        self.feature_dim = cfg["feature_dim"]
        self.linear_trans = nn.parameter.Parameter(torch.randn(self.timestep, self.embed_dim, self.feature_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: [B, L, in_dim]; 
        # B is the batch size; L is the sequence length, in_dim is the input temporal dimension
        # output: [B, feature_dim]

        b = inputs.shape[0]
        device = inputs.device
        # assert t_d == self.AutoEncoder_cfg["in_dim"], \
        #     "Input temporal dimension should be the same as the in_dim in AutoEncoder"
        # assert c == self.encoder.in_channel, \
        #     "Input channel should be the same as the in_channel in AutoEncoder"
        encoder_outputs = torch.stack([self.encoder(inputs[i, ...])
                                       for i in range(b)]).to(device)
        # assert encoder_outputs.shape == (b, l, self.embed_dim), \
        #     "Encoder output shape should be [B, L, Embedding]"

        feature = self.regressor(encoder_outputs) + self.input_residue_proj(inputs)  # [B, feature_dim]
        # pred: [time_step, B, embed_dim, 1]
        pred = torch.matmul(self.linear_trans.unsqueeze(1), feature.unsqueeze(2))
        # assert pred.shape == (self.timestep, b, self.embed_dim, 1), \
        #     "pred shape should be [time_step, B, embed_dim, 1]"

        return pred.squeeze(-1)


class Feature_Extractor(nn.Module):
    def __init__(self, cfg: dict = None, pos_type: str = "sine", 
                 in_type: str = "1d") -> None:
        super(Feature_Extractor, self).__init__()
        assert cfg is not None, "cfg should be a dict"
        padding_input_proj = calculate_conv1d_padding(stride=31, kernel_size=127,
                                                      d_in=cfg["Temporal_dim"], 
                                                      d_out=cfg["feature_dim"])
        self.input_residue_proj = nn.Sequential(
            nn.Conv2d(in_channels=cfg["contrast_sequence_length"],
                      out_channels=cfg["contrast_sequence_length"],
                      kernel_size=(2, 1), stride=(1, 1), padding=(0, 0)),
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Conv1d(in_channels=cfg["contrast_sequence_length"],
                      out_channels=cfg["contrast_sequence_length"],
                      kernel_size=127, stride=31, padding=padding_input_proj),
            nn.Conv1d(in_channels=cfg["contrast_sequence_length"],
                      out_channels=1,
                      kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=-2, end_dim=-1))
        
        self.AutoEncoder_cfg = {
            "in_dim": (cfg["num_frames_per_clip"], cfg["Temporal_dim"]),
            "in_channel": cfg["in_channels"],
            "drop_path": cfg["drop_path"],
            "with_atten": cfg["Encoder_attention"],
        } if in_type == "2d" else {
            "in_dim": cfg["Temporal_dim"],
            "in_channel": cfg["in_channels"],
            "drop_path": cfg["drop_path"],
            "with_atten": cfg["Encoder_attention"],
        }
        self.encoder = Conv2d_AutoEncoder(**self.AutoEncoder_cfg) \
            if in_type == "2d" else \
                Conv1d_AutoEncoder(**self.AutoEncoder_cfg)
        self.embed_dim = self.encoder.channel
        self.frames_per_clip = cfg["num_frames_per_clip"]
        self.in_dim = cfg["Temporal_dim"]
        assert self.embed_dim == cfg["contrast_embed_dim"], \
            "embed_dim should be the same as the output of Conv1d_AutoEncoder"
        
        if not cfg["contrast_attention"]:
            self.AutoRegressive_cfg = {
                "num_blocks": cfg["num_contrast_blocks"],
                "feature_dim": cfg["feature_dim"],
                "num_layers": cfg["num_contrast_layers"],
                "d_model": self.embed_dim,
                "nhead": cfg["nhead"],
                "dropout": cfg["dropout"],
                "drop_path": cfg["drop_path"],
                "sequence_length": cfg["contrast_sequence_length"],
                "dim_feedforward": cfg["dim_feedforward"],
                "normalize_before": cfg["contrast_normalize_before"],
                "pos_type": pos_type,
            }
        else:
            self.AutoRegressive_cfg = {
                "feature_dim": cfg["feature_dim"],
                "num_layers": cfg["num_contrast_layers"],
                "d_model": self.embed_dim,
                "nhead": cfg["nhead"],
                "dropout": cfg["dropout"],
                "drop_path": cfg["drop_path"],
                "sequence_length": cfg["contrast_sequence_length"],
                "dim_feedforward": cfg["dim_feedforward"],
                "normalize_before": cfg["contrast_normalize_before"],
                "pos_type": pos_type,
            }

        self.regressor = Autoregressive(**self.AutoRegressive_cfg) if not cfg["contrast_attention"] else \
                         Autoregressive_Attention(**self.AutoRegressive_cfg)

        self.feature_dim = cfg["feature_dim"]

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: [B, L, 2, in_dim]; 
        # B is the batch size; L is the sequence length, in_dim is the input temporal dimension
        # output: [B, feature_dim]

        b = inputs.shape[0]
        device = inputs.device
        # assert t_d == self.AutoEncoder_cfg["in_dim"], \
        #     "Input temporal dimension should be the same as the in_dim in AutoEncoder"
        # assert c == self.encoder.in_channel, \
        #     "Input channel should be the same as the in_channel in AutoEncoder"
        encoder_outputs = torch.stack([self.encoder(inputs[i, ...])
                                       for i in range(b)]).to(device)
        # assert encoder_outputs.shape == (b, l, self.embed_dim), \
        #     "Encoder output shape should be [B, L, Embedding]"

        feature = self.regressor(encoder_outputs)  # [B, feature_dim]
        
        return feature + self.input_residue_proj(inputs)


class Feature_Extractor_cls(nn.Module):
    def __init__(self, cfg: dict = None, pos_type: str = "sine", 
                 in_type: str = "1d") -> None:
        super().__init__()
        
        self.feature_extractor = Feature_Extractor(cfg, pos_type, in_type)
        self.cls_head = nn.Linear(cfg["feature_dim"], cfg["num_classes"])
    
    def forward(self, inputs: Tensor) -> Tensor:
        feature = self.feature_extractor(inputs)
        return self.cls_head(feature)


def build_contrastive_model(cfg: dict = None, timestep: int = 12, pos_type: str = "sine", in_type: str = "1d") -> nn.Module:
    return Encoder_Regressor(cfg, timestep, pos_type, in_type=in_type)


def build_feature_extractor(cfg: dict = None, pos_type: str = "sine", in_type: str = "1d") -> nn.Module:
    return Feature_Extractor(cfg, pos_type, in_type)


def build_feature_extractor_cls(cfg: dict = None, pos_type: str = "sine", in_type: str = "1d") -> nn.Module:
    return Feature_Extractor_cls(cfg, pos_type, in_type)
