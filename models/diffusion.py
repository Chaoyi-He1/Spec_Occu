import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
from .positional_embedding import *
from .transformer import *
from typing import Optional, Tuple, Union, List, Dict


class VarianceSchedule(Module):
    """
    Variance schedule for diffusion process.
    Parameters
    ----------
    num_steps: int, number of steps in the diffusion process. (Markov chain length)
    mode: str, 'linear' or 'cosine', the mode of the variance schedule.
    beta_1: float, the initial value of beta.
    beta_T: float, the final value of beta.
    cosine_s: float, the cosine annealing start value.

    Attributes
    ----------
    betas: Tensor, [T+1], the beta values.
    alphas: Tensor, [T+1], the alpha values. alpha = 1 - beta
    alpha_bars: Tensor, [T+1], the cumulative sum of alpha. alpha_bar_t = sum_{i=0}^{t-1} alpha_i
    sigmas_flex: Tensor, [T+1], the flexible part of the variance schedule. sigma_t = sqrt(beta_t)
    sigmas_inflex: Tensor, [T+1], the inflexible part of the variance schedule. sigma_t = sqrt(beta_t)
    """
    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


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


class Conv1d_BN_Relu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv1d_BN_Relu, self).__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return super(Conv1d_BN_Relu, self).forward(x)


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


class Conv1d_encoder(nn.Module):
    def __init__(self, cfg: dict = None) -> None:
        super(Conv1d_encoder, self).__init__()
        assert cfg is not None, "cfg is None"

        self.channel = cfg["T2F_encoder_sequence_length"]
        self.temp_dim = cfg["T2F_encoder_embed_dim"]

        self.ResNet = nn.ModuleList()
        self.step_embedding = nn.ModuleList()
        self.context_embedding = nn.ModuleList()
        
        self.res_params = list(zip([4, 4, 8, 8, 6], [11, 9, 9, 7, 5],   # num_blocks, kernel_size
                                   [3, 3, 5, 3, 3], [1, 3, 5, 3, 3]))   # stride, dilation
        self.cum_blocks = np.cumsum([4, 4, 8, 8, 6]) + np.arange(5)
        for i, (num_blocks, kernel_size, stride, dilation) in enumerate(self.res_params):
            self.ResNet.extend([ResBlock_1d_with_Attention(self.channel, kernel_size, 
                                                           stride, self.temp_dim, dilation)
                                for _ in range(num_blocks)])
            self.step_embedding.append(nn.Embedding(cfg["diffusion_num_steps"], self.temp_dim))
            self.context_embedding.append(nn.Linear(cfg["feature_dim"], self.temp_dim))
            
            if i != len(self.res_params) - 1:
                pad = calculate_conv1d_padding(stride, kernel_size, self.temp_dim, self.temp_dim // 2, dilation)
                self.ResNet.append(Conv1d_BN_Relu(self.channel, self.channel * 2,
                                                  kernel_size, stride, pad, dilation))
                self.channel *= 2
                self.temp_dim //= 2
                
        self.channel_back_proj = nn.Conv1d(in_channels=self.channel, 
                                           out_channels=cfg["T2F_encoder_sequence_length"],
                                           kernel_size=1, stride=1, padding=0, dilation=1)
        # self.temp_proj_step_embed = nn.Embedding(cfg["diffusion_num_steps"], cfg["feature_dim"])
        self.temp_back_proj = ConcatSquashLinear(dim_in=self.temp_dim, 
                                                 dim_out=cfg["T2F_encoder_embed_dim"],
                                                 dim_ctx=cfg["feature_dim"] + 3)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def forward(self, inputs: Tensor, context: Tensor, t, beta) -> Tensor:
        """
        Args:
            inputs: [B, L, T]
                B: batch size
                L: sequence_length, length of time frames
                T: embed_dim, length of time dimension
        """
        x = inputs
        t = torch.tensor(t, requires_grad=False).to(inputs.device) - 1
        idx = 0
        # b, l, t_dim = inputs.shape
        for i, conv_layer in enumerate(self.ResNet):
            step_embed = self.step_embedding[idx](t)
            context_embed = self.context_embedding[idx](context)
            x = x + (step_embed + context_embed).unsqueeze(1)
            x = conv_layer(x)
            idx += 1 if i == self.cum_blocks[idx] else 0
        x = self.channel_back_proj(x)
        time_emb = torch.stack([beta, torch.sin(beta), 
                                torch.cos(beta)], dim=1).unsqueeze(1)
        ctx_emb = torch.cat([time_emb, context.unsqueeze(1)], dim=-1)
        x = self.temp_back_proj(ctx=ctx_emb, x=x)
        return x
    

class TrajNet(Module):
    def __init__(self, point_dim: int = 1024, time_embed_dim: int = 3,
                 context_dim: int = 256, seq_len: int = 32,
                 residual: bool = True):
        super(TrajNet, self).__init__()

        self.act = F.leaky_relu
        self.residual = residual
        self.time_embed_dim = time_embed_dim
        self.reduce_dim_conv = nn.Conv2d(in_channels=seq_len,
                                         out_channels=seq_len,
                                         kernel_size=(2, 1), stride=(1, 1), padding=(0, 0))
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 2048, context_dim + time_embed_dim),
            ConcatSquashLinear(2048, 2048, context_dim + time_embed_dim),
            ConcatSquashLinear(2048, 4096, context_dim + time_embed_dim),
            ConcatSquashLinear(4096, 2048, context_dim + time_embed_dim),
            ConcatSquashLinear(2048, 2048, context_dim + time_embed_dim),
            ConcatSquashLinear(2048, point_dim, context_dim + time_embed_dim),
        ])
        self.time_embed = None if time_embed_dim == 3 else \
                          PositionEmbeddingSine(context_dim, normalize=True)
        self.increment_dim_conv = nn.Conv2d(in_channels=seq_len,
                                            out_channels=seq_len,
                                            kernel_size=(2, 1), stride=(1, 1), padding=(1, 0))

    def forward(self, x, beta, context, t):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        out = self.reduce_dim_conv(x).squeeze(-2)      # (B, N, seq_len)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        # (B, 1, time_embed_dim)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1) \
                   if self.time_embed_dim == 3 else \
                   self.time_embed(context)
        
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        #pdb.set_trace()
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        out = out.unsqueeze(-2)
        out = self.increment_dim_conv(out)

        if self.residual:
            return x + out
        else:
            return out


class TransformerConcatLinear(Module):
    def __init__(self, cfg, point_dim, context_dim, embed_dim=512,
                 tf_layer=4, residual=True, seq_len=32):
        super().__init__()
        self.residual = residual
        self.init_norm = nn.BatchNorm2d(seq_len)
        self.reduce_dim_conv = nn.Conv2d(in_channels=seq_len,
                                         out_channels=seq_len,
                                         kernel_size=(2, 1), stride=(1, 1), padding=(0, 0))
        self.conv_1d_encoder = Conv1d_encoder(cfg=cfg)
        
        self.pos_emb = PositionEmbeddingSine(embed_dim, normalize=True)

        self.concat1 = ConcatSquashLinear(dim_in=point_dim, dim_out=embed_dim, 
                                          dim_ctx=context_dim + 3)
        
        self.encoder_param = {
            "num_layers": tf_layer,
            "d_model": embed_dim,
            "nhead": 8,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "drop_path": 0.0,
            "normalize_before": True,
            "ctx_dim": context_dim + 3,
        }
        self.transformer_encoder = ConcatTransformer_Encoder(**self.encoder_param)

        self.concat3 = ConcatSquashLinear(dim_in=embed_dim, dim_out=embed_dim,
                                          dim_ctx=context_dim + 3)
        self.concat4 = ConcatSquashLinear(dim_in=embed_dim, dim_out=embed_dim * 2,
                                          dim_ctx=context_dim + 3)
        
        self.linear = ConcatSquashLinear(dim_in=embed_dim * 2, dim_out=point_dim, 
                                         dim_ctx=context_dim + 3)
        
        self.increase_dim_conv = nn.Sequential(nn.Conv2d(in_channels=seq_len, out_channels=seq_len,
                                                         kernel_size=(2, 1), stride=(1, 1), 
                                                         padding=(1, 0)),
                                               nn.Conv2d(in_channels=seq_len, out_channels=seq_len,
                                                         kernel_size=(3, 3), stride=(1, 1),
                                                         padding=(1, 1)),
                                               nn.BatchNorm2d(seq_len),
                                               nn.Conv2d(in_channels=seq_len, out_channels=seq_len,
                                                         kernel_size=(3, 3), stride=(1, 1),
                                                         padding=(1, 1)))
        self.input_trans = nn.Sequential(
            nn.Conv2d(in_channels=seq_len, out_channels=seq_len,
                      kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            nn.Conv2d(in_channels=seq_len, out_channels=seq_len,
                      kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.Conv2d(in_channels=seq_len, out_channels=seq_len,
                      kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.Conv2d(in_channels=seq_len, out_channels=seq_len,
                      kernel_size=1, stride=1, padding=0),)         
        #self.linear = nn.Linear(128,2)

    def forward(self, x, beta, context, t):
        batch_size = x.size(0)
        out = self.init_norm(x)
        out = self.reduce_dim_conv(out).squeeze(-2)
        out = self.conv_1d_encoder(out, context, t, beta)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = self.concat1(ctx_emb, out)
        # final_emb = x.permute(1,0,2).contiguous()
        out += self.pos_emb(out)

        trans = self.transformer_encoder(out, ctx_emb)  # b * L+1 * diffusion_embed_dim

        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        trans = self.linear(ctx_emb, trans).unsqueeze(-2)
        x_ = self.input_trans(x)
        return self.increase_dim_conv(trans) + x_ if self.residual \
            else self.increase_dim_conv(trans)
    

class TransformerLinear(Module):
    def __init__(self, cfg, point_dim, context_dim, 
                 tf_layer=4, residual=True, seq_len=32) -> None:
        super().__init__()
        self.residual = residual
        self.reduce_dim_conv = nn.Conv2d(in_channels=seq_len,
                                         out_channels=seq_len,
                                         kernel_size=(2, 1), stride=(1, 1), padding=(0, 0))
        self.conv_1d_encoder = Conv1d_encoder(cfg=cfg)
        
        self.pos_emb = PositionEmbeddingSine(context_dim, normalize=True)

        self.y_up = nn.Linear(point_dim, 2048)
        self.ctx_up = nn.Linear(context_dim + 3, 2048)

        self.encoder_param = {
            "num_layers": tf_layer,
            "d_model": 2 * context_dim,
            "nhead": 8,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "normalize_before": True,
        }
        self.transformer_encoder = Transformer_Encoder(**self.encoder_param)

        self.linear = nn.Linear(2048, point_dim)
     
    def forward(self, x, beta, context, t):
        batch_size = x.size(0)
        x = self.reduce_dim_conv(x).squeeze(-2)
        x = self.conv_1d_encoder(x, context, t)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        ctx_emb = self.ctx_up(ctx_emb)
        emb = self.y_up(x)
        final_emb = torch.cat([ctx_emb, emb], dim=1)
        # pdb.set_trace()
        final_emb += self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb)  # b * L+1 * 128
        trans = trans[1:]   # B * L * 128, drop the first one which is the conditional feature
        return self.linear(trans)


class LinearDecoder(Module):
    def __init__(self, seq_len=32, cfg: dict = None) -> None:
            super().__init__()
            self.act = F.leaky_relu
            self.reduce_dim_conv = nn.Conv2d(in_channels=seq_len,
                                         out_channels=seq_len,
                                         kernel_size=(2, 1), stride=(1, 1), padding=(0, 0))
            self.conv_1d_encoder = Conv1d_encoder(cfg=cfg)
            
            self.layers = ModuleList([
                #nn.Linear(2, 64),
                nn.Linear(32, 64),
                nn.Linear(64, 128),
                nn.Linear(128, 256),
                nn.Linear(256, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 12)
                #nn.Linear(2, 64),
                #nn.Linear(2, 64),
            ])
    def forward(self, code, context, t):
        code = self.reduce_dim_conv(code).squeeze(-2)
        code = self.conv_1d_encoder(code, context, t)
        out = code
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        return out


def build_diffusion_model(diffnet: str = "TransformerConcatLinear",
                          cfg: dict = None):
    transformer_param = {
        "point_dim": cfg["Temporal_dim"],
        "context_dim": cfg["feature_dim"],
        "tf_layer": cfg["diffu_num_trans_layers"],
        "residual": cfg["diffu_residual_trans"],
        "seq_len": cfg["T2F_encoder_sequence_length"],
        "embed_dim": cfg["diffusion_embed_dim"], 
        "cfg": cfg
    }
    if diffnet == "TransformerConcatLinear":
        return TransformerConcatLinear(**transformer_param)
    elif diffnet == "TransformerLinear":
        return TransformerLinear(**transformer_param)
    elif diffnet == "LinearDecoder":
        return LinearDecoder(cfg["T2F_encoder_sequence_length"], cfg)
    elif diffnet == "TrajNet":
        return TrajNet(point_dim=cfg["Temporal_dim"], time_embed_dim=3,
                       context_dim=cfg["feature_dim"], seq_len=cfg["T2F_encoder_sequence_length"])
    else:
        raise NotImplementedError
