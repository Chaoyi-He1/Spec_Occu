import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
from .positional_embedding import *
from .transformer import *


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


class TrajNet(Module):

    def __init__(self, point_dim: int = 1024, time_embed_dim: int = 256,
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

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        x = self.reduce_dim_conv(x).squeeze(-2)      # (B, N, seq_len)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        # (B, 1, time_embed_dim)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1) \
                   if self.time_embed_dim == 3 else \
                   self.time_embed(context)
        
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        #pdb.set_trace()
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class TransformerConcatLinear(Module):
    def __init__(self, point_dim, context_dim, tf_layer=4, residual=True, seq_len=32):
        super().__init__()
        self.residual = residual
        self.reduce_dim_conv = nn.Conv2d(in_channels=seq_len,
                                         out_channels=seq_len,
                                         kernel_size=(2, 1), stride=(1, 1), padding=(0, 0))
        self.pos_emb = PositionEmbeddingSine(2 * context_dim, normalize=True)

        self.concat1 = ConcatSquashLinear(dim_in=point_dim, dim_out=2 * context_dim, 
                                          dim_ctx=context_dim+3)
        
        self.encoder_param = {
            "num_layers": tf_layer,
            "d_model": 2 * context_dim,
            "nhead": 8,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "normalize_before": True,
        }
        self.transformer_encoder = Transformer_Encoder(**self.encoder_param)

        self.concat3 = ConcatSquashLinear(dim_in=2 * context_dim, dim_out=context_dim,
                                          dim_ctx=context_dim + 3)
        self.concat4 = ConcatSquashLinear(dim_in=context_dim, dim_out=context_dim // 2,
                                          dim_ctx=context_dim + 3)
        
        self.linear = ConcatSquashLinear(dim_in=context_dim // 2, dim_out=point_dim, 
                                         dim_ctx=context_dim + 3)
        #self.linear = nn.Linear(128,2)

    def forward(self, x, beta, context):
        batch_size = x.size(0)
        x = self.reduce_dim_conv(x).squeeze(-2)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        x = self.concat1(ctx_emb, x)
        # final_emb = x.permute(1,0,2).contiguous()
        x += self.pos_emb(x)

        trans = self.transformer_encoder(x)

        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)


class TransformerLinear(Module):
    def __init__(self, point_dim, context_dim, tf_layer=4, residual=True, seq_len=32) -> None:
        super().__init__()
        self.residual = residual
        self.reduce_dim_conv = nn.Conv2d(in_channels=seq_len,
                                         out_channels=seq_len,
                                         kernel_size=(2, 1), stride=(1, 1), padding=(0, 0))
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
     
    def forward(self, x, beta, context):
        batch_size = x.size(0)
        x = self.reduce_dim_conv(x).squeeze(-2)
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
    def __init__(self, seq_len=32):
            super().__init__()
            self.act = F.leaky_relu
            self.reduce_dim_conv = nn.Conv2d(in_channels=seq_len,
                                         out_channels=seq_len,
                                         kernel_size=(2, 1), stride=(1, 1), padding=(0, 0))
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
    def forward(self, code):
        code = self.reduce_dim_conv(code).squeeze(-2)
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
        "seq_len": cfg["T2F_encoder_sequence_length"]
    }
    if diffnet == "TransformerConcatLinear":
        return TransformerConcatLinear(**transformer_param)
    elif diffnet == "TransformerLinear":
        return TransformerLinear(**transformer_param)
    elif diffnet == "LinearDecoder":
        return LinearDecoder(cfg["T2F_encoder_sequence_length"])
    else:
        raise NotImplementedError
