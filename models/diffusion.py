import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
import pdb
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


class DiffusionTraj(Module):
    def __init__(self, model, var_sched:VarianceSchedule):
        super().__init__()
        self.model = model
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):
        """
        Diffusion loss.
        Based on Denoising Diffusion Probabilistic Models
        equation (14) in
        https://arxiv.org/abs/2006.11239
        Loss = ||\epsilon - \epsilon_theta(\sqrt(\alpha_bar_t x0) + \sqrt(1 - \alpha_bar_t \epsilon)
                                          , t)||^2
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1)

        e_rand = torch.randn_like(x_0).cuda()  # (B, N, d)

        e_theta = self.model(c0 * x_0 + c1 * e_rand, beta=beta, context=context)
        loss = F.mse_loss(e_theta, e_rand, reduction='mean')
        return loss

    def sample(self, num_points, context, sample, bestof, 
               point_dim=2, flexibility=0.0, ret_traj=False, sampling="ddpm", step=1):
        """
        Sample from the diffusion model.
        DDPM: Denoising Diffusion Probabilistic Models
        https://arxiv.org/abs/2006.11239
        DDIM: Denoising Diffusion Implicit Models
        https://arxiv.org/abs/2010.02502
        """
        traj_list = []
        for _ in range(sample):
            batch_size = context.size(0)
            if bestof:
                x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
            else:
                x_T = torch.zeros([batch_size, num_points, point_dim]).to(context.device)
            traj = {self.var_sched.num_steps: x_T}
            stride = step

            for t in range(self.var_sched.num_steps, 0, -stride):
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                alpha_bar_next = self.var_sched.alpha_bars[t-stride]    # next: closer to 1
                # pdb.set_trace()
                sigma = self.var_sched.get_sigmas(t, flexibility)

                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                x_t = traj[t]
                beta = self.var_sched.betas[[t] * batch_size]
                e_theta = self.model(x_t, beta=beta, context=context)
                if sampling == "ddpm":
                    x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                elif sampling == "ddim":
                    x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * e_theta
                else:
                    pdb.set_trace()
                traj[t-stride] = x_next.detach()     # Stop gradient and save trajectory.
                traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
                if not ret_traj:
                   del traj[t]

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])
        return torch.stack(traj_list)

class TrajNet(Module):

    def __init__(self, point_dim: int = 1024, time_embed_dim: int = 256,
                 context_dim: int = 256, 
                 residual: bool = True):
        super(TrajNet, self).__init__()

        self.act = F.leaky_relu
        self.residual = residual
        self.time_embed_dim = time_embed_dim
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
    def __init__(self, point_dim, context_dim, tf_layer=4, residual=True):
        super().__init__()
        self.residual = residual
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
    def __init__(self, point_dim, context_dim, tf_layer=4, residual=True) -> None:
        super().__init__()
        self.residual = residual
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
    def __init__(self):
            super().__init__()
            self.act = F.leaky_relu
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
        out = code
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        return out


def build_diffusion_model(diffnet: str = "TransformerConcatLinear",
                          Temporal_dim: int = 1024, feature_dim: int = 128, 
                          num_trans_layers: int = 4, residual_trans: bool = True):
    transformer_param = {
        "point_dim": Temporal_dim,
        "context_dim": feature_dim,
        "tf_layer": num_trans_layers,
        "residual": residual_trans,
    }
    if diffnet == "TransformerConcatLinear":
        return TransformerConcatLinear(**transformer_param)
    elif diffnet == "TransformerLinear":
        return TransformerLinear(**transformer_param)
    elif diffnet == "LinearDecoder":
        return LinearDecoder()
