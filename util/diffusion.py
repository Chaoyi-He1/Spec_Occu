import torch
import torch.nn as nn
from torch import Tensor
from models.diffusion import VarianceSchedule
import torch.nn.functional as F
import pdb


class Diffusion_utils(nn.Module):
    def __init__(self, var_sched: VarianceSchedule):
        super().__init__()
        self.var_sched = var_sched


    def get_loss(self, x_0, context, t=None, model: nn.Module=None):
        """
        Diffusion loss.
        Based on Denoising Diffusion Probabilistic Models
        equation (14) in
        https://arxiv.org/abs/2006.11239
        Loss = ||\epsilon - \epsilon_theta(\sqrt(\alpha_bar_t x0) + \sqrt(1 - \alpha_bar_t \epsilon)
                                          , t)||^2
        """
        batch_size, _, _ = x_0.size()   # (B, N, d)
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1)

        e_rand = torch.randn_like(x_0).cuda()  # (B, N, d)

        e_theta = model(c0 * x_0 + c1 * e_rand, beta=beta, context=context)
        loss = F.mse_loss(e_theta, e_rand, reduction='mean')
        return loss


    def sample(self, num_points, context, sample, bestof, model: nn.Module,
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
                e_theta = model(x_t, beta=beta, context=context)
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


def compute_batch_statistics(predictions, gt_future):
    """
    Args:
        predictions: (R, B, N, d), the generated future signal from the model
        gt_future: (B, N, d), the ground truth future signal
    ADE error: average displacement error
    FDE error: final displacement error
    """
    r, _, _, _ = predictions.shape
    errors = torch.sqrt(((predictions.sum(dim=0) / r - 
                          gt_future)**2).sum(dim=2))  # (B, N)
    ADE = errors.mean(dim=1)  # (B, )
    FDE = errors[:, -1]  # (B, )
    ADE_percents = ADE / torch.sqrt((gt_future**2).sum(dim=2)).mean(dim=1)
    FDE_percents = FDE / torch.sqrt((gt_future**2).sum(dim=2))[:, -1]
    return ADE, FDE, ADE_percents, FDE_percents
