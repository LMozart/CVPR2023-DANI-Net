import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Code reimplemented from: https://github.com/junxuan-li/SCPS-NIR
class ASG(nn.Module):
    def __init__(
            self,
            num_bases,
            k_low,
            k_high,
            trainable_k,
    ):
        super(ASG, self).__init__()
        self.num_bases = num_bases

        self.trainable_k = trainable_k
        if self.trainable_k:
            kh = math.log10(k_high)
            kl = math.log10(k_low)
            self.k_x = nn.Parameter((torch.linspace(kh, kl, num_bases, dtype=torch.float32))[None, :])
            self.k_y = nn.Parameter((torch.linspace(kh, kl, num_bases, dtype=torch.float32))[None, :])
        else:
            kh = math.log10(k_high)
            kl = math.log10(k_low)
            self.k = torch.linspace(kh, kl, num_bases, dtype=torch.float32)[None, :]

    def forward(self, light, normal, mu, view=None):
        if view is None:
            view = torch.zeros_like(light)
            view[..., 2] = -1
            view = view.detach()
        light, view, normal = F.normalize(light, p=2, dim=-1), F.normalize(view, p=2, dim=-1), F.normalize(normal, p=2, dim=-1)
        H = F.normalize((view + light) / 2, p=2, dim=-1)
        if self.trainable_k:
            k_x = self.k_x
            k_y = self.k_y
        else:
            k_x = self.k_x.to(light.device)
            k_x = self.k_y.to(light.device)

        rate_x = (10 ** k_x).clip(1, 1000) # range: 1 ~ 1000
        rate_y = (10 ** k_y).clip(1, 1000) # range: 1 ~ 1000
        
        # Anisotropic
        x, y = self.asg_axis(normal, view)
        out = mu * torch.exp(-rate_x * ((H * x).sum(dim=-1, keepdim=True)**2) - rate_y * ((H * y).sum(dim=-1, keepdim=True)**2))[..., None]  # (batch, num_bases, 3)
        return out

    def asg_axis(self, normal, ld):
        # Anisotropic axis calculation, Sec 4.2.
        proj_n = (ld * normal).sum(-1, keepdim=True) * normal
        x      = torch.nn.functional.normalize(ld - proj_n)
        x_len  = (x * x).sum(-1)

        # Prevent axis x from [0., 0., 0.]
        ids    = torch.where(x_len == 0.)
        x[ids] = torch.tensor([1., 0., 0.], device=x.device)
        y      = torch.nn.functional.normalize(torch.cross(normal, x, dim=-1), dim=-1)
        return x, y

def dynamic_basis(input, current_epoch, total_epoch, num_bases):
    """
    Args:
        input:  (batch, num_bases, 3)
        current_epoch:
        total_epoch:
        num_bases:
    Returns:
    """
    alpha = current_epoch / total_epoch * (num_bases)
    k = torch.arange(num_bases, dtype=torch.float32, device=input.device)
    weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(math.pi).cos_()) / 2
    weight = weight[None, :, None]
    weighted_input = input * weight
    return weighted_input