"""
Semantic Segmentation loss functions for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import torch
from torch import nn


class VarianceAttenuationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, log_variance):

        res = (output - target) ** 2
        coef = torch.exp(-log_variance)
        loss = .5 * coef * res + .5 * log_variance
        return loss.mean()

def exp_buckets(size, bin_size=10, base = 1.1):
    i = 0
    y = []
    done = bin_size
    bins = []
    idxs = []

    for j in range(size):
        if j >= done and (size - done) > bin_size * base**(i+1):
            i += 1
            done += bin_size * base**i
            bins.append(torch.tensor(idxs))
            idxs = []
        y.append(i % 2)
        idxs.append(size -1 - j)
    bins.append(torch.tensor(idxs))
    return bins

class VarianceAttenuationBinsLoss(nn.Module):
    def __init__(self, size, small_bin=10, base=1.1):
        super().__init__()

        self.bins = exp_buckets(size, bin_size=small_bin, base=base)

    def compute_loss(self, output:torch.Tensor, target:torch.Tensor, log_variance:torch.Tensor):

        res = (output - target) ** 2
        coef = torch.exp(-log_variance)
        loss = .5 * coef * res + .5 * log_variance
        return loss.mean()

    def forward(self, output:torch.Tensor, target:torch.Tensor, log_variance:torch.Tensor):
        vals = []
        
        for bin in self.bins:
            vals +=[
                self.compute_loss(
                    output=output[:,bin],
                    target=target[:,bin],
                    log_variance=log_variance[:,bin],
                )
            ]
        return torch.stack(vals).mean()


class WeightedVarianceAttenuationLoss(nn.Module):
    def __init__(self, b=.5):
        super().__init__()
        self.b = b

    def forward(self, output:torch.Tensor, target:torch.Tensor, log_variance:torch.Tensor) -> torch.Tensor:
        res = (output - target) ** 2
        coef = torch.exp(-log_variance) # 1/s_2
        loss = .5 * coef * res + .5 * log_variance
        if self.b > 0:
            weight = torch.pow(coef.detach(), -self.b) # (1/s_2) ** (-b) = (s_2)**b = s_2b
            loss = loss*weight
        return loss.mean()