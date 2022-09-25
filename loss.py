import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence


###############################################################################################################
class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        input = torch.log(input)
        target = torch.log(target)
        g = input - target

        mse = nn.MSELoss()
        Dg = torch.var(g) + mse(input, target)
        # Dg = torch.mean(torch.pow(g, 2)) - 0.5 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)
        # return Dg
