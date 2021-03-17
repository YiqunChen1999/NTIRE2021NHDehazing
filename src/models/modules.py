
"""
Author:
    Yiqun Chen
Docs:
    Necessary modules for model.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils


class NHDBlockV3(nn.Module):
    """
    Info:
        Variation of NHDBlockV1, add residual connection from input to output.
    """
    def __init__(self, num_blocks, in_channels, hidden_channels, *args, **kwargs):
        super(NHDBlockV3, self).__init__()
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self._build()

    def _build(self):
        self.model = nn.Sequential(*[NHDUnitV2(self.in_channels, self.hidden_channels) for i in range(self.num_blocks)])

    def forward(self, inp):
        feat = inp + self.model(inp)
        return feat


class NHDUnitV2(nn.Module):
    def __init__(self, in_channels, hidden_channels, *args, **kwargs):
        super(NHDUnitV2, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self._build()

    def _build(self):
        self.conv_1 = nn.Conv2d(self.in_channels, self.hidden_channels, 1)
        self.conv_2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, stride=1, padding=1)
        self.conv_trans_1 = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1)
        self.conv_trans_2 = nn.Conv2d(self.hidden_channels, 1, 1)
        self.conv_atmos = nn.Conv2d(self.hidden_channels, self.in_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inp):
        feat = self.relu(self.conv_2(self.relu(self.conv_1(inp))))
        trans = self.sigmoid(self.conv_trans_2(self.relu(self.conv_trans_1(feat))))
        atmos = self.sigmoid(self.conv_atmos(self.global_avg(feat)))
        out = inp * trans + (1 - trans) * atmos
        return out


