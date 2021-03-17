
"""
Author:
    Yiqun Chen
Docs:
    Decoder classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
from collections import OrderedDict
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
from models.modules import *

_DECODER = {}

def add_decoder(decoder):
    _DECODER[decoder.__name__] = decoder
    return decoder


@add_decoder
class ResNeSt101DecoderV1(nn.Module):
    """
    """
    def __init__(self, cfg, *args, **kwargs):
        super(ResNeSt101DecoderV1, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.block_4_1, self.block_4_2 = self._build_block(2, 2048, 1024)
        self.block_3_1, self.block_3_2 = self._build_block(2, 1024, 512)
        self.block_2_1, self.block_2_2 = self._build_block(2, 512, 256)
        self.block_1_1, self.block_1_2 = self._build_block(2, 256, 128)

        self.nhdb_5 = NHDBlockV3(4, 2048, 512)
        self.nhdb_4 = NHDBlockV3(4, 1024, 256)
        self.nhdb_3 = NHDBlockV3(4, 512, 128)
        self.nhdb_2 = NHDBlockV3(4, 256, 64)
        self.nhdb_1 = NHDBlockV3(4, 128, 32)
        
        self.out_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 3, 3, stride=1, padding=1), 
            nn.Sigmoid(), 
        )

    def _build_block(self, num_conv, in_channels, out_channels):
        block_1 = nn.Sequential(OrderedDict([
            ("upsampling", nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)), 
            ("relu", nn.ReLU()), 
        ]))
        layer_list = []
        for idx in range(num_conv):
            layer_list.extend([
                ("conv_"+str(idx), nn.Conv2d(out_channels*2 if idx == 0 else out_channels, out_channels, 3, stride=1, padding=1)), 
                ("relu_"+str(idx), nn.ReLU()), 
            ])
        block_2 = nn.Sequential(OrderedDict(layer_list))
        return block_1, block_2

    def forward(self, inp, *args, **kwargs):
        feat_enc_1, feat_enc_2, feat_enc_3, feat_enc_4, feat_enc_5 = inp
        
        feat_dec_5 = self.nhdb_5(feat_enc_5)

        # decoder block 4
        feat_dec_4 = self.block_4_1(feat_dec_5)
        feat_dec_4 = F.interpolate(feat_dec_4, feat_enc_4.shape[2: ])
        feat_dec_4 = torch.cat([feat_dec_4, self.nhdb_4(feat_enc_4)], dim=1)
        feat_dec_4 = self.block_4_2(feat_dec_4)

        # decoder block 3
        feat_dec_3 = self.block_3_1(feat_dec_4)
        feat_dec_3 = F.interpolate(feat_dec_3, feat_enc_3.shape[2: ])
        feat_dec_3 = torch.cat([feat_dec_3, self.nhdb_3(feat_enc_3)], dim=1)
        feat_dec_3 = self.block_3_2(feat_dec_3)

        # decoder block 2
        feat_dec_2 = self.block_2_1(feat_dec_3)
        feat_dec_2 = F.interpolate(feat_dec_2, feat_enc_2.shape[2: ])
        feat_dec_2 = torch.cat([feat_dec_2, self.nhdb_2(feat_enc_2)], dim=1)
        feat_dec_2 = self.block_2_2(feat_dec_2)

        # decoder block 1
        feat_dec_1 = self.block_1_1(feat_dec_2)
        feat_dec_1 = F.interpolate(feat_dec_1, feat_enc_1.shape[2: ])
        feat_dec_1 = torch.cat([feat_dec_1, self.nhdb_1(feat_enc_1)], dim=1)
        feat_dec_1 = self.block_1_2(feat_dec_1)

        out = self.out_block(feat_dec_1)
        return out

