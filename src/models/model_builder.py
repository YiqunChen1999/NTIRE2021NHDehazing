
"""
Author:
    Yiqun Chen
Docs:
    Build model from configurations.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
from torch import nn
import torch.nn.functional as F

from utils import utils
from .encoder import _ENCODER
from .decoder import _DECODER


class Model(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(Model, self).__init__()
        self.cfg = cfg
        self._build_model()

    def _build_model(self):
        self.encoder = _ENCODER[self.cfg.MODEL.ENCODER](self.cfg)
        self.decoder = _DECODER[self.cfg.MODEL.DECODER](self.cfg)
        
    def forward(self, data, *args, **kwargs):
        feats = self.encoder(data)
        out = self.decoder(feats)
        return out


def build_model(cfg, logger=None):
    with utils.log_info(msg="Build model from configurations.", level="INFO", state=True, logger=logger):
        model = Model(cfg)
    return model