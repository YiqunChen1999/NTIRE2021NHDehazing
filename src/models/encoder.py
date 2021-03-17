
"""
Author:
    Yiqun Chen
Docs:
    Encoder classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from utils import utils

_ENCODER = {}

def add_encoder(encoder):
    _ENCODER[encoder.__name__] = encoder
    return encoder
    

@add_encoder
class ResNeSt101EncoderV1(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(ResNeSt101EncoderV1, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
        self.normalize = transforms.Normalize(mean=[255*0.485, 255*0.456, 255*0.406], std=[255*0.229, 255*0.224, 255*0.225])

    def forward(self, x):
        x = self.normalize(x*255)
        feats = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        feats.append(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        feats.append(x)
        x = self.model.layer2(x)
        feats.append(x)
        x = self.model.layer3(x)
        feats.append(x)
        x = self.model.layer4(x)
        feats.append(x)
        return feats
    
    