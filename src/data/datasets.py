
"""
Author:
    Yiqun Chen
Docs:
    Dataset classes.
"""

import os, sys, random
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2, copy
from tqdm import tqdm
import numpy as np

from utils import utils

_DATASET = {}

def add_dataset(dataset):
    _DATASET[dataset.__name__] = dataset
    return dataset


@add_dataset
class NTIRE2021NHHAZE(torch.utils.data.Dataset):
    r"""
    Info:
        Non-Homogeneous Hazy dataset, with following structure:
        test:
            source:
                img_idx_0.png
                img_idx_1.png
                ...
    """
    def __init__(self, cfg, split, *args, **kwargs):
        super(NTIRE2021NHHAZE, self).__init__()
        self.cfg = cfg
        assert split in ["test"], "Unknown dataset split {}".format(split)
        self.split = split
        self.path2dataset = self.cfg.DATA.DIR.NTIRE2021NHHAZE
        self._build()

    def _build(self):
        self.items = []
        
        src_items = sorted(os.listdir(os.path.join(self.path2dataset, self.split, "source")))
        for idx in range(len(src_items)):
            item = {
                "img_idx": src_items[idx].split(".")[0], 
                "source": src_items[idx], 
            }
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data = {}
        
        path2src = os.path.join(self.path2dataset, self.split, "source")
        info = self.items[idx]
        
        src = cv2.imread(os.path.join(path2src, info["source"]), -1).transpose(2, 0, 1).astype(np.float32)
        data["src"] = (torch.from_numpy(src) - torch.tensor(self.cfg.DATA.MEAN).unsqueeze(1).unsqueeze(1)) / torch.tensor(self.cfg.DATA.NORM).unsqueeze(1).unsqueeze(1)
        
        data["img_idx"] = info["img_idx"]
        return data

        