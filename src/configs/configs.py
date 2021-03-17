
r"""
Author:
    Yiqun Chen
Docs:
    Configurations, should not call other custom modules.
"""

import os, sys, copy, argparse
from attribdict import AttribDict as Dict

configs = Dict()
cfg = configs

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--resume", default="false", choices=["true", "false"], type=str, required=True)
parser.add_argument("--gpu", type=str, required=True)
args = parser.parse_args()

# ================================ 
# GENERAL
# ================================ 
cfg.GENERAL.ROOT                                =   os.path.join(os.getcwd(), ".")
cfg.GENERAL.ID                                  =   "{}".format(args.id)
cfg.GENERAL.BATCH_SIZE                          =   args.batch_size
cfg.GENERAL.RESUME                              =   True if args.resume == "true" else False
cfg.GENERAL.GPU                                 =   eval(args.gpu)

# ================================ 
# MODEL
# ================================ 
cfg.MODEL.ENCODER                               =   "ResNeSt101EncoderV1" 
cfg.MODEL.DECODER                               =   "ResNeSt101DecoderV1" 
cfg.MODEL.CKPT_DIR                              =   os.path.join(cfg.GENERAL.ROOT, "checkpoints", cfg.GENERAL.ID)
cfg.MODEL.PATH2CKPT                             =   os.path.join(cfg.MODEL.CKPT_DIR, "checkpoints.pth")

# ================================ 
# DATA
# ================================ 
cfg.DATA.DIR                                    =   {
    "NTIRE2021NHHAZE": "/home/chenyiqun/data/NTIRE2021NHHAZE", 
}
cfg.DATA.NUMWORKERS                             =   4
cfg.DATA.DATASET                                =   args.dataset # "MNHHAZE"
cfg.DATA.MEAN                                   =   [0., 0., 0.]
cfg.DATA.NORM                                   =   [255, 255, 255]
cfg.DATA.AUGMENTATION                           =   True

# ================================ 
# SAVE
# ================================ 
cfg.SAVE.DIR                                    =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "results", cfg.GENERAL.ID, cfg.DATA.DATASET))
cfg.SAVE.SAVE                                   =   True


assert cfg.DATA.DATASET in cfg.DATA.DIR.keys(), "Unknown dataset {}".format(cfg.DATA.DATASET)

_paths = [
    cfg.SAVE.DIR, 
]
_paths.extend(list(cfg.DATA.DIR.as_dict().values()))

for _path in _paths:
    if not os.path.exists(_path):
        os.makedirs(_path)

