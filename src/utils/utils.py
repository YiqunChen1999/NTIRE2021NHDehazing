
r"""
Author:
    Yiqun Chen
Docs:
    Utilities, should not call other custom modules.
"""

_INFER_VERSION = 1

import os, sys, copy, functools, time, contextlib, math
import torch, torchvision
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from scipy import ndimage
from termcolor import colored


def notify(msg="", level="INFO", logger=None, fp=None):
    level = level.upper()
    if level == "WARNING":
        level = "[" + colored("{:<8}".format(level), "yellow") + "]"
    elif level == "ERROR":
        level = "[" + colored("{:<8}".format(level), "red") + "]"
    elif level == "INFO":
        level = "[" + colored("{:<8}".format(level), "blue") + "]"
    if logger is None:
        msg = "[{:<20}] {:<8} {}".format(time.asctime(), level, msg)
        _notify = print
    else:
        msg = "{:<8} {}".format(level, msg)
        _notify = logger.log_info
    _notify(msg)
    
    if fp is None:
        return
    elif isinstance(fp, str):
        try:
            with open(fp, 'a') as _fp:
                _fp.write(msg)
        except:
            notify(msg="Can not write message to file {}".format(fp), level="WARNING")
    else:
        try:
            fp.write(msg)
        except:
            notify(msg="Can not write message to file.", level="WARNING")


@contextlib.contextmanager
def log_info(msg="", level="INFO", state=False, logger=None):
    _state = "[" + colored("{:<8}".format("RUNNING"), "green") + "]" if state else ""
    notify(msg="{} {}".format(_state, msg), level=level, logger=logger)
    yield
    if state:
        _state = "[" + colored("{:<8}".format("DONE"), "green") + "]" if state else ""
        notify(msg="{} {}".format(_state, msg), level=level, logger=logger)


def inference(model, data, device):
    r"""
    Info:
        Inference once, without calculate any loss.
    Args:
        - model (nn.Module):
        - data (dict): necessary keys: "l_view", "r_view"
        - device (torch.device)
    Returns:
        - out (Tensor): predicted.
    """
    _INFER_FUNC = {}
    def add_infer_func(infer_func):
        _INFER_FUNC[infer_func.__name__] = infer_func
        return infer_func

    @add_infer_func
    def _inference_V1(model, data, device):
        src = data["src"]
        src = src.to(device)
        out = model(src)
        return out, 

    @add_infer_func
    def _inference_V2(model, data, device):
        src, dcp = data["src"], data["dcp"]
        src, dcp = src.to(device), dcp.to(device)
        src = torch.cat([src, dcp], dim=1)
        out = model(src)
        return out, 

    @add_infer_func
    def _inference_V3(model, data, device):
        # Serve for multi-resolution model
        src = data["src"]
        src = src.to(device)
        out = model(src)
        return out[0], out[1: ]

    # return _inference_V2(model, data, device) 
    return _INFER_FUNC["_inference_V{}".format(_INFER_VERSION)](model, data, device) 


def save_image(output, mean, norm, path2file):
    r"""
    Info:
        Save output to specific path.
    Args:
        - output (Tensor | ndarray): takes value from range [0, 1].
        - mean (float):
        - norm (float): 
        - path2file (str | os.PathLike):
    Returns:
        - (bool): indicate succeed or not.
    """
    if isinstance(output, torch.Tensor):
        output = output.numpy()
    output = ((output.transpose((1, 2, 0)) * norm) + mean).astype(np.uint8)
    try:
        cv2.imwrite(path2file, output)
        return True
    except:
        return False


def set_device(model: torch.nn.Module, gpu_list: list, logger=None):
    with log_info(msg="Set device for model.", level="INFO", state=True, logger=logger):
        if not torch.cuda.is_available():
            notify(msg="CUDA is not available, using CPU instead.", level="WARNING", logger=logger)
            device = torch.device("cpu")
        elif len(gpu_list) == 0:
            notify(msg="Use CPU.", level="INFO", logger=logger)
            device = torch.device("cpu")
        elif len(gpu_list) == 1:
            notify(msg="Use GPU {}.".format(gpu_list[0]), level="INFO", logger=logger)
            device = torch.device("cuda:{}".format(gpu_list[0]))
            model = model.to(device)
        elif len(gpu_list) > 1:
            raise NotImplementedError("Multi-GPU mode is not implemented yet.")
    return model, device

