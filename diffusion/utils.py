from datetime import datetime
from functools import lru_cache
import json
import os
import os.path as osp
import torch
from PIL import Image

DATASET_ROOT = os.getenv('DATASET_ROOT', 'saba/datasets')
LOG_DIR = os.getenv('LOG_DIR', '/work/gn21/h62001/Diffusion-Classifier/data')


def get_formatstr(n):
    # get the format string that pads 0s to the left of a number, which is at most n
    digits = 0
    while n > 0:
        digits += 1
        n //= 10
    return f"{{:0{digits}d}}"

