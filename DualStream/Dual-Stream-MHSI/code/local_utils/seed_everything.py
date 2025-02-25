#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:59:13 2020
@author: Boxiang Yun   School:ECNU&HFUT   Email:971950297@qq.com
"""
import os
import random

# Third party libraries
import numpy as np
import torch


def seed_reproducer(seed=2020):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2020)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2020).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # set all gpus seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # if input data type and channels' changes arent' large use it improve train efficient
        torch.backends.cudnn.enabled = True
