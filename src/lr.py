import random
from typing import Tuple

import numpy as np

import torch
from torch import Tensor, nn


_Optimizer = torch.optim.Optimizer


def seed_everything(seed: int) -> None:
    """
    시드 고정 method

    :param seed: 시드
    :type seed: int
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer: _Optimizer) -> float:
    """
    optimizer 통해 lr 얻는 method

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :return: learning_rate
    :rtype: float
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]