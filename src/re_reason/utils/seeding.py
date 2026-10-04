"""Deterministic seeding utilities for reproducible execution across CPU and GPU/ROCm."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> int:
    """Sets random seeds across Python, NumPy, and PyTorch (CPU and ROCm/CUDA).

    Args:
        seed: Integer seed value. Defaults to 42.

    Returns:
        The seed integer applied.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
