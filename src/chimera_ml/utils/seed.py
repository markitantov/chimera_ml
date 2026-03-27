import os
import random

import numpy as np
import torch


def define_seed(seed: int = 0) -> None:
    """Fix random seed for reproducibility.

    This function fixes:
    - Python random
    - NumPy
    - PyTorch (CPU & CUDA)
    - cuDNN deterministic behavior
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
