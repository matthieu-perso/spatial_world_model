# ==================
# function: set seed
# ==================

import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)

    # ensures deterministic operations on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable benchmark mode
