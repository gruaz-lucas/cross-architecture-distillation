import torch
import random
import numpy as np
import yaml
from types import SimpleNamespace

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return SimpleNamespace(**cfg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
