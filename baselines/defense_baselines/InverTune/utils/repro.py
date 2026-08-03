import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    path = Path(path).resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)


def cpu_state_dict(model):
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
