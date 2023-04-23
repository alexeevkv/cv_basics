import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from ..io.utils import resolve_outpath


def resolve_cuda(cuda=None):
    """Return true if cuda=True or if cuda is available, otherwise returns False"""
    if cuda is None:
        return torch.cuda.is_available()
    else:
        return cuda


def to_cuda(x, cuda=None):
    """Send module to cuda, if possible."""
    cuda = resolve_cuda(cuda)
    if cuda or (cuda is None and torch.cuda.is_available()):
        x = x.cuda()
    return x


def to_np(x: torch.Tensor) -> np.ndarray:
    """Send array to cpu"""
    return x.data.cpu().numpy()


def to_var(x: np.ndarray, cuda: bool = None, requires_grad: bool = False) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(x))
    if requires_grad:
        x.requires_grad_()
    return to_cuda(x, cuda)


def save_model_state(model: nn.Module, path):
    resolve_outpath(path)
    torch.save(model.state_dict(), path)


def load_model_state(model: nn.Module, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model


def train_test_val_split(dataset, test_size, val_size, random_state, stratify_by_cols):
    train, test = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset[stratify_by_cols] if stratify_by_cols else None
    )

    dataset2train_proportion = len(dataset) / len(train)

    train, val = train_test_split(
        train,
        test_size=dataset2train_proportion * val_size,
        random_state=random_state,
        stratify=train[stratify_by_cols] if stratify_by_cols else None
    )

    return train, test, val


def select_device(raw_config):
    device = raw_config['DEVICE']
    if device == 'cuda' and raw_config['GPU_NUM'] == 1 and 'DEVICE_NUM' in raw_config.keys():
        os.environ['CUDA_VISIBLE_DEVICES'] = raw_config['DEVICE_NUM']
    return device
