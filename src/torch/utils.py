import warnings

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


def check_required_key(conf, key=''):
    if key not in conf.keys():
        raise ValueError(f"For {conf['model_type']} type it is absolutely necessary to specify {key}")


def load_net(net_conf):
    '''
    Function is used for loading DL models. You can use in three scenarios:
        1. you can use pretrained_models from torchvision.models or others but with similar api.
            you can also pass weights, but if you dont want to, you should specify them to null in config
        2. you can load model from pytorch state_dict 
    '''
    model_type = net_conf['model_type']

    if model_type == 'trained_models':
        check_required_key(net_conf, key='model_path')
        check_required_key(net_conf, key='device')
        return torch.load(net_conf['model_path'], map_location=torch.device(net_conf['device']))
    
    if model_type == 'state_dict_models':
        check_required_key(net_conf, key='net')
        check_required_key(net_conf, key='path2state_dict')
        net = net_conf['net']
        return load_model_state(net, net_conf['path2state_dict'])

    if model_type == 'torch_models':
        check_required_key(net_conf, key='net')
        check_required_key(net_conf, key='num_classes')
        if 'weights' not in net_conf.keys():
            warnings.warn(f'You are using {model_type} type but have not specified weights to be None!')
            weights = None
        else:
            weights = net_conf['weights']   

        return net_conf['net'](pretrained=False, num_classes=net_conf['num_classes'])
