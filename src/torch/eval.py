from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch
from .utils import to_np, to_cuda


def predict(net, batch_iter, verbose=False):
    net.eval()

    ys = []
    if verbose:
        batch_iter = tqdm(batch_iter)
    with torch.no_grad():
        for batch in batch_iter:
            x = batch['image']
            
            ys.extend(to_np(net(to_cuda(x, cuda=True))))
    return ys


def write_metrics(writer: SummaryWriter, name2metric, step):
    for metric, value in name2metric.items():
        writer.add_scalar(metric, value, step)
