import torch
import numpy as np
from tqdm import tqdm

from .utils import to_np, to_cuda
from ..visualize.plots import plot_confusion_matrix, plot_representations
from ..visualize.utils import get_representations, get_pca, get_tsne


def get_predicts(net, dataloader):
    net.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            y_pred.extend(to_np(net(to_cuda(x, cuda=True))))
            y_true.extend(to_np(y))

    y_true, y_pred = np.vstack(y_true), np.vstack(y_pred)
    return y_true, np.argmax(y_pred, axis=1)


def get_weights_and_grads(named_parameters):
    avg_weights = list()
    avg_grads = list()

    for name, parameter in named_parameters:
        mean_weight = parameter.detach().cpu().abs().mean()
        mean_grad = parameter.grad.detach().abs().mean()

        avg_weights.append(mean_weight)
        avg_grads.append(mean_grad)

    avg_weight = sum(avg_weights) / len(avg_weights)
    avg_grad = sum(avg_grads) / len(avg_grads)

    avg_weight = float(avg_weight.cpu().numpy())
    avg_grad = float(avg_grad.cpu().numpy())

    return avg_weight, avg_grad


def transform_inputs(y_true, y_pred, task_type='classification'):
    if task_type == 'classification':
        y_pred_np = to_np(torch.argmax(y_pred, dim=1))
        y_true_np = to_np(y_true)   

        return y_true_np, y_pred_np


def compute_metrics(y_true, y_pred, metrics):
    y_true_transformed, y_pred_transformed = transform_inputs(y_true, y_pred, task_type='classification')

    computed_metrics = dict()

    for metric_name, metric_func in metrics.items():
        computed_metrics[metric_name] = metric_func(y_true_transformed, y_pred_transformed)

    return computed_metrics
