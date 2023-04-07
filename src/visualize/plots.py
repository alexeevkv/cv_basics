import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix

from .utils import get_image_figsize


def show_images(images: dict, scale_coef=1):
    n = len(images)
    figsize = get_image_figsize(images[list(images.keys())[0]], scale_coef=scale_coef)
    fig, ax = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))

    if n == 1:
        ax = [ax]

    for a, key in zip(ax, images.keys()):
        a.imshow(images[key])
        a.set_title(key.title())

    fig.tight_layout()


def plot_confusion_matrix(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 10))
    sns.heatmap(CM, annot=True)
    return plt.gcf()


def plot_representations(data, labels, title="", n_images=None):
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="hsv")
    ax.set_title(title)
    ax.grid()

    return fig


def plot_grad_flow(named_parameters):
    '''
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    '''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return plt.gcf()
