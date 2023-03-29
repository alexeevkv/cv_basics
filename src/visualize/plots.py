import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

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
