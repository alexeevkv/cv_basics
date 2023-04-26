import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn import manifold

from src.torch.utils import to_np, to_cuda


def get_image_figsize(image, scale_coef=1):
    img_shape = np.array(image.shape[:2][::-1])

    return scale_coef * img_shape / np.gcd(*img_shape)


def generate_colors4labels(labels, colormap="hsv"):
    u_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap(colormap, len(u_labels) + 1)
    return {label: np.array(cmap(idx)[:3]) * 255 for idx, label in enumerate(u_labels)}


def get_representations(model, dataloader):
    model.eval()

    outputs = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch

            y_pred = model(to_cuda(x))

            outputs.extend(to_np(y_pred))
            labels.extend(to_np(y))
            
    return np.asarray(outputs), np.asarray(labels)


def get_pca(data, n_components=2):
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data


def get_tsne(data, n_components=2, n_images=None):
    if n_images is not None:
        data = data[:n_images]

    tsne = manifold.TSNE(n_components=n_components, random_state=0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data


def vis_batch(x, y=None, idx2class: dict = None):
    """
    Inputs:
    x - Tensor representening the batch of images with shape [B, C, H, W]
    """
    batch_size = x.shape[0]
    fig, ax = plt.subplots(nrows=1, ncols=batch_size, figsize=(12, 12))

    for idx, image in enumerate(x):
        ax[idx].imshow(to_np(image).transpose(1, 2, 0))
        ax[idx].axis("off")
        if y is not None:
            title = idx2class[int(y[idx])] if idx2class is not None else int(y[idx])
            ax[idx].set_title(title)
    plt.show()
