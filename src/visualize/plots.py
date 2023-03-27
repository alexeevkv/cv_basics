import numpy as np
from matplotlib import pyplot as plt

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
