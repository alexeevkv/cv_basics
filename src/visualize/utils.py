import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_image_figsize(image, scale_coef=1):
    img_shape = np.array(image.shape[:2][::-1])

    return scale_coef * img_shape / np.gcd(*img_shape)


def generate_colors4labels(labels, colormap='hsv'):
    u_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap(colormap, len(u_labels) + 1)
    return {label:np.array(cmap(idx)[:3])*255 for idx, label in enumerate(u_labels)}


# TODO либо указать формат rectangle либо добавить форматирование (можно отдельную функцию)
def add_rectangle(img, rectengle, color=(255, 0, 0), label=None):
    img = img.copy()
    xmin, ymin, xmax, ymax = rectengle
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

    if label is not None:
        cv2.putText(img, str(label), (xmin, ymax + 20), cv2.FONT_ITALIC, 0.5, color, 2)

    return img

def add_rectangles(img, rectangles, labels=None, label2color=None):
    img = img.copy()

    if label2color is None:
        label2color = {None: (255, 0, 0)} if labels is None else generate_colors4labels(labels)

    if labels is None:
        labels = [None]*len(rectangles)

    for rectangle, label in zip(rectangles, labels):
        img = add_rectangle(img, rectengle=rectangle, color=label2color[label], label=label)
    
    return img

def add_mask(img, mask, color=(255, 255, 0), alpha=0.5):
    img = img.copy()

    color_mask = np.zeros(img.shape)
    for color_idx, color_i in enumerate(color):
        color_mask[:,:,color_idx] = mask*color_i
    color_mask = color_mask.astype(np.uint8)

    return cv2.addWeighted(img, 1, color_mask, alpha, 0)

