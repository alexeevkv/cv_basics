import torch
from skimage.util import random_noise
from torchvision import transforms


MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


class GaussianNoise(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class PepperNoise(object):
    def __init__(self, amount=0.1):
        self.amount = amount

    def __call__(self, tensor):
        return torch.tensor(random_noise(tensor, mode="pepper", amount=self.amount))

    def __repr__(self):
        return self.__class__.__name__ + "(amount={0})".format(self.amount)


def get_train_transforms():
    trans = [
        transforms.Resize((32, 32)),
        transforms.RandomResizedCrop(32, scale=(0.25, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomApply([GaussianNoise(0, 0.05)], p=0.25),
    ]
    return transforms.Compose(trans)


def get_val_transforms():
    trans = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    return transforms.Compose(trans)
