import torch 
from torchvision import transforms
from skimage.util import random_noise


class GaussianNoise(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class PepperNoise(object):
    def __init__(self, amount=0.1):
        self.amount = amount
    
    def __call__(self, tensor):
        return torch.tensor(random_noise(tensor, mode='pepper', amount=self.amount))

    def __repr__(self):
        return self.__class__.__name__ + '(amount={0})'.format(self.amount)
