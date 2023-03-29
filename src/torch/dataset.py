import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from sklearn.model_selection import train_test_split


class CifarDataset(Dataset):
    def __init__(self, images, targets, augm=None, transform=None):
        self.images = images
        self.targets = targets
        self.augm = augm
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]
            
        if self.transform is not None:
            image = self.transform(image)
        if self.augm is not None:
            image = self.augm(image=image)['image']
            
        #image = image.transpose(2, 0, 1).astype('float32')

        return image, target
    
    def __len__(self):
        return len(self.images)


def load_cifar_data(download=False):
    train_data = CIFAR10(root="./data/CIFAR10", train=True, download=download)
    test_data = CIFAR10(root="./data/CIFAR10", train=False, download=download)

    return train_data, test_data


def data_split(
    train_batch_size, 
    val_batch_size, 
    test_batch_size, 
    train_transforms,
    test_transforms,
    train_augm=None,
    test_augm=None,
    val_size=0.2, 
    download=True, 
    random_state=42,
    num_workers=4
):
    train_data, test_data = load_cifar_data(download=download)
    
    train_images = [data[0] for data in train_data]
    train_targets = [data[1] for data in train_data]

    test_images = [data[0] for data in test_data]
    test_targets = [data[1] for data in test_data]
    
    train_images, val_images, train_targets, val_targets = \
        train_test_split(
            train_images,
            train_targets,
            test_size=val_size,
            random_state=random_state
        )

    # train, val
    train_dataset = CifarDataset(train_images, train_targets, transform=train_transforms, augm=train_augm)
    val_dataset = CifarDataset(val_images, val_targets, transform=test_transforms, augm=test_augm)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    # test 
    test_dataset = CifarDataset(test_images, test_targets, transform=test_transforms, augm=test_augm)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    # datasets = {
    #     'train': train_dataset,
    #     'val': val_dataset,
    #     'test': test_dataset
    # } 

    return train_loader, val_loader, test_loader
