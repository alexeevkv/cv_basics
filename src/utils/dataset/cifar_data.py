import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from sklearn.model_selection import train_test_split

# Так как при использовании colorjitter, значение статистик действительно немного меняется, 
# То используем просто 0.5 для MEAN и STD

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

NUM_WORKERS = 4 


class CifarData(Dataset):
    def __init__(self, images, targets, aug=None, transform=None):
        self.images = images
        self.targets = targets
        self.aug = aug
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]
        
        if self.aug is not None:
            augmented = self.aug(image=image)
            image = augmented['image']
            
        if self.transform is not None:
            image = self.transform(image)
            
        return {
            'image': image,
            'target': torch.tensor(target, dtype=torch.long)
        }
        
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
    val_size=0.2, 
    download=True, 
    random_state=42
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
    train_dataset = CifarData(train_images, train_targets, transform=train_transforms)
    val_dataset = CifarData(val_images, val_targets, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # test 
    test_dataset = CifarData(test_images, test_targets, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=NUM_WORKERS)

    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    } 

    data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    return datasets, data_loaders
