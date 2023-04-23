import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10

from sklearn.model_selection import train_test_split


class CifarDataset(Dataset):
    def __init__(self, images, targets, transform=None, augm=None, idx2class_mapping=None):
        self.images = images
        self.targets = targets
        self.augm = augm
        self.transform = transform
        self.idx2class_mapping = idx2class_mapping

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]
            
        if self.transform is not None:
            image = self.transform(image)
        if self.augm is not None:
            image = np.asarray(self.images[index])
            image = self.augm(image=image)['image']
            image = image.transpose(2, 0, 1).astype('float32')

        return image, target
    
    def __len__(self):
        return len(self.images)


def load_cifar_data(root='./data/CIFAR10', download=False):
    train_data = CIFAR10(root=root, train=True, download=download)
    test_data = CIFAR10(root=root, train=False, download=download)

    classes2idx = train_data.class_to_idx
    idx2classes = {v: k for (k, v) in classes2idx.items()}

    return train_data, test_data, idx2classes


def get_dataloaders(
    train_batch_size, 
    val_batch_size, 
    test_batch_size, 
    train_transforms,
    test_transforms,
    train_augm=None,
    test_augm=None,
    val_size=0.2, 
    download=True,
    root='./data/CIFAR10',
    random_state=42,
    num_workers=4
):
    train_data, test_data, idx2classes = load_cifar_data(root, download=download)
    
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
    train_dataset = CifarDataset(train_images, train_targets, train_transforms, train_augm, idx2classes)
    val_dataset = CifarDataset(val_images, val_targets, test_transforms, test_augm, idx2classes)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    # test 
    test_dataset = CifarDataset(test_images, test_targets, test_transforms, test_augm, idx2classes)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
