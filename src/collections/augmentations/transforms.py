from torchvision import transforms
from .custom_transforms import GaussianNoise

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


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


def get_test_transforms():
    trans = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    return transforms.Compose(trans)
