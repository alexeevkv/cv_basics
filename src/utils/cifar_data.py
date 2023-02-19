import torch
from torch.utils.data import Dataset

from torchvision import transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


class CifarData(Dataset):
    def __init__(self, images, targets, aug=None, transform=True):
        self.images = images
        self.targets = targets
        self.aug = aug
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]

        if self.aug is not None:
            augmented = self.aug(image=image)
            image = augmented["image"]

        if self.transform:
            transform = transforms.Compose(
                [
                    transforms.ColorJitter(),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD),
                ]
            )
            image = transform(image)

        return {"image": image, "target": torch.tensor(target, dtype=torch.long)}

    def __len__(self):
        return len(self.images)
