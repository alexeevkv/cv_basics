import torch
from torch import nn

from torchvision import transforms

from skimage.util import random_noise

from src.models.vgg import VGG
from src.models.resnet import ResNet18
from src.utils.train.train import train
from src.utils.train.tracker import NetTracker
from src.utils.dataset.cifar_data import data_split
from src.utils.evaluation.evaluate import evaluate_model
from src.utils.evaluation.visualize import get_representations, get_pca, get_tsne, plot_representations

# init params
USE_GPU = True

TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

RANDOM_STATE = 42

DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

VGG_TYPE = 'VGG11'

EPOCH_NUM = 15


# transforms 
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


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


train_transforms = transforms.Compose(
    [
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
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
)

# data 

datasets, loaders = data_split(
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    TEST_BATCH_SIZE,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_size=0.15,
    download=True,
    random_state=RANDOM_STATE
)
# init model

model = VGG(VGG_TYPE).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train 

tracker = NetTracker(starting_epoch=0)

train(
    model,
    tracker,
    EPOCH_NUM,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE
)

# evaluate model

evaluate_model(
    model,
    tracker,
    model_name=VGG_TYPE,
    device=DEVICE,
)

# visualize
outputs, labels = get_representations(model, loaders['test'], DEVICE)

output_pca_data = get_pca(outputs)
plot_representations(output_pca_data, labels, title=f'pca for {VGG_TYPE}')

output_tsne_data = get_tsne(outputs)
plot_representations(output_tsne_data, labels, title=f'tsne for {VGG_TYPE}')

# resume training

tracker.update_starting_epoch()

MORE_EPOCHS = 1 

train(
    model,
    tracker,
    MORE_EPOCHS,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE
)

evaluate_model(
    model,
    tracker,
    model_name=VGG_TYPE,
    device=DEVICE,
)

# visualize
outputs, labels = get_representations(model, loaders['test'], DEVICE)

output_pca_data = get_pca(outputs)
plot_representations(output_pca_data, labels, title=f'pca for {VGG_TYPE}')

output_tsne_data = get_tsne(outputs)
plot_representations(output_tsne_data, labels, title=f'tsne for {VGG_TYPE}')
