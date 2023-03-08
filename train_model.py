import torch

from src.modeling.dataset.cifar_data import data_split
from src.modeling.train import train, NetTracker
from src.modeling.evaluation import evaluate_model
from src.modeling.evaluation.visualize import get_representations, get_pca, get_tsne, plot_representations

from src.collections.augmentations.transforms import get_train_transforms, get_test_transforms
from src.collections.models import VGG16, VGG19, ResNet18, ResNet34

# init params
USE_GPU = True

TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

RANDOM_STATE = 42

DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# datasplit

datasets, loaders = data_split(
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    TEST_BATCH_SIZE,
    train_transforms=get_train_transforms(),
    test_transforms=get_test_transforms(),
    val_size=0.15,
    download=True,
    random_state=RANDOM_STATE
)

# init model

net = VGG16()
net = net.to(DEVICE)

# train params 

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

tracker = NetTracker(starting_epoch=0)

EPOCH_NUM = 300

# train 

train(
    net,
    tracker,
    EPOCH_NUM,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE
)
