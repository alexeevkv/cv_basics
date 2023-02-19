import torch
from torch import nn
from src.models.vgg import VGG
from src.utils.train import train
from src.utils.cifar_data import data_split
from src.utils.evaluate import evaluate_model


USE_GPU = True

TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

RANDOM_STATE = 42

DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

VGG_TYPE = 'VGG11'

EPOCH_NUM = 5

# data 

datasets, loaders = data_split(
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    TEST_BATCH_SIZE,
    val_size=0.2,
    download=True,
    random_state=RANDOM_STATE
)
# train 

model = VGG(VGG_TYPE).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model, train_losses, val_losses = train(
    model,
    epoch_num=EPOCH_NUM,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    criterion=criterion, 
    optimizer=optimizer, 
    evaluate_training=True
)

evaluate_model(
    model,
    train_losses=train_losses,
    val_losses=val_losses,
    model_name=VGG_TYPE,
    device=DEVICE,
)
