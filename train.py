import click

import pandas as pd

from pathlib import Path
from torch import nn

from omegaconf import OmegaConf

from hydra.utils import instantiate

from src.config import prepare_config


class DataLoader:
    def __init__(*args, **kwargs):
        pass


def train_epoch(
        model, 
        epoch_num,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        writer=None,
        device='cpu'
) -> nn.Module:
    print('kek')
    pass


# @click.command()
def train_model(
    net_config: Path,
    dataset_config_path: Path,
    model_outpath: Path
) -> Path:
    
    # Model
    net = instantiate(
        prepare_config(net_config, config_key='net', resolve=True)
    )

    optimizer = instantiate(
        prepare_config(net_config, config_key='optimizer', resolve=True)(net.parameters())
    )

    criterion = instantiate(
        prepare_config(net_config, config_key='criterion', resolve=True)
    )

    epochs = prepare_config(net_config, config_key='epochs')

    # DataLoader

    dataset_conf = OmegaConf.to_object(OmegaConf.load(dataset_config_path))

    train_loader = DataLoader(
        dataset=pd.read_csv(dataset_conf['path2train']),
        augmentation=prepare_config(dataset_conf['path2train_augm_conf'], config_key='Compose', resolve=True)
    )

    val_loader = DataLoader(
        dataset=pd.read_csv(dataset_conf['path2val']),
        augmentation=prepare_config(dataset_conf['path2test_augm_conf'], config_key='Compose', resolve=True)
    )

    # train 

    for epoch in epochs:
        model = train_epoch(
            net,
            epoch,
            train_loader,
            val_loader,
            criterion,
            optimizer
        )


NET_CONFIG = './conf/models/net.yaml'
DATASET_CONFIG = './conf/dataset/dataset.yaml'
MODEL_OUTPATH = './'

train_model(NET_CONFIG, DATASET_CONFIG, MODEL_OUTPATH)
