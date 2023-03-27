'''
python src/pipeline/train.py \
    --params params.yaml \
    --model_outpath renset_18.pt \
    --run_name resnet18 \
    --resume_from_checkpoint False 
'''
from pathlib import Path

import click
import mlflow
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import __repo__
from src.torch.utils import save_model_state, load_net
from src.torch.dataset import data_split
from src.config import prepare_config
from src.torch.lightning import PLModelWrapper, get_trainer


@click.command()
@click.option('--params', 'params')
@click.option('--model_outpath', 'model_outpath')
@click.option('--run_name', 'run_name')
@click.option('--resume_from_checkpoint', 'resume_from_checkpoint', required=False, default=False, type=bool)
def train_model(
    params: Path,
    model_outpath: Path,
    run_name: str,
    resume_from_checkpoint: bool = False
) -> None:
    net_conf = prepare_config(params, config_key='model', resolve=True)
    trainer_conf = prepare_config(params, config_key='trainer', resolve=True)
    
    dataset_params = prepare_config('./params.yaml', config_key='dataset', resolve=True)['cifar10']

    train_dataloader, val_dataloader, test_dataloader = data_split(**dataset_params)

    net = load_net(net_conf)

    optimizer = prepare_config(params, config_key='optimizer', resolve=True)['optimizer'](net.parameters())
    criterion = prepare_config(params, config_key='criterion', resolve=True)['criterion']
    scheduler = prepare_config(params, config_key='scheduler', resolve=True)['scheduler'](optimizer)
    metrics = prepare_config(params, config_key='metrics', resolve=True)

    pl_net = PLModelWrapper(
        model=net, 
        loss=criterion, 
        optimizer=optimizer, 
        lr_scheduler=scheduler, 
        metrics2log=metrics
    )
    trainer = get_trainer(trainer_conf['trainer_kwargs'], resume_from_checkpoint=resume_from_checkpoint)

    #mlflow.set_tracking_uri('http://localhost:5782')  # set up connection
    mlflow.set_experiment('diploma')          # set the experiment
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        trainer.fit(pl_net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    net = pl_net.retrieve_torch_model()
    save_model_state(net, model_outpath)


if __name__ == '__main__':
    train_model()
