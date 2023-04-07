import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
CLI example

python main.py \
    --params params.yaml \
    --model_outpath model.pt \
    --experiment_name debug \
    --run_name test_run
'''
from pathlib import Path

import click
import mlflow

from src.torch.utils import save_model_state
from src.torch.dataset import data_split
from src.torch.eval import get_predicts
from src.config import prepare_config
from src.torch.lightning import PLModelWrapper, get_trainer
from src.visualize.plots import plot_confusion_matrix, plot_representations
from src.visualize.utils import get_representations, get_pca, get_tsne
from src.io import load_yaml
from src.collections.models.resnet import ResNet18, ResNet50
from src.collections.models.vgg import VGG19


@click.command()
@click.option('--params', 'params')
@click.option('--model_outpath', 'model_outpath')
@click.option('--experiment_name', 'experiment_name')
@click.option('--run_name', 'run_name')
@click.option('--ckpt_path', 'ckpt_path', required=False, default=None, type=Path)
def train_model(
    params: Path,
    model_outpath: Path,
    experiment_name: str,
    run_name: str,
    ckpt_path: Path = None
) -> None:
    device = load_yaml(params)['DEVICE']
    model_type = load_yaml(params)['MODEL']

    trainer_conf = prepare_config(params, config_key='trainer', resolve=True)
    dataset_params = prepare_config('./params.yaml', config_key='dataset', resolve=True)

    train_dataloader, val_dataloader, test_dataloader = data_split(**dataset_params)

    if model_type == 'ResNet18':
        net = ResNet18().to(device)
    elif model_type == 'ResNet50':
        net = ResNet50().to(device)
    elif model_type == 'VGG19':
        net = VGG19().to(device)
    else: 
        raise ValueError(f"model type {model_type} is not defined correctly")

    optimizer = prepare_config(params, config_key='optimizer', resolve=True)['optimizer']
    optimizer = optimizer(net.parameters())
    criterion = prepare_config(params, config_key='criterion', resolve=True)['criterion']
    scheduler = prepare_config(params, config_key='scheduler', resolve=True)['scheduler']
    scheduler = scheduler(optimizer)
    metrics = prepare_config(params, config_key='metrics', resolve=True)

    pl_net = PLModelWrapper(
        model=net, 
        loss=criterion, 
        optimizer=optimizer, 
        lr_scheduler=scheduler, 
        metrics2log=metrics
    )
    trainer = get_trainer(trainer_conf['trainer_kwargs'], ckpt_path=ckpt_path)

    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name=run_name):
        trainer.fit(pl_net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        trainer.test(pl_net, test_dataloader)

        net = pl_net.retrieve_torch_model().to(device)
        save_model_state(net, model_outpath)

        y_true, y_pred = get_predicts(net, test_dataloader)

        fig = plot_confusion_matrix(y_true, y_pred)
        mlflow.log_figure(fig, 'confusion_matrix.png')

        outputs, labels = get_representations(net, test_dataloader)
        output_pca_data = get_pca(outputs)
        fig = plot_representations(output_pca_data, labels, title='pca')
        mlflow.log_figure(fig, 'pca_representations.png')

        output_tsne_data = get_tsne(outputs)
        fig = plot_representations(output_tsne_data, labels, title='tsne')
        mlflow.log_figure(fig, 'tsne_representations.png')


if __name__ == '__main__':
    train_model()
