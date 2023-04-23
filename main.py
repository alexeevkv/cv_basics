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

from src.torch.utils import save_model_state, select_device
from src.torch.dataset import data_split
from src.torch.eval import get_predicts
from src.config import prepare_config
from src.torch.lightning import PLModelWrapper, get_trainer
from src.visualize.plots import plot_confusion_matrix, plot_representations
from src.visualize.utils import get_representations, get_pca, get_tsne
from src.io import load_yaml


@click.command()
@click.option('--params', 'params')
@click.option('--model_outpath', 'model_outpath')
@click.option('--experiment_name', 'experiment_name')
@click.option('--run_name', 'run_name')
def train_model(
    params: Path,
    model_outpath: Path,
    experiment_name: str,
    run_name: str,
    ckpt_path: Path = None
) -> None:
    raw_conf = load_yaml(params)
    net_conf = prepare_config(params, config_key='model', resolve=True)
    trainer_conf = prepare_config(params, config_key='trainer', resolve=True)
    dataset_conf = prepare_config(params, config_key='dataset', resolve=True)
    optimizer_conf = prepare_config(params, config_key='optimizer', resolve=True)
    criterion_conf = prepare_config(params, config_key='criterion', resolve=True)
    scheduler_conf = prepare_config(params, config_key='scheduler', resolve=True)
    metrics_conf = prepare_config(params, config_key='metrics', resolve=True)

    device = select_device(raw_conf)  # sets specific gpu if more than 1 is available
    ckpt_path = raw_conf['CKPT_PATH']
    net = net_conf['net'].to(device)
    trainer = get_trainer(trainer_conf['trainer_kwargs'], ckpt_path=ckpt_path)
    train_dataloader, val_dataloader, test_dataloader = data_split(**dataset_conf)
    optimizer = optimizer_conf['optimizer'](net.parameters())
    criterion = criterion_conf['criterion']
    scheduler = scheduler_conf['scheduler'](optimizer)

    pl_net = PLModelWrapper(
        model=net, 
        loss=criterion, 
        optimizer=optimizer, 
        lr_scheduler=scheduler, 
        metrics2log=metrics_conf
    )

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
