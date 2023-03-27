import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.torch.utils import to_np, to_cuda


# TODO Добавить корректное вычисление метрик для разных задач: для классификации и сегментации 


def get_weights_and_grads(named_parameters):
    avg_weights = list()
    avg_grads = list()

    for name, parameter in named_parameters:
        mean_weight = parameter.detach().cpu().abs().mean()
        mean_grad = parameter.grad.detach().abs().mean()

        avg_weights.append(mean_weight)
        avg_grads.append(mean_grad)

    avg_weight = sum(avg_weights) / len(avg_weights)
    avg_grad = sum(avg_grads) / len(avg_grads)

    avg_weight = float(avg_weight.cpu().numpy())
    avg_grad = float(avg_grad.cpu().numpy())

    return avg_weight, avg_grad


def transform_inputs(y_true, y_pred, task_type='classification'):
    if task_type == 'classification':
        y_pred_np = to_np(torch.argmax(y_pred, dim=1))
        y_true_np = to_np(y_true)   

        return y_true_np, y_pred_np

    if task_type == 'segmentation':
        return y_true, y_pred


def compute_metrics(y_true, y_pred, metrics):
    y_true_transformed, y_pred_transformed = transform_inputs(y_true, y_pred, task_type='classification')

    computed_metrics = dict()

    for metric_name, metric_func in metrics.items():
        computed_metrics[metric_name] = metric_func(y_true_transformed, y_pred_transformed)

    return computed_metrics


class PLModelWrapper(pl.LightningModule):
    # TODO Добавить логгирование на один и тот же график (train versus val)
    def __init__(self, model, loss, optimizer, lr_scheduler, metrics2log=None, samples2track=None):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics2log = metrics2log if metrics2log is not None else {}
        self.samples2track = samples2track

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.model.forward(x)
        loss = self.loss(logits, y)
        batch_dict = {'loss': loss}

        metrics = compute_metrics(y, logits, self.metrics2log)
        batch_dict.update(metrics)

        return batch_dict
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log("loss/train", avg_loss, on_epoch=True)

        for metric_name in self.metrics2log.keys():
            avg_metric_value = np.stack([x[metric_name] for x in outputs]).mean()
            self.log(f"{metric_name}/train", avg_metric_value, on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.model.forward(x)
        loss = self.loss(logits, y)
        batch_dict = {'loss': loss}

        metrics = compute_metrics(y, logits, self.metrics2log)
        
        batch_dict.update(metrics)

        return batch_dict 

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log("loss/val", avg_loss, on_epoch=True)

        for metric_name in self.metrics2log.keys():
            avg_metric_value = np.stack([x[metric_name] for x in outputs]).mean()
            self.log(f"{metric_name}/val", avg_metric_value, on_epoch=True)
     
    def on_after_backward(self):
        if self.trainer.is_last_batch:
            grad_vec = None
            weight_vec = None
            for p in self.parameters():
                if grad_vec is None:
                    grad_vec = p.grad.data.view(-1)
                else:
                    grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))

                if weight_vec is None:
                    weight_vec = p.data.view(-1)
                else:
                    weight_vec = torch.cat((weight_vec, p.data.view(-1)))
                    
            self.log('abs_avg_weight', float(to_np(torch.mean(weight_vec.abs()))))
            self.log('abs_avg_grad', float(to_np(torch.mean(grad_vec.abs()))))

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        # self.log("test_loss", loss)       

    def configure_optimizers(self):
        optimizer = self.optimizer
        lr_scheduler = {'scheduler': self.lr_scheduler, 'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]
    
    def retrieve_torch_model(self):
        return self.model


# class PrintCallback(Callback):
#     def on_train_start(self, trainer, pl_module):
#         print("Training is started!")

#     def on_train_end(self, trainer, pl_module):
#         print("Training is done.")

#     def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         return super().on_train_epoch_start(trainer, pl_module)


def get_trainer(trainer_kwargs, resume_from_checkpoint=False):
    if resume_from_checkpoint:
        trainer = pl.Trainer(resume_from_checkpoint='./model.ckpt')
    else:
        trainer = pl.Trainer(**trainer_kwargs)

    return trainer
