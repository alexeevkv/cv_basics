import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.torch.utils import to_np, to_cuda
from src.torch.eval import compute_metrics


class PLModelWrapper(pl.LightningModule):
    def __init__(self, model, loss, optimizer, lr_scheduler, metrics2log=None):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics2log = metrics2log if metrics2log is not None else {}

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
            self.log('max_grad', float(torch.max(grad_vec)))
            self.log('min_grad', float(torch.min(grad_vec)))

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        batch_dict = {'loss': loss}

        metrics = compute_metrics(y, logits, self.metrics2log)
        
        batch_dict.update(metrics)

        return batch_dict   

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log("loss/test", avg_loss, on_epoch=True)

        for metric_name in self.metrics2log.keys():
            avg_metric_value = np.stack([x[metric_name] for x in outputs]).mean()
            self.log(f"{metric_name}/test", avg_metric_value, on_epoch=True)

    def configure_optimizers(self):
        optimizer = self.optimizer
        lr_scheduler = {'scheduler': self.lr_scheduler, 'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]
    
    def retrieve_torch_model(self):
        return self.model


def get_trainer(trainer_kwargs, resume_from_checkpoint=False):
    if resume_from_checkpoint:
        trainer = pl.Trainer(**trainer_kwargs, resume_from_checkpoint='lightning_logs/version_66/checkpoints/epoch=90-step=30303.ckpt')
    else:
        trainer = pl.Trainer(**trainer_kwargs)

    return trainer
