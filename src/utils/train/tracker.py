import torch
import numpy as np
import matplotlib.pyplot as plt


class NetTracker:
    def __init__(
        self, 
        starting_epoch: int,
    ) -> None:

        self.train_losses = list()
        self.val_losses = list()

        self.global_weights = list()
        self.global_grads = list()

        self.starting_epoch = starting_epoch

    def update_starting_epoch(self):
        self.starting_epoch = len(self.track_losses) + 1

    def track_losses(self, train_losses, val_losses):
        self.train_losses.append(train_losses)
        self.val_losses.append(val_losses)
    
    def track_weights(self, avg_param_weights, avg_grad_weights):
        self.global_weights.append(avg_param_weights)
        self.global_grads.append(avg_grad_weights)

    def save_weights(self, model, epoch):
        self.state = {
            'model': model.state_dict(),
            'epoch': self.starting_epoch + epoch
        }

    def dump_model_state_dict(self, model_name):
        torch.save(self.state, f'{model_name}.ckpt')

    def retrieve_model(self, model, device):
        model.load_state_dict(self.state['model'])

        start_epoch = self.state['epoch']

        return model, start_epoch

    def plot_training_process(self):
        epochs = [i + self.starting_epoch for i in range(0, len(self.train_losses))]

        nrow, ncol = 1, 3

        size_one_fig = 9

        fig, ax = plt.subplots(nrow, ncol, figsize=(size_one_fig * ncol, size_one_fig * nrow))

        # plot losses
        ax[0].plot(epochs, self.train_losses)
        ax[0].plot(epochs, self.val_losses)
        ax[0].set_title('Losses')
        ax[0].legend(['train_loss', 'val_loss'])
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('loss')
        ax[0].grid()
        ax[0].set_ylim((0, None))

        # plot avg weights

        ax[1].plot(epochs, self.global_weights)
        ax[1].set_title('Avg_weights')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('avg_weight')
        ax[1].grid()
        ax[1].set_ylim((0, None))

        # plot avg weights

        ax[2].plot(epochs, self.global_grads)
        ax[2].set_title('Avg_grads')
        ax[2].set_xlabel('epoch')
        ax[2].set_ylabel('avg_grad')
        ax[2].grid()
        ax[2].set_ylim((0, None))

        plt.show()
