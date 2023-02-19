import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
    

def tensors2numpy_array(tensors):
    temp = np.array([])
    for i in range(len(tensors)):
        if torch.cuda.is_available():
            dvb = tensors[i].cpu().numpy()
        else: 
            dvb = tensors[i].numpy()
        temp = np.concatenate((temp, dvb), axis=0)
    return temp


def plot_confusion_matrix(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 10))
    sns.heatmap(CM, annot=True)
    plt.show()


def plot_losses(train_losses, val_losses, ylim=10, model_name='VGG'):
    epochs = [i for i in range(0, len(train_losses))]
    nrow, ncol = 1, 1

    size_one_fig = 9

    fig, ax = plt.subplots(nrow, ncol, figsize=(size_one_fig * ncol, size_one_fig * nrow))

    ax.plot(epochs, train_losses)
    ax.plot(epochs, val_losses)
    ax.set_title('VGG')
    ax.legend(['train_loss', 'val_loss'])
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid()
    ax.set_ylim((0, ylim))

    plt.show()


def evaluate_model(model, test_loader, train_losses, val_losses, model_name, device):
    predictions = list()

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data['image'], data['target']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, pred = torch.max(outputs, dim=1)
            predictions.append(pred)

    y_true = test_loader.dataset.targets
    y_pred = tensors2numpy_array(predictions)

    print(f' Accuracy on test = {accuracy_score(y_true, y_pred)}')

    plot_losses(train_losses, val_losses, model_name=model_name)

    plot_confusion_matrix(y_true, y_pred)

    return y_true, y_pred
