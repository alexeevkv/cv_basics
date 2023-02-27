import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from train.tracker import NetTracker


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


def evaluate_model(model, tracker: NetTracker, test_loader, model_name, device):
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

    tracker.plot_training_process()

    plot_confusion_matrix(y_true, y_pred)

    # return y_true, y_pred
