import torch

import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import manifold


def get_representations(model, loader, device):

    model.eval()

    outputs = []
    labels = []

    with torch.no_grad():
        
        for data in loader:
            inputs, targets = data['image'], data['target']
            inputs, targets = inputs.to(device), targets.to(device)

            y_pred = model(inputs)

            if device == 'cuda':
                outputs.append(y_pred.cpu())
                labels.append(targets.cpu())
            else:
                outputs.append(y_pred)
                labels.append(targets)
        
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs, labels


def get_pca(data, n_components=2):
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data


def get_tsne(data, n_components=2, n_images=None):
    
    if n_images is not None:
        data = data[:n_images]
        
    tsne = manifold.TSNE(n_components=n_components, random_state=0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data


def plot_representations(data, labels, title='', n_images=None):
            
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]
                
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='hsv')
    ax.set_title(title)
    ax.grid()
    plt.show()
