import torch
import matplotlib

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col


def visualize(source_feature: torch.Tensor, labels: torch.Tensor,
              filename: str, source_color='r'):
    """
    Visualize features from different domains using t-SNE.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'

    """
    label = labels.numpy()

    source_feature = source_feature.numpy()

    features = np.concatenate([source_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
    print(label.shape)
    print(X_tsne.shape)
    domains = (np.ones(len(source_feature)))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([source_color]), s=2)
    # visualize using matplotlib
    plt.figure(figsize=(10, 10))
    plt.savefig(filename)