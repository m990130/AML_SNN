import torch
from sklearn.decomposition import PCA
import numpy as np


class WeightCollector(object):
    def __init__(self):
        self.weights = []
        self.loss = []

    def capture(self, model, loss):
        weights = list(map(lambda t: t.view(-1), model.state_dict().values()))
        self.weights.append(torch.cat(weights).detach().cpu().numpy())
        self.loss.append(loss)

    def get_components(self, size=12):
        X = np.array(self.weights)
        pca = PCA(n_components=size)
        principalComponents = pca.fit_transform(X)
        return principalComponents
