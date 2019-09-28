import pickle
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from lib.process.Training_CNN import Training
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

import lib.snn as snn

import torch
from lib.datasets import SMNIST
from models.CNN import Tailored_CNN

# CONSTANTS:
USE_CUDA = False  # torch.cuda.is_available()
EPOCHs = 6
SMALL = True
DATASETMODE = 'classification'
MODEL_PTH = 'cnn'
TRAIN = True

if TRAIN:
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # Create network instance.
    model = Tailored_CNN().to(device)

    # Learning stats instance.
    stats = snn.learningStats()

    trainset = MNIST('./data', train=TRAIN, download=True, transform=transforms.Compose([
        transforms.ToTensor()]))

    for momentum in np.arange(0, 0.99, 0.1):
        # Define optimizer module.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=momentum)

        training = Training(device, optimizer, trainset, classification=True, collect_weights=True)

        for epoch in range(EPOCHs):
            print('{} / {}'.format(epoch, EPOCHs))
            # Reset training stats.
            stats.training.reset()
            # Training
            training.train(model, stats, epoch=epoch)
            # Update training stats.
            stats.training.update()

# load weights
for momentum in np.arange(0, 0.99, 0.1):
    model = Tailored_CNN()
    with open('weights' + str(model.__class__) + '_' + str(momentum) + '.p', 'rb') as f:
        wc = pickle.load(f)

    comps = wc.get_components(12)
    loss = wc.loss

    mpl.rcParams['legend.fontsize'] = 10

    for (pc1, pc2) in combinations(range(0, 12), 2):
        print(pc1)
        print(pc2)
        z = - np.array(loss)
        x = comps[:, pc1]
        y = comps[:, pc2]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(x, y, z, label='parametric curve')
        ax.set_xlabel('pc_{}'.format(pc1))
        ax.set_ylabel('pc_{}'.format(pc2))
        ax.set_zlabel('negative loss')
        ax.legend()
        # plt.show()
        fig.savefig(
            './saved_figs/pca_' + str(pc1) + '_' + str(pc2) + '_' + str(momentum) + '_' + str(model.__class__) + '.pdf')
        plt.close(fig)
