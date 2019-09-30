import pickle
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from lib.process.Training_CNN import Training
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from itertools import combinations
from scipy.signal import savgol_filter

import lib.snn as snn

import torch
from lib.datasets import SMNIST
from models.CNN import Raw_CNN

# CONSTANTS:
USE_CUDA = False  # torch.cuda.is_available()
EPOCHs = 6
SMALL = False
DATASETMODE = 'classification'
MODEL_PTH = 'cnn'
TRAIN = False

if TRAIN:
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # Create network instance.
    model = Raw_CNN().to(device)

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

## load weights
for momentum in [ 0.2, 0.9]: #np.arange(0, 0.99, 0.1):
    model = Raw_CNN()
    with open('/Volumes/GG/cnn/weights' + str(model.__class__) + '_' + str(momentum) + '.p', 'rb') as f:
        wc = pickle.load(f)

    comps = wc.get_components(12)
    loss = wc.loss
    N = len(loss)
    L = - np.array(loss)
    L = savgol_filter(L, 51, 3)  # window size 51, polynomial order 3

    mpl.rcParams['legend.fontsize'] = 10
    print('plotting tuples vs loss\n')
    for (pc1, pc2) in tqdm(combinations(range(0, 12), 2)):
        z = L
        x = comps[:, pc1]
        y = comps[:, pc2]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('pc_{}'.format(pc1))
        ax.set_ylabel('pc_{}'.format(pc2))
        ax.set_zlabel('negative loss')


        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = (z-z.min())/(z.max()-z.min())
        cmap = plt.get_cmap('plasma')
        colors = [cmap(norm[ii]) for ii in range(N-1)]

        for ii in range(N-1):
            segii = segments[ii]
            lii, = ax.plot(segii[:, 0], segii[:, 1], segii[:, 2], color=colors[ii], linewidth=2)
            # lii.set_dash_joinstyle('round')
            # lii.set_solid_joinstyle('round')
            lii.set_solid_capstyle('round')

        cmap = mpl.cm.plasma
        norm = mpl.colors.Normalize(vmin=z.min(), vmax=z.max())

        cbaxes = fig.add_axes([0.025, 0.1, 0.03, 0.8])
        cb1 = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                        norm=norm,
                                        )

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min()-0.01, y.max()+0.01)

        # plt.show()
        print('/Volumes/GG/cnn/loss/pca_' + str(pc1) + '_' + str(pc2) + '_' + str(momentum) + '_' + str(model.__class__.__name__) + '.pdf')
        fig.savefig(
            '/Volumes/GG/cnn/loss/pca_' + str(pc1) + '_' + str(pc2) + '_' + str(momentum) + '_' + str(model.__class__.__name__) + '.pdf')
        plt.close(fig)

    print('printing triplets\n')
    for (pc1, pc2, pc3) in tqdm(combinations(range(0, 12), 3)):
        x = comps[:, pc1]
        y = comps[:, pc2]
        z = comps[:, pc3]
        t = L

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('pc_{}'.format(pc1))
        ax.set_ylabel('pc_{}'.format(pc2))
        ax.set_zlabel('pc_{}'.format(pc3))


        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = (t-t.min())/(t.max()-t.min())
        cmap = plt.get_cmap('plasma')
        colors = [cmap(norm[ii]) for ii in range(N-1)]

        for ii in range(N-1):
            segii = segments[ii]
            lii, = ax.plot(segii[:, 0], segii[:, 1], segii[:, 2], color=colors[ii], linewidth=2)
            # lii.set_dash_joinstyle('round')
            # lii.set_solid_joinstyle('round')
            lii.set_solid_capstyle('round')

        cmap = mpl.cm.plasma
        norm = mpl.colors.Normalize(vmin=-4, vmax=0)

        cbaxes = fig.add_axes([0.025, 0.1, 0.03, 0.8])
        cb1 = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                        norm=norm,
                                        )

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min()-0.01, y.max()+0.01)

        # plt.show()
        print('/Volumes/GG/cnn/triplets/pca_' + str(pc1) + '_' + str(pc2) + '_' + str(pc3) + '_' + str(momentum) + '_' + str(model.__class__.__name__) + '.pdf')
        fig.savefig(
            '/Volumes/GG/cnn/triplets/pca_' + str(pc1) + '_' + str(pc2) + '_' + str(pc3) + '_' + str(momentum) + '_' + str(model.__class__.__name__) + '.pdf')
        plt.close(fig)
