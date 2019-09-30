import pickle
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm

import lib.snn as snn
from lib.datasets import SMNIST
from models.Classifiers import SlayerVgg16, SlayerSNN, SlayerSNN2
from lib.process.Training import Training
from itertools import combinations

# CONSTANTS:
USE_CUDA = torch.cuda.is_available()
EPOCHs = 6
SMALL = False
DATASETMODE = 'classification'
TRAIN = True

netParams = snn.params('network_specs/slayer_snn.yaml')
print(netParams)

if TRAIN:
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # Create network instance.
    model = SlayerSNN2(netParams, input_channels=1).to(device)

    # Learning stats instance.
    stats = snn.learningStats()

    trainingSet = SMNIST(datasetPath=netParams['training']['path']['in'],
                         samplingTime=netParams['simulation']['Ts'],
                         sampleLength=netParams['simulation']['tSample'],
                         mode=DATASETMODE,
                         small=SMALL)
    for momentum in np.arange(0, 0.99, 0.1):
        # Define optimizer module.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=momentum)

        training = Training(netParams, device, optimizer, trainingSet, classification=True, collect_weights=True)

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
    model = SlayerSNN2(netParams, input_channels=1)
    with open('weights' + str(model.__class__) + '_' + str(momentum) + '.p', 'rb') as f:
        wc = pickle.load(f)

    comps = wc.get_components(12)
    loss = wc.loss
    N = len(loss)

    mpl.rcParams['legend.fontsize'] = 10
    print('plotting tuples vs loss\n')
    for (pc1, pc2) in tqdm(combinations(range(0, 12), 2)):
        z = - np.array(loss)
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
        fig.savefig(
            '/home/sarah/Documents/studies/SoSe19/aml_assignments/aml_project/slayer_pca/pca_loss_' + str(
                model.__class__) + '_' + '{0:.2f}'.format(momentum) + '_' + str(pc1) + '_' + str(pc2) + '.pdf')
        plt.close(fig)

    print('printing triplets\n')
    for (pc1, pc2, pc3) in tqdm(combinations(range(0, 12), 3)):
        x = comps[:, pc1]
        y = comps[:, pc2]
        z = comps[:, pc3]
        t = -np.array(loss)

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
        norm = mpl.colors.Normalize(vmin=t.min(), vmax=t.max())

        cbaxes = fig.add_axes([0.025, 0.1, 0.03, 0.8])
        cb1 = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                        norm=norm,
                                        )

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min()-0.01, y.max()+0.01)

        # plt.show()
        fig.savefig(
            '/home/sarah/Documents/studies/SoSe19/aml_assignments/aml_project/slayer_pca/pca_triplets_' + str(
                model.__class__) + '_' + '{0:.2f}'.format(momentum) + '_' + str(pc1) + '_' + str(pc2) + '_' + str(
                pc3) + '.pdf')
        plt.close(fig)
