import pickle
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import torch
import lib.snn as snn
from lib.datasets import SMNIST
from models.Classifiers import SlayerVgg16, SlayerSNN, SlayerSNN2
from lib.process.Training import Training

# CONSTANTS:
USE_CUDA = torch.cuda.is_available()
EPOCHs = 6
SMALL = True
DATASETMODE = 'classification'
MODEL_PTH = 'slayer_snn'
TRAIN = True

if TRAIN:
    netParams = snn.params('network_specs/slayer_snn.yaml')
    print(netParams)

    device = torch.device("cuda" if USE_CUDA else "cpu")

    # Create network instance.
    model = SlayerSNN2(netParams).to(device)

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

# loag weights
with open('weights.p', 'rb') as f:
    wc = pickle.load(f)

comps = wc.get_components(12)
loss = wc.loss

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = loss
x = comps[:, 4]
y = comps[:, 9]
ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()
