from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import lib.snn as snn
from lib.utils import save_model, load_model

from lib.process.Training import Training
from lib.process.Evaluation import Evaluation
from lib.datasets import SCIFAR
from models.Classifiers import SlayerVgg16

# CONSTANTS:
USE_CUDA = torch.cuda.is_available()
EPOCHs = 100
SMALL = True
DATASETMODE = 'classification'
MODEL_PTH = 'slayer_cifar'

netParams = snn.params('network_specs/slayer_cifar.yaml')
print(netParams)

device = torch.device("cuda" if USE_CUDA else "cpu")

# Create network instance.
model = SlayerVgg16(netParams).to(device)

# Load model
load_model(MODEL_PTH, model)

# Define optimizer module.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

# Learning stats instance.
stats = snn.learningStats()

trainingSet = SCIFAR(datasetPath=netParams['training']['path']['in'],
                     samplingTime=netParams['simulation']['Ts'],
                     sampleLength=netParams['simulation']['tSample'],
                     mode=DATASETMODE,
                     small=SMALL)

testingSet = SCIFAR(datasetPath=netParams['training']['path']['in'],
                    samplingTime=netParams['simulation']['Ts'],
                    sampleLength=netParams['simulation']['tSample'],
                    mode=DATASETMODE,
                    small=SMALL)

training = Training(netParams, device, optimizer, trainingSet, classification=True)
testing = Evaluation(netParams, device, optimizer, testingSet, classification=True)

# training loop
for epoch in range(EPOCHs):
    print('{} / {}'.format(epoch, EPOCHs))

    training.train(model, stats, epoch=epoch)

    testing.test(model, stats)

    # Update stats.
    stats.update()
    if epoch % 100 == 0 and epoch != 0:
        save_model(MODEL_PTH, model)

save_model(MODEL_PTH, model)

# Plot the results.
plt.figure(1)
plt.semilogy(stats.training.lossLog, label='Training')
plt.semilogy(stats.testing.lossLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.figure(2)
plt.plot(stats.training.accuracyLog, label='Training')
plt.plot(stats.testing.accuracyLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
