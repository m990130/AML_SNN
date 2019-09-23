from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import lib.snn as snn
from lib.datasets.mnistdataset import SMNIST
from lib.utils import spikeTensorToProb, save_model

from lib.process.Training import Training
from lib.process.Evaluation import Evaluation
from Autoencoder import Autoencoder

# CONSTANTS:
USE_CUDA = torch.cuda.is_available()
EPOCHs = 2
SMALL = True
DATASETMODE = 'autoencoderSpike'

netParams = snn.params('network_specs/autoencoder.yaml')
print(netParams)

device = torch.device("cuda" if USE_CUDA else "cpu")

# Create network instance.
net = Autoencoder(netParams).to(device)

# Define optimizer module.
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, amsgrad=True)

# Learning stats instance.
stats = snn.learningStats()

trainingSet = SMNIST(datasetPath=netParams['training']['path']['in'],
                     samplingTime=netParams['simulation']['Ts'],
                     sampleLength=netParams['simulation']['tSample'],

                     mode=DATASETMODE,
                     small=SMALL)

testingSet = SMNIST(datasetPath=netParams['training']['path']['in'],
                    samplingTime=netParams['simulation']['Ts'],
                    sampleLength=netParams['simulation']['tSample'],
                    mode=DATASETMODE,
                    small=SMALL)

training = Training(netParams, device, optimizer, trainingSet)
testing = Evaluation(netParams, device, optimizer, testingSet)

# training loop
for epoch in range(EPOCHs):
    print('{} / {}'.format(epoch, EPOCHs))
    tSt = datetime.now()

    training.train(net, stats)

    testing.test(net, stats)

    # Update stats.
    stats.update()
    if epoch % 100 == 0 and epoch != 0:
        save_model('autoencoder', net)

save_model('autoencoder', net)

# Plot the results.
plt.figure(1)
plt.semilogy(stats.training.lossLog, label='Training')
# plt.semilogy(stats.testing.lossLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
net = net.eval()
x, y, c = trainingSet[0]
x = x.to(device)
x = x.unsqueeze(0)
output = net(x)
pred = spikeTensorToProb(output.squeeze())
# target_grid = torchvision.utils.make_grid(target)
# pred_grid = torchvision.utils.make_grid(pred)
plt.subplot(2, 1, 1)
plt.imshow(pred.detach().cpu().numpy())
plt.subplot(2, 1, 2)

if DATASETMODE == 'autoencoder':
    plt.imshow(y.detach().cpu().numpy().squeeze())
elif DATASETMODE == 'autoencoderSpike':
    y = spikeTensorToProb(y)
    plt.imshow(y.detach().cpu().numpy().squeeze())
# plt.show()show
# plt.plot(stats.training.accuracyLog, label='Training')
# plt.plot(stats.testing.accuracyLog, label='Testing')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

plt.show()
