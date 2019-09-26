from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import lib.snn as snn
from lib.datasets.mnistdataset import SMNIST
from lib.utils import spikeTensorToProb, save_model, load_model

from lib.process.Training import Training
from lib.process.Evaluation import Evaluation
from models.Autoencoder import VAE

# CONSTANTS:
USE_CUDA = torch.cuda.is_available()
EPOCHs = 200
SMALL = True
DATASETMODE = 'autoencoderSpike'
MODEL_PTH = 'vae'
LR = 0.001

netParams = snn.params('network_specs/vae.yaml')
print(netParams)

device = torch.device("cuda" if USE_CUDA else "cpu")

# Create network instance.
model = VAE(netParams, hidden_size=100, latent_size=50).to(device)

# Load model
load_model(MODEL_PTH, model)
# Define optimizer module.
optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)

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
min_loss = 10000000
for epoch in range(EPOCHs):
    print('{} / {}'.format(epoch, EPOCHs))
    stats.training.reset()
    training.train(model, stats, epoch=epoch)
    stats.training.update()
    stats.testing.reset()
    testing.test(model, stats, epoch=epoch)
    stats.testing.update()
    print(stats.testing.minloss)
    if min_loss > stats.testing.loss():
        print('Saving model. Model improved : currect acc ', stats.testing.loss(), ' min loss, ', min_loss)
        print('\n\n\n\n')
        save_model(MODEL_PTH, model)
        min_loss = stats.testing.loss()

    else:
        print('Model didn\'t improve : currect acc ', stats.testing.loss(), ' min loss, ', min_loss)
        print('\n\n\n\n')

# Plot the results.
plt.figure(1)
plt.semilogy(stats.training.lossLog, label='Training')
# plt.semilogy(stats.testing.lossLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
model = model.eval()
x, y, c = trainingSet[0]
x = x.to(device)
x = x.unsqueeze(0)
output = model(x)
pred = spikeTensorToProb(output[0].squeeze())
# target_grid = torchvision.utils.make_grid(target)
# pred_grid = torchvision.utils.make_grid(pred)
plt.subplot(2, 1, 1)
plt.imshow(pred.detach().cpu().numpy().reshape(28, 28))
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
