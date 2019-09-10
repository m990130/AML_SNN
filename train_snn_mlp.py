import lib.snn as snn
import  lib.spikeFileIO as io


from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import lib.snn as snn
import lib.spikeFileIO as io
import zipfile
import os

# CONSTANTS

USE_CUDA = torch.cuda.is_available()
#

netParams = snn.params('data/mnistMLP/network.yaml')
print(netParams)


# Dataset definition
class nmnistDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        inputIndex = self.samples[index, 0]
        classLabel = self.samples[index, 1]

        inputSpikes = io.read2Dspikes(
            self.path + str(inputIndex.item()) + '.bs2'
        ).toSpikeTensor(torch.zeros((2, 34, 34, self.nTimeBins)),
                        samplingTime=self.samplingTime)
        desiredClass = torch.zeros((10, 1, 1, 1))
        desiredClass[classLabel, ...] = 1
        # Input spikes are reshaped to ignore the spatial dimension and the neurons are placed in channel dimension.
        # The spatial dimension can be maintained and used as it is.
        # It requires different definition of the dense layer.
        return inputSpikes.reshape((-1, 1, 1, inputSpikes.shape[-1])), desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]


# Network definition
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # Initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # Define network functions
        # The commented line below should be used if the input spikes were not reshaped
        # self.fc1   = slayer.dense((34, 34, 2), 512)
        self.fc1 = slayer.dense((34 * 34 * 2), 512)
        self.fc2 = slayer.dense(512, 10)

    def forward(self, spikeInput):
        # Both set of definitions are equivalent. The uncommented version is much faster.

        # spikeLayer1 = self.slayer.spike(self.fc1(self.slayer.psp(spikeInput)))
        # spikeLayer2 = self.slayer.spike(self.fc2(self.slayer.psp(spikeLayer1)))

        spikeLayer1 = self.slayer.spike(self.slayer.psp(self.fc1(spikeInput)))
        spikeLayer2 = self.slayer.spike(self.slayer.psp(self.fc2(spikeLayer1)))

        return spikeLayer2
        # return spikeInput, spikeLayer1, spikeLayer2


def extract_dataset(path='./',dataset_path= './data/mnistMLP/NMNISTsmall.zip'):
    # Extract NMNIST samples
    with zipfile.ZipFile(dataset_path) as zip_file:
        for member in zip_file.namelist():
            if not os.path.exists(os.path.join(path,member)):
                zip_file.extract(member, path)
            else:
                print('extraction path already exist', os.path.join(path,member))


# extract_dataset()

device = torch.device("cuda" if USE_CUDA else "cpu")

#Create network instance.
net = Network(netParams).to(device)

# Create snn loss instance.
error = snn.loss(netParams).to(device)

# Define optimizer module.
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)

# Dataset and dataLoader instances.
trainingSet = nmnistDataset(datasetPath =netParams['training']['path']['in'],
                            sampleFile  =netParams['training']['path']['train'],
                            samplingTime=netParams['simulation']['Ts'],
                            sampleLength=netParams['simulation']['tSample'])
trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers=4)

testingSet = nmnistDataset(datasetPath  =netParams['training']['path']['in'],
                            sampleFile  =netParams['training']['path']['test'],
                            samplingTime=netParams['simulation']['Ts'],
                            sampleLength=netParams['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=8, shuffle=False, num_workers=4)

# Learning stats instance.
stats = snn.learningStats()

# # # Visualize the network.
# for i in range(5):
#   input, target, label = trainingSet[i]
#   io.showTD(io.spikeArrayToEvent(input.reshape((2, 34, 34, -1)).cpu().data.numpy()))

# training loop
for epoch in range(100):
    tSt = datetime.now()

    # Training loop.
    for i, (input, target, label) in enumerate(trainLoader, 0):
        # Move the input and target to correct GPU.
        input = input.to(device)
        target = target.to(device)

        # Forward pass of the network.
        output = net.forward(input)

        # Gather the training stats.
        stats.training.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
        stats.training.numSamples += len(label)

        # Calculate loss.
        loss = error.numSpikes(output, target)

        # Reset gradients to zero.
        optimizer.zero_grad()

        # Backward pass of the network.
        loss.backward()

        # Update weights.
        optimizer.step()

        # Gather training loss stats.
        stats.training.lossSum += loss.cpu().data.item()

        # Display training stats.
        stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

    # Testing loop.
    # Same steps as Training loops except loss backpropagation and weight update.
    for i, (input, target, label) in enumerate(testLoader, 0):
        input = input.to(device)
        target = target.to(device)

        output = net.forward(input)

        stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
        stats.testing.numSamples += len(label)

        loss = error.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().data.item()
        stats.print(epoch, i)

    # Update stats.
    stats.update()

# Plot the results.
plt.figure(1)
plt.semilogy(stats.training.lossLog, label='Training')
plt.semilogy(stats.testing.lossLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(stats.training.accuracyLog, label='Training')
plt.plot(stats.testing.accuracyLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()