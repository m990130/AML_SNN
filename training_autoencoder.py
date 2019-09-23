from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import lib.snn as snn
from lib.datasets.mnistdataset import SMNIST
from lib.utils import spikeTensorToProb, save_model

# CONSTANTS:

USE_CUDA = torch.cuda.is_available()
DATASETMODE = 'autoencoderSpike'
# DATASETMODE = 'autoencoder'
EPOCHs = 300
SMALL = False

netParams = snn.params('network_specs/autoencoder.yaml')
print(netParams)


# Network definition
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.conv1 = slayer.conv(1, 16, 5, padding=2)
        self.pool1 = slayer.pool(2)
        self.conv2 = slayer.conv(16, 32, 3, padding=1)
        self.pool2 = slayer.pool(2)
        self.conv3 = slayer.conv(32, 64, 3, padding=1)
        self.unpool3 = slayer.upsampling2d(2, mode='bilinear')
        self.conv4 = slayer.conv(64, 32, 3, padding=1)
        self.unpool4 = slayer.upsampling2d(2, mode='bilinear')
        self.convOut = slayer.conv(32, 1, 1, padding=0)

    def forward(self, spikeInput):
        spikeLayer1 = self.slayer.spike(self.conv1(self.slayer.psp(spikeInput)))  # 32, 32, 16
        spikeLayer2 = self.slayer.spike(self.pool1(self.slayer.psp(spikeLayer1)))  # 16, 16, 16
        spikeLayer3 = self.slayer.spike(self.conv2(self.slayer.psp(spikeLayer2)))  # 16, 16, 32
        spikeLayer4 = self.slayer.spike(self.pool2(self.slayer.psp(spikeLayer3)))  # 8,  8, 32
        spikeLayer5 = self.slayer.spike(self.conv3(self.slayer.psp(spikeLayer4)))  # 8,  8, 64
        spikeLayer6 = self.slayer.spike(self.unpool3(self.slayer.psp(spikeLayer5)))  # 8,  8, 64
        spikeLayer7 = self.slayer.spike(self.conv4(self.slayer.psp(spikeLayer6)))  # 8,  8, 64
        spikeLayer8 = self.slayer.spike(self.unpool4(self.slayer.psp(spikeLayer7)))  # 8,  8, 64
        spikeOut = self.slayer.spike(self.convOut(self.slayer.psp(spikeLayer8)))  # 8,  8, 64
        return spikeOut


device = torch.device("cuda" if USE_CUDA else "cpu")

# Create network instance.
net = Network(netParams).to(device)

# Create snn loss instance.
error = snn.loss(netParams).to(device)

# Define optimizer module.
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, amsgrad=True)

# Dataset and dataLoader instances.


trainingSet = SMNIST(datasetPath=netParams['training']['path']['in'],
                     samplingTime=netParams['simulation']['Ts'],
                     sampleLength=netParams['simulation']['tSample'],
                     mode=DATASETMODE,
                     small=SMALL)
trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers=4)

testingSet = SMNIST(datasetPath=netParams['training']['path']['in'],
                    samplingTime=netParams['simulation']['Ts'],
                    sampleLength=netParams['simulation']['tSample'],
                    mode=DATASETMODE,
                    small=SMALL)
testLoader = DataLoader(dataset=testingSet, batch_size=2, shuffle=False, num_workers=4)

# Learning stats instance.
stats = snn.learningStats()


# # # # Visualize the network.
# # for i in range(5):
# #   input, target, label = trainingSet[i]
# #   io.showTD(io.spikeArrayToEvent(input.reshape((2, 34, 34, -1)).cpu().data.numpy()))
#
def loss_mse(x, y):
    x = spikeTensorToProb(x).view(-1).float()
    y = y.view(-1).float()
    error = (x - y) ** 2
    return error.sum()


# training loop
for epoch in range(EPOCHs):
    tSt = datetime.now()

    # Training loop.
    net.train()
    for i, (sample, target, label) in enumerate(trainLoader, 0):
        # Move the input and target to correct GPU.
        sample = sample.to(device)
        target = target.to(device)

        # Forward pass of the network.
        output = net.forward(sample)

        # # Gather the training stats.
        # stats.training.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
        stats.training.numSamples += len(label)
        #
        # sample = spikeTensorToProb(sample)

        # Calculate loss.
        # loss = error.numSpikes(output, target)
        # loss = loss_mse(output,target)
        if DATASETMODE == 'autoencoderSpike':
            loss = error.spikeTime(output, target)
        elif DATASETMODE == 'autoencoder':
            loss = loss_mse(output, target)
        else:
            raise Exception('invalid dataset mode.loss faide')
        # Reset gradients to zero.
        optimizer.zero_grad()

        # Backward pass of the network.
        loss.backward()

        # Update weights.
        optimizer.step()

        # print('loss ', loss.item())
        # Gather training loss stats.
        stats.training.lossSum += loss.cpu().data.item()

        # Display training stats.
        # stats.print(epoch, i, (datetime.now() - tSt).total_seconds())
        if i % 100 == 0:
            stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

    #     pred = spikeTensorToProb(output)
    #     target_grid = torchvision.utils.make_grid(target)
    #     pred_grid = torchvision.utils.make_grid(pred)
    #     plt.subplot(2, 1, 1)
    #     plt.imshow(pred_grid.detach().cpu().numpy().transpose(1, 2, 0))
    #     plt.subplot(2, 1, 2)
    #     plt.imshow(target_grid.detach().cpu().numpy().transpose(1, 2, 0))
    #     plt.show()

    # # Testing loop.
    # # Same steps as Training loops except loss backpropagation and weight update.
    # for i, (sample, target, label) in enumerate(testLoader, 0):
    #     sample = sample.to(device)
    #     target = target.to(device)
    #
    #     output = net.forward(sample)
    #
    #     stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
    #     stats.testing.numSamples += len(label)
    #
    #     loss = error.numSpikes(output, target)
    #     stats.testing.lossSum += loss.cpu().data.item()
    #     stats.print(epoch, i)

    # Update stats.
    stats.update()
    if epoch % 100 == 0:
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
plt.show()
# plt.plot(stats.training.accuracyLog, label='Training')
# plt.plot(stats.testing.accuracyLog, label='Testing')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

plt.show()
