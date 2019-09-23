from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import lib.snn as snn
from lib.datasets.mnistdataset import SMNIST
from lib.utils import spikeTensorToProb, save_model
from Criterion import Criterion


class Training():
    def __init__(self, netParams, device, optimizer, trainingSet):
        self.netParams = netParams
        self.trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers=4)
        self.device = device
        self.optimizer = optimizer
        error = snn.loss(self.netParams).to(self.device)
        self.criterion = Criterion(error, netParams['training']['error']['type'])

    def train(self, net, stats):
        # Training loop.
        net.train()
        for i, (sample, target, label) in enumerate(self.trainLoader, 0):
            # Move the input and target to correct GPU.
            sample = sample.to(self.device)
            target = target.to(self.device)

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
            loss = self.criterion(output, target)
            # Reset gradients to zero.
            self.optimizer.zero_grad()

            # Backward pass of the network.
            loss.backward()

            # Update weights.
            self.optimizer.step()

            # print('loss ', loss.item())
            # Gather training loss stats.
            stats.training.lossSum += loss.cpu().data.item()

            # Display training stats.
            # if i % 100 == 0:
            #    stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

    def eval(self):
        pass
