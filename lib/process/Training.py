from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import lib.snn as snn
from lib.datasets.mnistdataset import SMNIST
from lib.utils import spikeTensorToProb, save_model
from Criterion import Criterion
from .WeightCollector import WeightCollector
import pickle

class Training():
    def __init__(self, netParams, device, optimizer, trainingSet, classification=False, collect_weights=False):
        self.netParams = netParams
        self.trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers=4)
        self.device = device
        self.optimizer = optimizer
        error = snn.loss(self.netParams).to(self.device)
        self.criterion = Criterion(error, netParams['training']['error']['type'])
        self.classification = classification
        self.weightCollector = WeightCollector() if collect_weights else None

    def train(self, net, stats, epoch=-1, tSt=None):
        tSt = datetime.now()
        # Training loop.
        net.train()
        N = 0
        correct = 0
        c = 0
        n = 0
        for i, (sample, target, label) in enumerate(self.trainLoader, 0):
            # Move the input and target to correct GPU.
            sample = sample.to(self.device)
            target = target.to(self.device)

            # Forward pass of the network.
            output = net.forward(sample)

            # # Gather the training stats.
            if self.classification:
                c = torch.sum(snn.predict.getClass(output) == label).data.item()
                correct += c
                n = len(label)
                N += n
                stats.training.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
            stats.training.numSamples += len(label)
            #
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
            if self.weightCollector and i % 10 == 0:
                self.weightCollector.capture(net, loss.item())

            # print('acc ', c/n) if n>0 else print('acc U')
            # Gather training loss stats.
            stats.training.lossSum += loss.cpu().data.item()

            # Display training stats.
            # if i % 100 == 0:
            #     stats.print(epoch, i, (datetime.now() - tSt).total_seconds())
            stats.print(epoch, i, (datetime.now() - tSt).total_seconds())
        if self.classification:
            print('acc epoch ', correct / N) if N > 0 else print('acc epoch U')
            print('\n\n\n\n')
        if self.weightCollector:
            with open('name.p', 'wb') as f:
                pickle.dump(self.weightCollector, f)

    def eval(self):
        pass
