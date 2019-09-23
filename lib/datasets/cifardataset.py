import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from .encoders import uniform_spike, poisson_spike
import matplotlib.pyplot as plt


class SCIFAR(Dataset):
    def __init__(self, datasetPath, samplingTime, sampleLength, small=True, train=True, encoding='uniform',
                 mode='classification'):
        self.mode = mode
        self.path = datasetPath
        if small:
            ds = CIFAR10(datasetPath, train=train, download=True, transform=transforms.Compose([
                transforms.ToTensor()]))
            self.samples = [ds[i] for i in range(0, 500)]
        else:
            self.samples = CIFAR10(datasetPath, train=train, download=True, transform=transforms.Compose([
                transforms.ToTensor()]))
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLength / samplingTime)
        self.encoding = encoding

    def __getitem__(self, index):
        x, classLabel = self.samples[index]
        if self.encoding == 'uniform':
            x_spikes = uniform_spike(x, self.nTimeBins)
        else:
            x_spikes = poisson_spike(x, self.nTimeBins)
        if self.mode == 'classification':
            desiredClass = torch.zeros((10, 1, 1, 1))
            desiredClass[classLabel, ...] = 1
            return x_spikes, desiredClass, classLabel
        elif self.mode == 'autoencoder':
            return x_spikes, x, classLabel
        elif self.mode == 'autoencoderSpike':
            return x_spikes, x_spikes, classLabel
        else:
            raise Exception(
                'mode is not valid {}. Valid are classification, autoencoder, autoencoderSpike'.format(self.mode))

    def __len__(self):
        return len(self.samples)


class CIFAR(Dataset):
    def __init__(self, datasetPath, train=False, transform=transforms.Compose([
        transforms.ToTensor()])):
        self.path = datasetPath
        ds = CIFAR10(datasetPath, train=train, download=True, transform=transform)
        self.samples = [ds[i] for i in range(0, 500)]

    def __getitem__(self, index):
        x, y = self.samples[index]
        return x, y

    def __len__(self):
        return len(self.samples)
