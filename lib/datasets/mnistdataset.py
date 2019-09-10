import torch
from torch.distributions import Poisson
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


def poisson_spike(x, time_bins):
    shape_org = list(x.shape)
    y = x.reshape(-1)
    samples = []
    for yy in y:
        m1 = Poisson(yy)
        samples.append(m1.sample(sample_shape=(time_bins,)) > 0)
    output = torch.stack(samples,dim=0).float()
    return output.reshape(shape_org+[time_bins])

def uniform_spike(x, time_bins):
    shape_org = list(x.shape)
    shape_target = shape_org+[time_bins]
    output = torch.rand(shape_target)
    a = x.unsqueeze(-1)
    b = torch.cat(10 * [a], dim=-1)
    C = 0.2
    output = (C*b > output)
    return output.float()




class SMNIST(Dataset):
    def __init__(self, datasetPath, samplingTime, sampleLength, small=True):
        self.path = datasetPath
        if small:
            ds = MNIST(datasetPath, train=False, download=True ,transform=transforms.Compose([
                           transforms.ToTensor()]))
            self.samples = [ds[i] for i in range(0,500)]
        else:
            self.samples = MNIST(datasetPath, train=False, download=True ,transform=transforms.Compose([
                           transforms.ToTensor()]))
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        x, classLabel = self.samples[index]
        x_spikes = poisson_spike(x, self.nTimeBins)

        desiredClass = torch.zeros((10, 1, 1, 1))
        desiredClass[classLabel, ...] = 1
        return x_spikes, desiredClass, classLabel

    def __len__(self):
        return len(self.samples)

