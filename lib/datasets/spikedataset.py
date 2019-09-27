import torch
from torch.utils.data import Dataset

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.transforms import transforms



class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im
# the following transformation is used in Cao's paper, and the value of piexl is within [0, 1]    


def transform_cifar(is_train):
    
    return transforms.Compose([
                                transforms.RandomCrop(size=26) if is_train else Identity(),
                                transforms.RandomHorizontalFlip(p=0.5) if is_train else Identity(),
                                transforms.CenterCrop(size=26) if not is_train else Identity(),
                                transforms.ToTensor(),
                            ])


# func to generate spikes, we use uniform as default
def poisson_spike(x, time_bins):
    shape_org = list(x.shape)
    y = x.reshape(-1)
    samples = []
    for yy in y:
        m1 = Poisson(yy)
        samples.append(m1.sample(sample_shape=(time_bins,)) > 0)
    output = torch.stack(samples, dim=0).float()
    return output.reshape(shape_org + [time_bins])


def uniform_spike(x, time_bins):
    shape_org = list(x.shape)
    shape_target = shape_org + [time_bins]
    output = torch.rand(shape_target)
    a = x.unsqueeze(-1)
    b = torch.cat(time_bins * [a], dim=-1)
    C = 0.33
    output = (C * b > output)
    return output.float()


class SpikeDataset(Dataset):
    def __init__(self, datasetPath, dataset,TimeBins=25, small=True, train=True, encoding='uniform',
                 mode='classification'):
        self.mode = mode
        self.path = datasetPath
        self.dataset = dataset
        
        if self.dataset == 'mnist':
            if small:
                ds = MNIST(datasetPath, train=train, download=True, transform=transforms.Compose([
                    transforms.ToTensor()]))
                self.samples = [ds[i] for i in range(0, 500)]
            else:
                self.samples = MNIST(datasetPath, train=train, download=True, transform=transforms.Compose([
                    transforms.ToTensor()]))
        
        elif self.dataset == 'fashion':
            if small:
                ds = FashionMNIST(datasetPath, train=train, download=True, transform=transforms.Compose([
                    transforms.ToTensor()]))
                self.samples = [ds[i] for i in range(0, 500)]
            else:
                self.samples = FashionMNIST(datasetPath, train=train, download=True, transform=transforms.Compose([
                    transforms.ToTensor()]))
        
        elif self.dataset == 'cifar10':
            cifar_transform = transform_cifar(train)
                        
            if small:
                ds = CIFAR10(datasetPath, train=train, download=True, transform=cifar_transform)
                self.samples = [ds[i] for i in range(0, 500)]
            else:
                self.samples = CIFAR10(datasetPath, train=train, download=True, transform=cifar_transform)
        else:
            raise Exception(
                'the choosen dataset {} is not valid . Valid are mnist, fashion and cifar10'.format(self.dataset))
        self.nTimeBins = TimeBins
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