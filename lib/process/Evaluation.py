from torch.utils.data import Dataset, DataLoader
import lib.snn as snn
from Criterion import Criterion
import torch
import matplotlib.pyplot as plt
import numpy as np
from lib.datasets.encoders import uniform_spike

class Evaluation:
    def __init__(self, netParams, device, optimizer, testSet, classification=False):
        self.netParams = netParams
        self.testLoader = DataLoader(dataset=testSet, batch_size=8, shuffle=False, num_workers=4)
        self.device = device
        self.optimizer = optimizer
        error = snn.loss(self.netParams).to(self.device)
        self.criterion = Criterion(error, netParams['training']['error']['type'])
        self.theta = netParams['neuron']['theta']
        self.classification = classification

    def test(self, model, stats, epoch=-1):
        # # Testing loop.
        # # Same steps as Training loops except loss backpropagation and weight update.
        for i, (sample, target, label) in enumerate(self.testLoader, 0):
            sample = sample.to(self.device)
            target = target.to(self.device)

            output = model.forward(sample)

            if self.classification:
                stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
            stats.testing.numSamples += len(label)

            loss = self.criterion(output, target)
            stats.testing.lossSum += loss.cpu().data.item()
            stats.print(epoch, i)
        print('\n\n\n\n')

    def make_grid(self, img, encoder, decoder, n=10, conditional=False, digit=None):

        normal = torch.distributions.normal.Normal(0, torch.ones(1))
        # we need to generate z values corresponding to a evenly spaced
        # probability, so the inverse cdf function is required
        z1 = normal.icdf(torch.linspace(1e-3, 1 - 1e-3, n))
        z2 = normal.icdf(torch.linspace(1e-3, 1 - 1e-3, n))
        # the min/max value for the axis
        z_min, z_max = z1.min().numpy(), z1.max().numpy()

        X = encoder(img)

        grid_z = torch.stack(torch.meshgrid(z1, z2), dim=2).view(-1, 2, 1, 1)
        grid_z = uniform_spike(grid_z, 50)
        z1 = X[0]
        z2 = X[1]
        plt.figure(figsize=(10, 10))
        pred_x = decoder(grid_z.to(self.device))  # .detach().cpu().view(n, n, 28, 28)

        plt.imshow(np.block(list(map(list, pred_x.numpy()))), cmap='gray',
                   origin='upper', extent=[z_min, z_max, z_min, z_max])
        plt.show()
