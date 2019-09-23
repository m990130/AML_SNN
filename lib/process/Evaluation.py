from torch.utils.data import Dataset, DataLoader
import lib.snn as snn
from Criterion import Criterion
import torch


class Evaluation:
    def __init__(self, netParams, device, optimizer, testSet, accuracy=False):
        self.netParams = netParams
        self.testLoader = DataLoader(dataset=testSet, batch_size=2, shuffle=False, num_workers=4)
        self.device = device
        self.optimizer = optimizer
        error = snn.loss(self.netParams).to(self.device)
        self.criterion = Criterion(error, netParams['training']['error']['type'])
        self.accuracy = accuracy

    def test(self, model, stats):
        # # Testing loop.
        # # Same steps as Training loops except loss backpropagation and weight update.
        for i, (sample, target, label) in enumerate(self.testLoader, 0):
            sample = sample.to(self.device)
            target = target.to(self.device)

            output = model.forward(sample)

            if self.accuracy:
                stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
                stats.testing.numSamples += len(label)

            loss = self.criterion(output, target)
            stats.testing.lossSum += loss.cpu().data.item()
            # stats.print(epoch, i)
