from unittest import TestCase
from .mnistdataset import SMNIST


class test_mnist_dataset(TestCase):
    def test_init(self):
        dataset = SMNIST(datasetPath='../data/smnist',samplingTime=1,sampleLength=200)

    def test_get_item(self):
        dataset = SMNIST(datasetPath='../data/smnist', samplingTime=1, sampleLength=200)
        item = dataset[0]
    def test_encoder(self):
        dataset = SMNIST(datasetPath='../data/smnist', samplingTime=1, sampleLength=200)
        input, target, label = dataset[0]
