import unittest
from unittest import TestCase
from lib.datasets.mnistdataset import SMNIST

from lib import snn
from .slayerUpsampling import UpSampling2D


class testClass(TestCase):

    def setUp(self):
        self.netParams = snn.params('../data/mnist/network.yaml')

        self.dataset = SMNIST(datasetPath=self.netParams['training']['path']['in'],
                              samplingTime=self.netParams['simulation']['Ts'],
                              sampleLength=self.netParams['simulation']['tSample'])
        self.m = UpSampling2D(scale_factor=2, mode='nearest')

    def test_forward(self):
        sample = self.dataset[0]
        data, target = sample[0], sample[1]
        print(data.shape)
        print(target.shape)
        upsampled = self.m(data)
        self.assertEqual(len(upsampled.shape), len(data.shape))
        # TODO: more asssertions ....


if __name__ == '__main__':
    unittest.main()
