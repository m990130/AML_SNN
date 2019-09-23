from unittest import TestCase
from .utils import spikeTensorToProb
import torch


class TestUtils(TestCase):
    def test_spike_to_prob(self):
        a = torch.tensor([[[1, 1, 1], [0, 0, 0]], [[0, 0, 1], [1, 0, 1]]], dtype=torch.float)
        result = spikeTensorToProb(a)
        self.assertEqual(result[0][0].item(), 1.0)
        self.assertEqual(result[0][1].item(), 0.0)
        self.assertAlmostEqual(result[1][0].item(), 0.33333333333333)
        self.assertAlmostEqual(result[1][1].item(), 0.66666666666666)
