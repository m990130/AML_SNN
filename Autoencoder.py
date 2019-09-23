import torch
import lib.snn as snn


# Network definition
class Autoencoder(torch.nn.Module):
    def __init__(self, netParams):
        super(Autoencoder, self).__init__()
        # initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.conv1 = slayer.conv(1, 16, 5, padding=2)
        self.pool1 = slayer.pool(2)
        self.conv2 = slayer.conv(16, 32, 3, padding=1)
        self.pool2 = slayer.pool(2)
        self.conv3 = slayer.conv(32, 64, 3, padding=1)
        self.unpool3 = slayer.upsampling2d(2, mode='bilinear')
        self.conv4 = slayer.conv(64, 32, 3, padding=1)
        self.unpool4 = slayer.upsampling2d(2, mode='bilinear')
        self.convOut = slayer.conv(32, 1, 1, padding=0)

    def forward(self, spikeInput):
        spikeLayer1 = self.slayer.spike(self.conv1(self.slayer.psp(spikeInput)))  # 32, 32, 16
        spikeLayer2 = self.slayer.spike(self.pool1(self.slayer.psp(spikeLayer1)))  # 16, 16, 16
        spikeLayer3 = self.slayer.spike(self.conv2(self.slayer.psp(spikeLayer2)))  # 16, 16, 32
        spikeLayer4 = self.slayer.spike(self.pool2(self.slayer.psp(spikeLayer3)))  # 8,  8, 32
        spikeLayer5 = self.slayer.spike(self.conv3(self.slayer.psp(spikeLayer4)))  # 8,  8, 64
        spikeLayer6 = self.slayer.spike(self.unpool3(self.slayer.psp(spikeLayer5)))  # 8,  8, 64
        spikeLayer7 = self.slayer.spike(self.conv4(self.slayer.psp(spikeLayer6)))  # 8,  8, 64
        spikeLayer8 = self.slayer.spike(self.unpool4(self.slayer.psp(spikeLayer7)))  # 8,  8, 64
        spikeOut = self.slayer.spike(self.convOut(self.slayer.psp(spikeLayer8)))  # 8,  8, 64
        return spikeOut
