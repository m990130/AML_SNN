import torch
import lib.snn as snn


class SlayerVgg16(torch.nn.Module):
    def __init__(self, netParams, input_channels=3):
        super(SlayerVgg16, self).__init__()
        # initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.conv1a = slayer.conv(input_channels, 16, 5, padding=1)
        self.conv2a = slayer.conv(16, 16, 3, padding=1)
        self.poola = slayer.pool(2)

        self.conv1b = slayer.conv(16, 32, 3, padding=1)
        self.conv2b = slayer.conv(32, 32, 3, padding=1)
        self.poolb = slayer.pool(2)

        self.conv1c = slayer.conv(32, 64, 3, padding=1)
        self.conv2c = slayer.conv(64, 64, 3, padding=1)
        self.conv3c = slayer.conv(64, 64, 3, padding=1)
        self.poolc = slayer.pool(2)

        # self.conv1d = slayer.conv(256, 512, 3, padding=1)
        # self.conv2d = slayer.conv(512, 512, 3, padding=1)
        # self.conv3d = slayer.conv(512, 512, 3, padding=1)
        # self.poold = slayer.pool(2)
        #
        # self.conv1e = slayer.conv(512, 512, 3, padding=1)
        # self.conv2e = slayer.conv(512, 512, 3, padding=1)
        # self.conv3e = slayer.conv(512, 512, 3, padding=1)
        # self.poole = slayer.pool(2)

        self.fc1 = slayer.dense((4, 4, 64), 10)
        # self.fc2 = slayer.dense(4096, 4096)
        # self.fc3 = slayer.dense(4096, 1000)
        # self.fc4 = slayer.dense(1000, 10)

    def forward(self, x):
        x = self.slayer.spike(self.conv1a(self.slayer.psp(x)))  # 32, 32, 16
        x = self.slayer.spike(self.conv2a(self.slayer.psp(x)))  # 32, 32, 16
        x = self.slayer.spike(self.poola(self.slayer.psp(x)))  # 16, 16, 16

        x = self.slayer.spike(self.conv1b(self.slayer.psp(x)))  # 32, 32, 16
        x = self.slayer.spike(self.conv2b(self.slayer.psp(x)))  # 32, 32, 16
        x = self.slayer.spike(self.poolb(self.slayer.psp(x)))  # 16, 16, 16

        x = self.slayer.spike(self.conv1c(self.slayer.psp(x)))  # 32, 32, 16
        x = self.slayer.spike(self.conv2c(self.slayer.psp(x)))  # 32, 32, 16
        x = self.slayer.spike(self.conv3c(self.slayer.psp(x)))  # 32, 32, 16
        x = self.slayer.spike(self.poolc(self.slayer.psp(x)))  # 8,  8, 32

        # x = self.slayer.spike(self.conv1d(self.slayer.psp(x)))  # 32, 32, 16
        # x = self.slayer.spike(self.conv2d(self.slayer.psp(x)))  # 32, 32, 16
        # # x = self.slayer.spike(self.conv3d(self.slayer.psp(x)))  # 16, 16, 32
        # x = self.slayer.spike(self.poold(self.slayer.psp(x)))  # 8,  8, 32

        # x = self.slayer.spike(self.conv1e(self.slayer.psp(x)))  # 32, 32, 16
        # x = self.slayer.spike(self.conv2e(self.slayer.psp(x)))  # 32, 32, 16
        # # x = self.slayer.spike(self.conv3e(self.slayer.psp(x)))  # 16, 16, 32
        # x = self.slayer.spike(self.poole(self.slayer.psp(x)))  # 8,  8, 32

        # print('1. x.max(), ', x.data.sum())

        x = self.slayer.spike(self.fc1(self.slayer.psp(x)))  # 10
        # x = self.slayer.spike(self.fc2(self.slayer.psp(x)))  # 10
        # x = self.slayer.spike(self.fc3(self.slayer.psp(x)))  # 10
        # x = self.slayer.spike(self.fc4(self.slayer.psp(x)))  # 10
        # print('2. x.max(), ', x.data.sum())

        return x


class SlayerSNN(torch.nn.Module):
    def __init__(self, netParams, input_channels=3):
        super(SlayerSNN, self).__init__()
        # initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.conv1 = slayer.conv(input_channels, 16, 5, padding=1)
        self.conv2 = slayer.conv(16, 32, 3, padding=1)
        self.conv3 = slayer.conv(32, 64, 3, padding=1)
        self.pool1 = slayer.pool(2)
        self.pool2 = slayer.pool(2)
        self.fc1 = slayer.dense((8, 8, 64), 10)

    def forward(self, spikeInput):
        spikeLayer1 = self.slayer.spike(self.conv1(self.slayer.psp(spikeInput)))  # 32, 32, 16
        spikeLayer2 = self.slayer.spike(self.pool1(self.slayer.psp(spikeLayer1)))  # 16, 16, 16
        spikeLayer3 = self.slayer.spike(self.conv2(self.slayer.psp(spikeLayer2)))  # 16, 16, 32
        spikeLayer4 = self.slayer.spike(self.pool2(self.slayer.psp(spikeLayer3)))  # 8,  8, 32
        spikeLayer5 = self.slayer.spike(self.conv3(self.slayer.psp(spikeLayer4)))  # 8,  8, 64
        spikeOut = self.slayer.spike(self.fc1(self.slayer.psp(spikeLayer5)))  # 10

        return spikeOut