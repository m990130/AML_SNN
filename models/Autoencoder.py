import torch
import lib.snn as snn


# Network definition
class UpsamplingAutoencoder(torch.nn.Module):
    def __init__(self, netParams):
        super(UpsamplingAutoencoder, self).__init__()
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


class Encoder(torch.nn.Module):
    def __init__(self, netParams, input_size=(28, 28, 1), hidden_size=200, latent_size=2, vae=False):
        super(Encoder, self).__init__()
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.vae = vae
        self.slayer = slayer
        self.fc1 = slayer.dense(input_size, hidden_size)
        if vae:
            self.fc_mean = slayer.dense(hidden_size, latent_size)
            self.fc_log_var = slayer.dense(hidden_size, latent_size)
        else:
            self.fc2 = slayer.dense(hidden_size, latent_size)

    def forward(self, x):

        # # x = B, C, H, W, T
        # s = x.shape
        # b, c, t = s[0], s[1], s[-1]
        # x = x.view(b,c,1,-1)
        x = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        if self.vae:
            means = self.fc_mean(self.slayer.psp(x))
            log_vars = self.fc_log_var(self.slayer.psp(x))
            return means, log_vars
        else:
            x = self.slayer.spike(self.fc2(self.slayer.psp(x)))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, netParams, output_size=28 * 28, hidden_size=200, latent_size=2, vae=False):
        super(Decoder, self).__init__()
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.vae = vae
        self.slayer = slayer
        self.fc1 = slayer.dense(latent_size, hidden_size)
        self.fc2 = slayer.dense(hidden_size, output_size)

    def forward(self, x):
        if self.vae:
            x = self.slayer.spike(self.fc1(self.slayer.psp(self.slayer.spike(x))))
        else:
            x = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        x = self.slayer.spike(self.fc2(self.slayer.psp(x)))
        return x


class SimpleAutoencoder(torch.nn.Module):
    def __init__(self, netparams, input_size=(28, 28, 1), output_size=28 * 28, hidden_size=200, latent_size=2):
        super(SimpleAutoencoder, self).__init__()

        self.encoder = Encoder(netParams=netparams, input_size=input_size, hidden_size=hidden_size,
                               latent_size=latent_size)
        self.decoder = Decoder(netParams=netparams, output_size=output_size, hidden_size=hidden_size,
                               latent_size=latent_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(-1, 1, 28, 28, x.shape[-1])
        return x


class VAE(torch.nn.Module):
    def __init__(self, netparams, input_size=(28, 28, 1), output_size=28 * 28, hidden_size=200, latent_size=2):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(netParams=netparams, input_size=input_size, hidden_size=hidden_size,
                               latent_size=latent_size, vae=True)
        self.decoder = Decoder(netParams=netparams, output_size=output_size, hidden_size=hidden_size,
                               latent_size=latent_size)

    def forward(self, x):
        # encoder net produces means and log(vars)
        means, log_var = self.encoder(x)
        # reparametrization trick
        std = log_var.mul(0.5).exp_()
        normal = torch.distributions.normal.Normal(0, torch.ones(self.latent_size))
        eps = normal.sample((len(x),)).to(x.device)
        eps = eps.unsqueeze(-1)
        eps = eps.unsqueeze(-1)
        eps = torch.stack(x.shape[-1] * [eps], dim=-1)
        z = eps.mul(std).add(means)

        x = self.decoder(z)
        x = x.reshape(-1, 1, 28, 28, x.shape[-1])
        return x, means, log_var
