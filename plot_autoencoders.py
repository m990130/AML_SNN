from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import lib.snn as snn
from lib.datasets.mnistdataset import SMNIST
from lib.utils import spikeTensorToProb, save_model, load_model

from lib.process.Training import Training
from lib.process.Evaluation import Evaluation
from models.Autoencoder import VAE

# CONSTANTS:
USE_CUDA = torch.cuda.is_available()
EPOCHs = 200
SMALL = False
DATASETMODE = 'autoencoderSpike'
MODEL_PTH = 'vae'
LR = 0.001

netParams = snn.params('network_specs/vae.yaml')
print(netParams)

device = torch.device("cuda" if USE_CUDA else "cpu")

# Create network instance.
model = VAE(netParams, hidden_size=100, latent_size=50).to(device)

# Load model
load_model(MODEL_PTH, model)
# Define optimizer module.
optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)

# define training set
testingSet = SMNIST(datasetPath=netParams['training']['path']['in'],
                    samplingTime=netParams['simulation']['Ts'],
                    sampleLength=netParams['simulation']['tSample'],
                    train=False,
                    mode=DATASETMODE,
                    small=SMALL)

testing = Evaluation(netParams, device, optimizer, testingSet)

testing.make_grid(model.decoder, n=5)
