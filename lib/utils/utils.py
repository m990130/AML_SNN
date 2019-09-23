import os

import torch
from torch import Tensor


def spikeTensorToProb(spikeTensor):
    assert isinstance(spikeTensor, Tensor)
    return spikeTensor.float().mean(dim=-1)


def save_model(name, model):
    print("Saving models...")
    model.eval()

    save_model_filename = 'saved_models/' + name + '.pt'

    torch.save(model.state_dict(), save_model_filename)


def load_model(name, model):
    dict_path = 'saved_models/' + name + '.pt'
    # if find dict path and copies forced to be in the cpu
    if os.path.exists(dict_path):
        model.load_state_dict(torch.load(dict_path, map_location=lambda storage, location: storage))


def load_dict(name):
    return torch.load('saved_models/' + name + '.pt')
