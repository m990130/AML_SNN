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


def test_acc(dataloader ,model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels =  data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network: %d %%' % (
        100 * correct / total))