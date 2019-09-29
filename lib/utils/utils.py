import os

import torch
from torch import Tensor

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt


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


def get_labels(model, testloader, snn=False):
    
    labels = []
    preds = []
    use_cuda = next(model.parameters()).is_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if snn:
        for i, (input, _, label) in enumerate(testloader, 0):
            
            input = input.to(device)

            pred  = model(input)
            pred = spikeClassifier.getClass(pred)
            labels.append(label)
            preds.append(pred)
    else:
        for i, (input, label) in enumerate(testloader, 0):
            input = input.to(device)

            pred  = model(input)
            pred = torch.argmax(pred,dim=1)
            labels.append(label)
            preds.append(pred)
            
    # return the results as a 1-D tensor
    labels = torch.cat([torch.stack(labels[:-1]).flatten(), labels[-1]])
    preds = torch.cat([torch.stack(preds[:-1]).flatten(), preds[-1]])
    
    return labels, preds

def plot_confusion_matrix(y_true, y_pred,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def test_method(labels, preds):
    print("The classification accuracy: ", accuracy_score(labels, preds))
    print()
    plot_confusion_matrix(labels, preds)
    

class spikeClassifier:
    '''
    It provides classification modules for SNNs.
    All the functions it supplies are static and can be called without making an instance of the class.
    '''
    @staticmethod
    def getClass(spike):
        '''
        Returns the predicted class label.
        It assignes single class for the SNN output for the whole simulation runtime.

        Usage:

        >>> predictedClass = spikeClassifier.getClass(spikeOut)
        '''
        numSpikes = torch.sum(spike, 4, keepdim=True).cpu()
        return torch.max(numSpikes.reshape((numSpikes.shape[0], -1)), 1)[1]