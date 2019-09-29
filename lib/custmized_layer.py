import torch
import torch.nn as nn
import torch.nn.functional as F


def LIF(spikes, theta, leak, V_min):
    '''
    Integrate-and-fire: given a tensor with shape (B,C,H,W,T), loop over T

    Params:
        spikes: the spikes from previous layer, containing 0 or 1.
        theta: threshold to fire a spike.
        l: leakage parameter.
        V_min: the resting state of membrane potential, usually is set to 0.

    return:
        V: the potential to fire (just for check, can be removed later)
        next_spikes: 0 or 1 tensor with the same shape of input

    '''

    # the padding controls where to pad, first two of the tuple control the last dim of the tensor
    _pad = nn.ConstantPad3d((1, 0, 0, 0, 0, 0), 0)
    pad_spikes = _pad(spikes)

    V = torch.zeros_like(pad_spikes)
    next_Spikes = torch.zeros_like(pad_spikes)

    T = pad_spikes.shape[-1]

    for t in range(1, T):
        # equation (1a)
        V[:, :, :, :, t] = V[:, :, :, :, t - 1] + leak + pad_spikes[:, :, :, :, t]
        # thresholding and fire spike (1b)
        mask_threshold = V[:, :, :, :, t] >= theta
        next_Spikes[:, :, :, :, t][mask_threshold] = 1
        # reset the potential to zero
        V[:, :, :, :, t][mask_threshold] = 0

        # reset the value to V_min if drops below (1c)
        mask_min = (V[:, :, :, :, t] < V_min)
        V[:, :, :, :, t][mask_min] = V_min

    return (V[:, :, :, :, 1:], next_Spikes[:, :, :, :, 1:])


class convLayer(nn.Conv3d):

    def __init__(self, inChannels, outChannels, kernelSize, theta, padding=0, leak=0, V_min=0, check_mode=False):
        
        kernel = (kernelSize, kernelSize, 1)
        
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))
        
        super(convLayer, self).__init__(inChannels, outChannels, kernel, bias=False)
        self.padding = padding
        self.theta = theta
        self.leak = leak
        self.V_min = V_min
        self.check_mode = check_mode

    def forward(self, input):
        # get X, namely eq(2)
        conv_spikes = F.conv3d(input,
                               self.weight, self.bias,
                               self.stride, self.padding, self.dilation, self.groups)

        output = LIF(conv_spikes, self.theta, self.leak, self.V_min)
        
        if self.check_mode:
            return (conv_spikes, *output)
        else:
            return output[1]


class poolLayer(nn.AvgPool3d):
    def __init__(self, kernelSize, theta, leak=0, V_min=0, check_mode=False):

        if type(kernelSize) == int:
            kernel_size = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel_size = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))
        super(poolLayer, self).__init__(kernel_size)
        self.theta = theta
        self.leak = leak
        self.V_min = V_min
        self.check_mode = check_mode

    def forward(self, input):

        # get X, namely eq(2)
        pool_spikes = F.avg_pool3d(input, self.kernel_size)

        output = LIF(pool_spikes, self.theta, self.leak, self.V_min)

        if self.check_mode:
            return (conv_spikes, *output)
        else:
            return output[1]

class denseLayer(nn.Conv3d):
    def __init__(self, inFeatures, outFeatures, theta, padding=0, leak=0, V_min=0, check_mode=False):
        '''
        '''
        # extract information for kernel and inChannels
        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures
        elif len(inFeatures) == 2:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = 1
        elif len(inFeatures) == 3:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = inFeatures[2]
        else:
            raise Exception('inFeatures should not be more than 3 dimension. It was: {}'.format(inFeatures.shape))

        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(outFeatures.shape))
            
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        super(denseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)
        self.padding = padding
        # params for the LIF
        self.theta = theta
        self.leak = leak
        self.V_min = V_min
        self.check_mode = check_mode

    def forward(self, input):
        fc_spikes = F.conv3d(input,
                             self.weight, self.bias,
                             self.stride, self.padding, self.dilation, self.groups)

        output = LIF(fc_spikes, self.theta, self.leak, self.V_min)

        if self.check_mode:
            return (conv_spikes, *output)
        else:
            return output[1]