from torch import nn


class UpSampling2D(nn.Conv3d):
    def __init__(self, inFeatures, outFeatures):
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
        # print('Kernel Dimension:', kernel)
        # print('Input Channels  :', inChannels)

        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(outFeatures.shape))
        # print('Output Channels :', outChannels)

        super(_denseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)

        if weightScale != 1:
            self.weight = torch.nn.Parameter(weightScale * self.weight)  # scale the weight if needed
            # print('In dense, using weightScale of', weightScale)

    def forward(self, input):
        '''
        '''
        return F.conv3d(input,
                        self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
# class Conv2DTranspose(object):