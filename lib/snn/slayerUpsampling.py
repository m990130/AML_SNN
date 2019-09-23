from torch import Tensor
from torch import nn
import torch.nn.functional as F

class UpSampling2D(nn.Upsample):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        '''
        '''
        super(UpSampling2D, self).__init__(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

        # if weightScale != 1:
        #     self.weight = torch.nn.Parameter(weightScale * self.weight)  # scale the weight if needed
        #     # print('In dense, using weightScale of', weightScale)

    def forward(self, input):
        '''
        '''
        assert len(input.shape) == 5 or len(input.shape) == 4
        assert isinstance(input, Tensor)

        has_batch = True
        if len(input.shape) == 4:
            input = input.unsqueeze(0)
            has_batch = False
        batch, c, h, w, t = input.shape[0], input.shape[1],input.shape[2], input.shape[3],input.shape[4]

        e = input.permute(0, 1, 4, 2, 3).contiguous().view(batch, c*t, h, w)
        f = F.interpolate(e, self.size, self.scale_factor, self.mode, self.align_corners)
        f = f.view(batch, c, t, f.shape[-2], f.shape[-1]).permute(0, 1, 3, 4, 2).contiguous()
        g = f if has_batch else f[0]
        return g




# class Conv2DTranspose(object):