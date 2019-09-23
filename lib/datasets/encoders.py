import torch
from torch.distributions import Poisson


def poisson_spike(x, time_bins):
    shape_org = list(x.shape)
    y = x.reshape(-1)
    samples = []
    for yy in y:
        m1 = Poisson(yy)
        samples.append(m1.sample(sample_shape=(time_bins,)) > 0)
    output = torch.stack(samples, dim=0).float()
    return output.reshape(shape_org + [time_bins])


def uniform_spike(x, time_bins):
    shape_org = list(x.shape)
    shape_target = shape_org + [time_bins]
    output = torch.rand(shape_target)
    a = x.unsqueeze(-1)
    b = torch.cat(time_bins * [a], dim=-1)
    C = 0.33
    output = (C * b > output)
    return output.float()
