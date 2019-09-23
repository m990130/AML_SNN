import torch
import numpy as np
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter, uniform_filter
import matplotlib.pyplot as plt
import torchvision.models as models
from skimage.transform import resize
from torch.autograd import Variable
from tqdm import tqdm

"""
The classes SaveFeatures and Visualization are based on the work from Fabio M. Graetz.
See https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
and https://github.com/fg91/visualizing-cnn-feature-maps/blob/master/filter_visualizer.ipynb.
"""


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # sourceTensor.clone().detach().requires_grad_(True)
        self.features = output  # Variable(output, requires_grad=True)

    def close(self):
        self.hook.remove()


class Visualization:
    def __init__(self, model, device=None, blurring=True):
        self.model = model.eval()
        self.device = torch.device('cpu') if device is None else device
        self.model.trainable = False
        for p in model.parameters():
            p.requires_grad = False
        self.blurring = blurring

    def visualize(self, layer, kernel, size=56, lr=0.1, upscaling_steps=12, upscaling_factor=1.2,
                  optimization_steps=20):
        # generate random image
        img = np.random.uniform(150, 180, (size, size, 3)) / 255
        # register hook
        conv_layer = list(self.model.children())[0]
        activations = SaveFeatures(list(conv_layer.children())[layer])
        # create tranformers;

        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=means,
                                         std=stds)
        tranformation = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize(size),
            transforms.ToTensor(),
            normalize,
        ])

        for i in range(upscaling_steps):

            img_tensor = tranformation(img)
            img_tensor = img_tensor.unsqueeze(0).float().to(self.device)
            img_opt = Variable(img_tensor, requires_grad=True)

            optimizer = torch.optim.Adam([img_opt], lr=lr, weight_decay=1e-6)

            for ii in range(optimization_steps):
                optimizer.zero_grad()
                self.model(img_opt)
                loss = -activations.features[0, kernel].mean()
                # loss = -img_opt.mean()
                loss.backward()
                optimizer.step()
                # print('Current loss: opt_step ', ii, 'upscaling step', i, loss.item())
            # denormalization
            for c, (m, s) in enumerate(zip(means, stds)):
                img_opt[0][c] = s * img_opt[0][c] + m

            img = img_opt.clone().detach().cpu().numpy()[0]
            img = uniform_filter(img, 2).transpose(1, 2, 0)
            # img = gaussian_filter(img, 0.3).transpose(1, 2, 0)
            # skimage.filters.gaussian_filter(im, 2, multichannel=True, mode='reflect', 'truncate=2')
            size = int(size * upscaling_factor)
            img = resize(img, (size, size, 3), order=3)

        output = img
        cropped_output = np.clip(output, 0, 1)
        fig = plt.figure()
        plt.imshow(cropped_output)
        plt.show()
        plt.imsave("layer_" + str(layer) + "_filter_" + str(kernel) + ".jpg", cropped_output)
        plt.close()
        activations.close()


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = models.vgg16(pretrained=True).to(device)
    print(model)
    v = Visualization(model, device)
    v.visualize(layer=28, kernel=400, optimization_steps=20, upscaling_steps=12)
