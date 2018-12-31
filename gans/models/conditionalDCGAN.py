from ..utils.layers import *
from .vanillaGAN import linearBlock
from torch import nn


def dcLayersG(input_dim):
    ngf = 64
    layersG = [
        nn.ConvTranspose2d(input_dim, ngf * 4, 4, 1, 0),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, output_padding=1),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf, 1, 4, 2, 1),
        nn.Tanh()
    ]
    return layersG


def dcLayersD(reduced_dim):
    ndf = 64
    layersD = [
        nn.Conv2d(1, ndf, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, reduced_dim, 3, 1, 0),
        nn.Sigmoid()
    ]
    return layersD


class CondDCGenerator(nn.Module):
    def __init__(self, noize_dim):
        super(CondDCGenerator, self).__init__()
        self.encoder = OneHotEncoder(torch.tensor([10]))
        input_dim = noize_dim + self.encoder.n_values_sum
        # Transposed Conv2d layers
        layers = dcLayersG(input_dim=input_dim)
        self.model = nn.Sequential(Unflatten2d(img_shape=(input_dim, 1, 1)), *layers)

    def forward(self, x, y):
        y_ = self.encoder(y)
        input_ = torch.cat((x, y_), 1)
        output = self.model(input_)
        return output


class CondDCDiscriminator(nn.Module):
    def __init__(self, reduced_dim):
        super(CondDCDiscriminator, self).__init__()
        self.encoder = OneHotEncoder(torch.tensor([10]))
        # Conv2d layers
        layers = dcLayersD(reduced_dim=reduced_dim)
        self.model_conv = nn.Sequential(*layers, Flatten())
        # Linear layers
        self.model_linear = nn.Sequential(
            *linearBlock(reduced_dim+self.encoder.n_values_sum, 256, normalize=False, activation='LeakyReLU'),
            *linearBlock(256, 128, normalize=False, activation='LeakyReLU'),
            *linearBlock(128, 1, normalize=False, activation='Sigmoid'),
        )

    def forward(self, x, y):
        x_ = self.model_conv(x)
        y_ = self.encoder(y)
        input_ = torch.cat((x_, y_), 1)
        output = self.model_linear(input_)
        return output

