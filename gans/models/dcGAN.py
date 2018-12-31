from ..utils.layers import *
from torch import nn

def dcLayersG(img_shape, noize_dim):
    C, H, _ = img_shape
    if H == 28:
        ngf = 64
        layersG = [
            nn.ConvTranspose2d(noize_dim, ngf * 4, 4, 1, 0),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, C, 4, 2, 1),
            nn.Tanh()
        ]
        return layersG
    elif H == 75:
        ngf = 32
        layersG = [
            nn.ConvTranspose2d(noize_dim, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, C, 3, 2, 1),
            nn.Tanh()
        ]
        return layersG
    else:
        raise ValueError('image_shape not supported yet.')


def dcLayersD(img_shape):
    C, H, _ = img_shape
    if H == 28:
        ndf = 64
        layersD = [
            nn.Conv2d(C, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 3, 1, 0),
            nn.Sigmoid()
        ]
        return layersD
    elif H == 75:
        ndf = 64
        layersD = [
            nn.Conv2d(C, ndf, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        ]
        return layersD
    else:
        raise ValueError('image_shape not supported yet.')


class DCGenerator(nn.Module):
    def __init__(self, img_shape, noize_dim):
        super(DCGenerator, self).__init__()
        layers = dcLayersG(img_shape=img_shape, noize_dim=noize_dim)
        self.model = nn.Sequential(Unflatten2d(img_shape=(noize_dim, 1, 1)), *layers)

    def forward(self, x):
        output = self.model(x)
        return output


class DCDiscriminator(nn.Module):
    def __init__(self, img_shape):
        super(DCDiscriminator, self).__init__()
        layers = dcLayersD(img_shape=img_shape)
        self.model = nn.Sequential(*layers, Flatten())

    def forward(self, x):
        output = self.model(x)
        return output
