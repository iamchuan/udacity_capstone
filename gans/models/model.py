from ..utils.layers import *
from torch import nn


def linearBlock(in_features, out_features, activation, normalize=True):

    # add Linear layer
    layers = [nn.Linear(in_features=in_features, out_features=out_features)]

    # add BatchNormal layer
    if normalize:
        layers.append(nn.BatchNorm1d(num_features=out_features))

    # add Activation layer
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation.lower() == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation.lower() == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation.lower() == 'tanh':
            layers.append(nn.Tanh())
        else:
            raise ValueError('Invalid activation name!')
    elif issubclass(activation.__class__, nn.modules.activation.Module):
        layers.append(activation)
    elif activation is None:
        pass
    else:
        raise ValueError('Invalid activation object!')

    return layers


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
        ndf = 32
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


class Generator(nn.Module):
    def __init__(self, layers, img_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(*layers, Unflatten2d(img_shape=img_shape))

    def forward(self, x):
        output = self.model(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, layers):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(Flatten(), *layers)

    def forward(self, x):
        output = self.model(x)
        return output


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


class ConditionalGenerator(nn.Module):
    def __init__(self, layers, img_shape):
        super(ConditionalGenerator, self).__init__()
        self.encoder = OneHotEncoder(torch.tensor([10]))
        self.model = nn.Sequential(*layers, Unflatten2d(img_shape=img_shape))

    def forward(self, x, y):
        y_ = self.encoder(y)
        input_ = torch.cat((x, y_), 1)
        output = self.model(input_)
        return output


class ConditionalDiscriminator(nn.Module):
    def __init__(self, layers):
        super(ConditionalDiscriminator, self).__init__()
        self.flatten = Flatten()
        self.encoder = OneHotEncoder(torch.tensor([10]))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x_ = self.flatten(x)
        y_ = self.encoder(y)
        input_ = torch.cat((x_, y_), 1)
        output = self.model(input_)
        return output


class ConDCGenerator(nn.Module):
    def __init__(self, img_shape, noize_dim, n_values):
        super(ConDCGenerator, self).__init__()
        self.encoder = OneHotEncoder(n_values=n_values)
        layers = dcLayersG(img_shape=img_shape, noize_dim=noize_dim+self.encoder.n_values_sum)
        self.model = nn.Sequential(Unflatten2d(img_shape=(noize_dim+self.encoder.n_values_sum, 1, 1)), *layers)

    def forward(self, x, y):
        y_ = self.encoder(y)
        input_ = torch.cat((x, y_), 1)
        output = self.model(input_)
        return output


class CDCDiscriminator(nn.Module):
    def __init__(self, layers, n_values):
        super(CDCDiscriminator, self).__init__()
        self.encoder = OneHotEncoder(n_values=n_values)
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x_ = self.flatten(x)
        y_ = self.encoder(y)
        input_ = torch.cat((x_, y_), 1)
        output = self.model(input_)
        return output


# class ConDCDiscriminator(nn.Module):
#     def __init__(self, img_shape):
#         super(ConDCDiscriminator, self).__init__()
#         layers = dcLayersD(img_shape=img_shape)
#         self.model = nn.Sequential(*layers[:-2], Flatten())
#         self.model_ = nn.Sequential(
#             nn.Linear(in_features=256, out_features=128, bias=True),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Linear(in_features=128, out_features=1, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         output = self.model(x)
#         return output