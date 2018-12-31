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

