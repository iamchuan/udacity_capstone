from ..utils.layers import *
from torch import nn


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

