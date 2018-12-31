import torch
from torch import nn


class Flatten(nn.Module):
    r"""Transform a input of shape (N, C, [H, W]) to an output of shape (N, C[*H*W]).
    """
    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


class Unflatten2d(nn.Module):
    r"""Transform an input of shape (N, C*H*W) to an output of shape (N, C, H, W).
    """
    def __init__(self, img_shape):
        super(Unflatten2d, self).__init__()
        self.C, self.H, self.W = img_shape

    def forward(self, x):
        return x.view(-1, self.C, self.H, self.W)

    def extra_repr(self):
        return 'img_shape=({}, {}, {})'.format(self.C, self.H, self.W)


class OneHotEncoder(nn.Module):

    def __init__(self, n_values):
        super(OneHotEncoder, self).__init__()
        self.n_values = n_values
        self.n_values_sum = self.n_values.sum()
        self.n_values_cumsum = torch.cat([torch.tensor([0]), torch.cumsum(self.n_values[:-1], dim=0)])

    def forward(self, labels):
        device = labels.device
        if labels.dim() == 1:
            labels = labels.view(labels.size(0), -1)
        if self.n_values.size(0) != labels.size(1):
            raise ValueError('Labels dimension is not consistent with n_values dimension')

        index = labels.add(self.n_values_cumsum.to(device))
        encoded = torch.zeros(labels.size(0),
                              self.n_values_sum,
                              device=labels.device).scatter_(1, index, 1)
        return encoded

