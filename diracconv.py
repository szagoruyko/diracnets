import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import dirac


def normalize(w):
    """Normalizes weight tensor over full filter."""
    return F.normalize(w.view(w.size(0), -1)).view_as(w)


class DiracConv(nn.Module):

    def init_params(self, out_channels):
        self.alpha = nn.Parameter(torch.Tensor(out_channels).fill_(1))
        self.beta = nn.Parameter(torch.Tensor(out_channels).fill_(0.1))
        self.register_buffer('delta', dirac(self.weight.data.clone()))
        assert self.delta.size() == self.weight.size()
        self.v = (-1,) + (1,) * (self.weight.dim() - 1)

    def transform_weight(self):
        return self.alpha.view(*self.v) * Variable(self.delta) + self.beta.view(*self.v) * normalize(self.weight)


class DiracConv1d(nn.Conv1d, DiracConv):
    """Dirac parametrized convolutional layer.

    Works the same way as `nn.Conv1d`, but has additional weight parametrizatoin:
        :math:`\alpha\delta + \beta W`,
    where:
        :math:`\alpha` and :math:`\beta` are learnable scalars,
        :math:`\delta` is such a tensor so that `F.conv1d(x, delta) = x`, ie
            Kroneker delta
        `W` is weight tensor

    It is user's responsibility to set correcting padding. Only stride=1 supported.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.init_params(out_channels)

    def forward(self, input):
        return F.conv1d(input, self.transform_weight(), self.bias, self.stride, self.padding, self.dilation)


class DiracConv2d(nn.Conv2d, DiracConv):
    """Dirac parametrized convolutional layer.

    Works the same way as `nn.Conv2d`, but has additional weight parametrizatoin:
        :math:`\alpha\delta + \beta W`,
    where:
        :math:`\alpha` and :math:`\beta` are learnable scalars,
        :math:`\delta` is such a tensor so that `F.conv2d(x, delta) = x`, ie
            Kroneker delta
        `W` is weight tensor

    It is user's responsibility to set correcting padding. Only stride=1 supported.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.init_params(out_channels)

    def forward(self, input):
        return F.conv2d(input, self.transform_weight(), self.bias, self.stride, self.padding, self.dilation)


class DiracConv3d(nn.Conv3d, DiracConv):
    """Dirac parametrized convolutional layer.

    Works the same way as `nn.Conv3d`, but has additional weight parametrizatoin:
        :math:`\alpha\delta + \beta W`,
    where:
        :math:`\alpha` and :math:`\beta` are learnable scalars,
        :math:`\delta` is such a tensor so that `F.conv3d(x, delta) = x`, ie
            Kroneker delta
        `W` is weight tensor

    It is user's responsibility to set correcting padding. Only stride=1 supported.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.init_params(out_channels)

    def forward(self, input):
        return F.conv3d(input, self.transform_weight(), self.bias, self.stride, self.padding, self.dilation)
