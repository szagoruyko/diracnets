import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import dirac


def dirac_delta(in_channels, out_channels, k):
    ni, no = in_channels, out_channels
    n = min(ni, no)
    size = (n, n) + k
    repeats = (max(no // ni, 1), max(ni // no, 1)) + (1,) * len(k)
    return dirac(torch.Tensor(*size)).repeat(*repeats)


class DiracConv1d(nn.Conv1d):
    """Dirac parametrized convolutional layer.

    Works the same way as `nn.Conv1d`, but has additional weight parametrizatoin:
        :math:`\alpha\delta + \beta W`,
    where:
        :math:`\alpha` and :math:`\beta` are learnable scalars,
        :math:`\delta` is such a tensor so that `F.conv1d(x, delta) = x`, ie
            Kroneker delta
        `W` is weight tensor
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DiracConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.alpha = nn.Parameter(torch.Tensor([5]))
        self.beta = nn.Parameter(torch.Tensor([1e-5]))
        self.register_buffer('delta', dirac_delta(in_channels, out_channels, self.weight.size()[2:]))
        assert self.delta.size() == self.weight.size()

    def forward(self, input):
        alpha = self.alpha.expand_as(self.weight)
        beta = self.beta.expand_as(self.weight)
        return F.conv1d(input, alpha * Variable(self.delta) + beta * self.weight, self.bias, self.stride,
                        self.padding, self.dilation)


class DiracConv2d(nn.Conv2d):
    """Dirac parametrized convolutional layer.

    Works the same way as `nn.Conv2d`, but has additional weight parametrizatoin:
        :math:`\alpha\delta + \beta W`,
    where:
        :math:`\alpha` and :math:`\beta` are learnable scalars,
        :math:`\delta` is such a tensor so that `F.conv2d(x, delta) = x`, ie
            Kroneker delta
        `W` is weight tensor
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DiracConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.alpha = nn.Parameter(torch.Tensor([5]))
        self.beta = nn.Parameter(torch.Tensor([1e-5]))
        self.register_buffer('delta', dirac_delta(in_channels, out_channels, self.weight.size()[2:]))
        assert self.delta.size() == self.weight.size()

    def forward(self, input):
        alpha = self.alpha.expand_as(self.weight)
        beta = self.beta.expand_as(self.weight)
        return F.conv2d(input, alpha * Variable(self.delta) + beta * self.weight, self.bias, self.stride,
                        self.padding, self.dilation)


class DiracConv3d(nn.Conv3d):
    """Dirac parametrized convolutional layer.

    Works the same way as `nn.Conv3d`, but has additional weight parametrizatoin:
        :math:`\alpha\delta + \beta W`,
    where:
        :math:`\alpha` and :math:`\beta` are learnable scalars,
        :math:`\delta` is such a tensor so that `F.conv2d(x, delta) = x`, ie
            Kroneker delta
        `W` is weight tensor
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DiracConv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.alpha = nn.Parameter(torch.Tensor([5]))
        self.beta = nn.Parameter(torch.Tensor([1e-5]))
        self.register_buffer('delta', dirac_delta(in_channels, out_channels, self.weight.size()[2:]))
        assert self.delta.size() == self.weight.size()

    def forward(self, input):
        alpha = self.alpha.expand_as(self.weight)
        beta = self.beta.expand_as(self.weight)
        return F.conv3d(input, alpha * Variable(self.delta) + beta * self.weight, self.bias, self.stride,
                        self.padding, self.dilation)
