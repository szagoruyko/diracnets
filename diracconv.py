import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import dirac


def dirac_delta(in_channels, out_channels, kh, kw):
    return dirac(torch.Tensor(out_channels, in_channels, kh, kw))


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
        self.register_buffer('delta', dirac_delta(in_channels, out_channels, self.weight.size(2), self.weight.size(3)))

    def forward(self, input):
        alpha = self.alpha.expand_as(self.weight)
        beta = self.beta.expand_as(self.weight)
        return F.conv2d(input, alpha * Variable(self.delta) + beta * self.weight, self.bias, self.stride,
                        self.padding, self.dilation)
